"""Anomaly model (PLAN.md Phase 4a): a conditional-neural-process-style transformer.

Observation tokens -> encoder (self-attention) ; query points -> cross-attend to encoded tokens ->
mean and log-variance of the (normalised) anomaly for fof2, hmf2, mufd. Global indices are a token too.
Deliberately small: beat the tuned kernel first, then grow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

F_TOK, F_QRY, F_GLOB, F_GLO, F_SPOT = 17, 10, 6, 14, 29
LAG_MAX = 48  # hours of query-token time lag covered by lag_bias (lead <= 24 h, tokens <= 24 h old)  # F_TOK 17 = v1..v5's 16 + log1p(rows pooled)/3 (build_samples --pool)


class MultiheadSDPA(nn.Module):
    """nn.MultiheadAttention replacement on F.scaled_dot_product_attention. With a key_padding_mask,
    nn.MultiheadAttention leaves the fused path and materialises the (B, heads, Nq, Nk) scores in fp32; here the
    padding mask is a broadcast (B,1,1,Nk) additive bias, which the memory-efficient kernel accepts, so attention
    memory is O(N) and the encoder can take 10k+ tokens. need_weights=True takes the explicit path (diagnostics)."""

    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.h, self.dk, self.dropout = heads, d // heads, dropout
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_bias=None):
        B, Nq, d = query.shape; Nk = key.shape[1]
        q = self.q_proj(query).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, self.h, self.dk).transpose(1, 2)
        bias = None
        if key_padding_mask is not None:
            bias = torch.zeros(B, 1, 1, Nk, dtype=q.dtype, device=q.device).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if attn_bias is not None:  # (B, heads, Nq, Nk) additive logit bias (AnomalyModel rel_bias)
            bias = attn_bias if bias is None else attn_bias + bias
        if need_weights:
            w = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
            if bias is not None:
                w = w + bias
            w = w.softmax(-1)
            out = w @ v; w = w.mean(1)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0); w = None
        return self.out_proj(out.transpose(1, 2).reshape(B, Nq, d)), w


class EncLayer(nn.Module):
    """Pre-LN TransformerEncoderLayer with MultiheadSDPA; parameter names match nn.TransformerEncoderLayer except
    self_attn.in_proj_* which the load hook splits into q/k/v."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.self_attn = MultiheadSDPA(d, heads, dropout)
        self.linear1 = nn.Linear(d, 4 * d); self.linear2 = nn.Linear(4 * d, d)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout); self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.dropout1(self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask)[0])
        return x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x))))))  # relu = nn.TransformerEncoderLayer's default (v1..v5 checkpoints)


class CrossAttnLayer(nn.Module):
    """Query -> memory cross-attention + FFN, pre-LN, and *no* self-attention among queries: each query's
    answer depends only on the encoded observations, so the output is independent of how queries are
    batched or chunked (a proper conditional neural process). The v0 model used nn.TransformerDecoderLayer,
    whose query self-attention made grid maps chunk-dependent (latitude banding)."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = MultiheadSDPA(d, heads, dropout)
        self.ln2 = nn.LayerNorm(d); self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memory, memory_key_padding_mask=None, need_weights=False, attn_bias=None):
        h = self.ln1(q)
        a, w = self.attn(h, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=need_weights, attn_bias=attn_bias)  # w: (B,M,N) head-averaged
        q = q + self.drop(a)
        return q + self.drop(self.ffn(self.ln2(q))), w


class AnomalyModel(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=4, dec_layers=2, dropout=0.1, query_self_attn=False, glotec=False, spots=False, f_spot=F_SPOT, f_qry=F_QRY, rel_bias=False, lag_bias=False, lag_fourier=0):
        super().__init__()
        # rel_bias: per decoder layer and head, a learned additive bias on the query->ionosonde-token attention logits,
        # linear in [distance/1000 km, (distance/1000 km)^2, dt/24 h, cos(2pi dt/24 h), cos(2pi dt/12 h)] with dt = query
        # time - token time. Gives attention a direct notion of near/far and of same-local-time-yesterday (the GP's
        # +24 h / +12 h temporal peaks) instead of leaving locality to dot products of absolute coordinates; zeros at
        # init, so the model starts as the unbiased one. Other token kinds (null, global, GloTEC, spots) get no bias.
        self.heads = heads
        self.rel_bias = nn.Parameter(torch.zeros(dec_layers, 5, heads)) if (rel_bias or lag_bias or lag_fourier) else None
        # lag_bias: replaces rel_bias's three smooth time terms with a free table per layer and head over time lag,
        # one value per hour 0..LAG_MAX h, linearly interpolated, so a sharp dip and a 24 h peak are both expressible.
        self.lag_table = nn.Parameter(torch.zeros(dec_layers, heads, LAG_MAX + 1)) if lag_bias else None
        # lag_fourier=K: the same job as lag_table, as K harmonics of a LAG_MAX-hour period. lag = query lead - token
        # time, so cos/sin of the lag split into a query factor times a token factor and the bias is a (B, heads, M, 2K)
        # x (B, N, 2K) matmul: no per-pair gather, whose backward scatter-add into 49 bins doubled E4's step time.
        # K=16 resolves features down to a 3 h period. Coefficients per layer and head: [a_1..a_K, b_1..b_K].
        self.lag_fourier = nn.Parameter(torch.zeros(dec_layers, heads, 2 * lag_fourier)) if lag_fourier else None
        self.f_qry = f_qry  # 10, or 18 with the nearest-station state features (build_samples --qstate)
        self.f_spot = f_spot  # 29 = midpoint-only tokens (v5); 52 = midpoint + control-point slots (see spot_tokens.py)
        self.tok_in = nn.Sequential(nn.Linear(F_TOK, d), nn.GELU(), nn.Linear(d, d))
        self.glotec, self.spots = glotec, spots
        self._register_load_state_dict_pre_hook(self._compat)  # older checkpoints: see _compat
        if glotec or spots:  # Phase 4b: extra token kinds, own projections, plus kind embeddings so the encoder knows the source
            self.kind_emb = nn.Parameter(torch.zeros(3, d))
        if glotec:
            self.glo_in = nn.Sequential(nn.Linear(F_GLO, d), nn.GELU(), nn.Linear(d, d))
        if spots:  # WSPR and FT8 share this projection; the token carries a source one-hot
            self.spot_in = nn.Sequential(nn.Linear(f_spot, d), nn.GELU(), nn.Linear(d, d))
        self.glob_in = nn.Sequential(nn.Linear(F_GLOB, d), nn.GELU(), nn.Linear(d, d))
        self.qry_in = nn.Sequential(nn.Linear(f_qry, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.ModuleList([EncLayer(d, heads, dropout) for _ in range(enc_layers)])
        if query_self_attn:  # v0 compatibility only
            self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True) for _ in range(dec_layers)])
        else:
            self.decoder = nn.ModuleList([CrossAttnLayer(d, heads, dropout) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))  # 3 means + 3 log-vars
        self.null = nn.Parameter(torch.zeros(1, 1, d))  # always-present token: with no observations the model falls back to it

    @classmethod
    def from_args(cls, args):
        """The model a checkpoint's saved args describe (defaults cover older checkpoints)."""
        return cls(d=args["d"], enc_layers=args["layers"], query_self_attn=args.get("query_self_attn", True), glotec=args.get("glotec", False),
                   spots=args.get("spots", False), f_spot=args.get("f_spot", 29), f_qry=args.get("f_qry", 10), rel_bias=args.get("rel_bias", False),
                   lag_bias=args.get("lag_bias", False), lag_fourier=args.get("lag_fourier", 0) or 0)

    def _compat(self, sd, prefix, *_):
        """Load v1..v5 checkpoints: kind_emb had 2 rows (no spot kind); tok_in took 16 features (the pooled-count
        column gets zero weight, so unpooled tokens give identical output); nn.MultiheadAttention's fused in_proj
        becomes q/k/v projections; nn.TransformerEncoder stored layers under encoder.layers.i."""
        k = prefix + "kind_emb"
        if k in sd and sd[k].shape[0] < self.kind_emb.shape[0]:
            sd[k] = torch.cat([sd[k], self.kind_emb.detach()[sd[k].shape[0]:].to(sd[k])])
        k = prefix + "tok_in.0.weight"
        if k in sd and sd[k].shape[1] < F_TOK:
            sd[k] = torch.cat([sd[k], sd[k].new_zeros(sd[k].shape[0], F_TOK - sd[k].shape[1])], 1)
        for k in [k for k in sd if k.startswith(prefix + "encoder.layers.")]:
            sd[k.replace("encoder.layers.", "encoder.", 1)] = sd.pop(k)
        for k in [k for k in sd if k.endswith("in_proj_weight")]:
            base = k[: -len("in_proj_weight")]
            for name, w, b in zip(("q_proj", "k_proj", "v_proj"), sd.pop(k).chunk(3), sd.pop(base + "in_proj_bias").chunk(3)):
                sd[base + name + ".weight"] = w.contiguous(); sd[base + name + ".bias"] = b.contiguous()

    def forward(self, tok, tok_mask, glob, qry, tok_g=None, g_mask=None, tok_s=None, s_mask=None):
        """tok (B,N,F_TOK), tok_mask (B,N) True=valid, glob (B,F_GLOB), qry (B,M,F_QRY),
        optional tok_g (B,G,F_GLO) / g_mask, tok_s (B,S,F_SPOT) / s_mask -> mean (B,M,3), logvar (B,M,3)"""
        h, pad = self.encode(tok, tok_mask, glob, tok_g, g_mask, tok_s, s_mask)
        return self.decode(qry, h, pad)

    def encode(self, tok, tok_mask, glob, tok_g=None, g_mask=None, tok_s=None, s_mask=None):
        """Observation tokens -> memory (B,N',d) and its padding mask; run once, then decode() any number of query chunks."""
        B = tok.shape[0]
        extra = self.glotec or self.spots
        parts = [self.null.expand(B, 1, -1), self.glob_in(glob)[:, None], self.tok_in(tok) + (self.kind_emb[0] if extra else 0)]
        pads = [torch.zeros(B, 2, dtype=torch.bool, device=tok.device), ~tok_mask]
        if self.glotec and tok_g is not None and tok_g.shape[1] > 0:
            parts.append(self.glo_in(tok_g) + self.kind_emb[1]); pads.append(~g_mask)
        if self.spots and tok_s is not None and tok_s.shape[1] > 0:
            parts.append(self.spot_in(tok_s) + self.kind_emb[2]); pads.append(~s_mask)
        x = torch.cat(parts, 1); pad = torch.cat(pads, 1)
        # decode() reads the ionosonde tokens' geometry for rel_bias: unit vectors and times, plus where they sit in memory
        self._geo = (_unit(tok), tok[..., 5] * 24.0, 2, x.shape[1]) if self.rel_bias is not None else None
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=pad)
        return x, pad

    def _attn_bias(self, qry):
        """Per decoder layer, (B, heads, M, N') logit bias: rel_bias terms on the ionosonde-token columns, 0 elsewhere."""
        tu, tt, start, n_all = self._geo
        f = _rel_feats(qry, self._geo)  # (B, M, N, 5)
        if self.lag_fourier is not None:  # time handled by the harmonics: separable query and token factors
            K = self.lag_fourier.shape[2] // 2; om = 2 * torch.pi / LAG_MAX * torch.arange(1, K + 1, device=qry.device, dtype=qry.dtype)
            Lq = qry[..., 5][..., None] * 24.0 * om; Tt = tt[..., None] * om  # (B, M, K), (B, N, K)
            cL, sL, Kmat = Lq.cos(), Lq.sin(), torch.cat([Tt.cos(), Tt.sin()], -1)
            f = torch.cat([f[..., :2], torch.zeros_like(f[..., 2:])], -1)
        if self.lag_table is not None:  # time handled by the table: lag in hours from the dt/24 column, interpolated
            lag = (f[..., 2] * 24.0).clamp(0, LAG_MAX); i0 = lag.floor().long().clamp(max=LAG_MAX - 1); w = (lag - i0)[:, None]
            f = torch.cat([f[..., :2], torch.zeros_like(f[..., 2:])], -1)
        out = []
        for l in range(self.rel_bias.shape[0]):
            b = torch.einsum("bmnf,fh->bhmn", f, self.rel_bias[l])
            if self.lag_fourier is not None:
                a_, b_ = self.lag_fourier[l, :, :K], self.lag_fourier[l, :, K:]  # (heads, K)
                Q = torch.cat([cL[:, None] * a_[None, :, None] + sL[:, None] * b_[None, :, None],
                               sL[:, None] * a_[None, :, None] - cL[:, None] * b_[None, :, None]], -1)  # (B, heads, M, 2K)
                b = b + torch.einsum("bhmf,bnf->bhmn", Q, Kmat)
            if self.lag_table is not None:
                t = self.lag_table[l]  # (heads, LAG_MAX + 1)
                b = b + t[:, i0].permute(1, 0, 2, 3) * (1 - w) + t[:, i0 + 1].permute(1, 0, 2, 3) * w  # (B, heads, M, N)
            full = b.new_zeros(b.shape[0], self.heads, b.shape[2], n_all); full[..., start:start + b.shape[3]] = b
            out.append(full)
        return out

    def decode(self, qry, h, pad, return_attn=False):
        """return_attn: also return cross-attention weights (B,M,N'), head- and layer-averaged, for the
        attention-mass diagnostic (analysis/attention_mass.py). Column order matches encode()'s token order."""
        qry = qry[..., :self.qry_in[0].in_features]  # samples may carry query-state columns a no-qstate model ignores
        biases = self._attn_bias(qry) if self.rel_bias is not None else [None] * len(self.decoder)
        q = self.qry_in(qry); ws = []
        for layer, b in zip(self.decoder, biases):
            q, w = layer(q, h, memory_key_padding_mask=pad, need_weights=return_attn, attn_bias=b); ws.append(w)
        out = self.head(q)
        if return_attn:
            return out[..., :3], out[..., 3:].clamp(-6, 4), torch.stack(ws).mean(0)
        return out[..., :3], out[..., 3:].clamp(-6, 4)


def _unit(x):
    """(..., >=3) features [lat/90, sin lon, cos lon, ...] -> (..., 3) unit vectors."""
    lat = x[..., 0] * (torch.pi / 2)
    return torch.stack([torch.cos(lat) * x[..., 2], torch.cos(lat) * x[..., 1], torch.sin(lat)], -1)


def _rel_feats(qry, geo):
    """(B, M, N, 5) pair features between queries and ionosonde tokens: chord distance/1000 km, its square,
    dt/24 h, cos(2pi dt/24 h), cos(2pi dt/12 h); dt = query lead - token age offset (hours, >= 0)."""
    tu, tt, _, _ = geo
    dist = 6.371 * (_unit(qry)[:, :, None] - tu[:, None]).norm(dim=-1)  # chord, 1000 km units (~arc below ~3000 km)
    dt = qry[..., 5][:, :, None] * 24.0 - tt[:, None]
    w = 2 * torch.pi * dt
    return torch.stack([dist, dist ** 2, dt / 24.0, torch.cos(w / 24.0), torch.cos(w / 12.0)], -1)


def gaussian_nll(mean, logvar, target, weight=None):
    """Masked Gaussian NLL over finite targets. Returns scalar loss and per-variable RMSE (for logging)."""
    m = torch.isfinite(target)
    t = torch.where(m, target, torch.zeros_like(target))
    nll = 0.5 * (logvar + (t - mean) ** 2 / logvar.exp())
    if weight is not None:
        nll = nll * weight[..., None]
    loss = (nll * m).sum() / m.sum().clamp(min=1)
    rmse = (((t - mean) ** 2 * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp(min=1)).sqrt()
    return loss, rmse.detach()


if __name__ == "__main__":
    # rel_bias at its zero init must not change the output; a no-qstate model must take 18-column query rows;
    # the bias must receive gradient once the model is trained.
    torch.manual_seed(0)
    B, N, S, M = 2, 50, 30, 64
    tok = torch.randn(B, N, F_TOK); tok[..., 0] = torch.rand(B, N) * 2 - 1; lon = torch.rand(B, N) * 6.28
    tok[..., 1], tok[..., 2], tok[..., 5] = lon.sin(), lon.cos(), -torch.rand(B, N)
    qry = torch.randn(B, M, 18); ql = torch.rand(B, M) * 6.28; qry[..., 0] = torch.rand(B, M) * 2 - 1; qry[..., 1], qry[..., 2], qry[..., 5] = ql.sin(), ql.cos(), torch.rand(B, M)
    args = dict(tok=tok, tok_mask=torch.ones(B, N, dtype=torch.bool), glob=torch.randn(B, F_GLOB), tok_s=torch.randn(B, S, 52), s_mask=torch.ones(B, S, dtype=torch.bool))
    a = AnomalyModel(spots=True, f_spot=52, f_qry=18).eval(); b = AnomalyModel(spots=True, f_spot=52, f_qry=18, rel_bias=True).eval()
    b.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(a(qry=qry, **args)[0], b(qry=qry, **args)[0], atol=1e-6), "zero rel_bias changed the output"
    c = AnomalyModel(spots=True, f_spot=52, f_qry=10, rel_bias=True)
    m, lv = c(qry=qry, **args); assert m.shape == (B, M, 3)
    with torch.no_grad():
        c.rel_bias[:, 0] = -2.0  # distance-penalised heads: attention should concentrate on the nearest tokens
    c(qry=qry, **args)[0].sum().backward()
    assert c.rel_bias.grad is not None and c.rel_bias.grad.abs().sum() > 0, "no gradient reaches rel_bias"
    # lag_bias: zero table is a no-op, and training reaches it.
    d_ = AnomalyModel(spots=True, f_spot=52, f_qry=18, lag_bias=True).eval(); d_.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(a(qry=qry, **args)[0], d_(qry=qry, **args)[0], atol=1e-6), "zero lag table changed the output"
    d_.train(); d_(qry=qry, **args)[0].sum().backward()
    assert (d_.lag_table.grad.abs().sum((0, 1)) > 0).sum() > 40, "lag table should get gradient across the lags present (0..48 h here)"
    # A table that is large at 24 h and very negative elsewhere must make a query attend to the 24 h-lag token.
    e = AnomalyModel(f_qry=10, lag_bias=True).eval(); tk = torch.zeros(1, 3, F_TOK); tk[..., 2] = 1.0; tk[0, :, 5] = torch.tensor([-0.5, -1.0, -0.25]) * 1.0
    qq = torch.zeros(1, 1, 10); qq[..., 2] = 1.0; qq[..., 5] = 0.0  # lead 0, same place: lags 12, 24 and 6 h
    with torch.no_grad():
        e.lag_table.fill_(-30.0); e.lag_table[:, :, 24] = 30.0
        h, pad = e.encode(tk, torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, F_GLOB))
        w = e.decode(qq, h, pad, return_attn=True)[2][0, 0, 2:]  # attention over the 3 tokens (after null, global)
    assert w.argmax() == 1 and w[1] > 0.9, w
    # lag_fourier: zero is a no-op; the separable form equals sum_k a_k cos(w_k lag) + b_k sin(w_k lag) evaluated
    # directly; a 24 h-peaked set of coefficients selects the 24 h-old token.
    f_ = AnomalyModel(spots=True, f_spot=52, f_qry=18, lag_fourier=16).eval(); f_.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(a(qry=qry, **args)[0], f_(qry=qry, **args)[0], atol=1e-6), "zero lag_fourier changed the output"
        f_.lag_fourier.normal_(); f_.rel_bias.normal_()
        f_.encode(tok, args["tok_mask"], args["glob"], None, None, args["tok_s"], args["s_mask"])
        got = f_._attn_bias(qry[..., :18])[1][:, :, :, 2:2 + N]  # layer 1, ionosonde columns
        lag = qry[..., 5][:, :, None] * 24.0 - tok[..., 5][:, None] * 24.0
        om = 2 * torch.pi / LAG_MAX * torch.arange(1, 17)
        a1, b1 = f_.lag_fourier[1, :, :16], f_.lag_fourier[1, :, 16:]
        want = torch.einsum("bmnk,hk->bhmn", (lag[..., None] * om).cos(), a1) + torch.einsum("bmnk,hk->bhmn", (lag[..., None] * om).sin(), b1)
        dist_part = torch.einsum("bmnf,fh->bhmn", torch.cat([_rel_feats(qry, f_._geo)[..., :2], torch.zeros(B, M, N, 3)], -1), f_.rel_bias[1])
        assert torch.allclose(got, want + dist_part, atol=1e-3), (got - want - dist_part).abs().max()
    e = AnomalyModel(f_qry=10, lag_fourier=16).eval()
    with torch.no_grad():
        om = 2 * torch.pi / LAG_MAX * torch.arange(1, 17)
        e.lag_fourier[:, :, :16] = 3.0 * (om * 24).cos(); e.lag_fourier[:, :, 16:] = 3.0 * (om * 24).sin()  # 3 * sum_k cos(w_k (lag - 24))
        h, pad = e.encode(tk, torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, F_GLOB))
        w = e.decode(qq, h, pad, return_attn=True)[2][0, 0, 2:]
    assert w.argmax() == 1 and w[1] > 0.9, w
    print("ok: zero bias is a no-op, 18-column queries accepted by a f_qry=10 model, rel_bias / lag_table / lag_fourier behave")
