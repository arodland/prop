"""Anomaly model (PLAN.md Phase 4a): a conditional-neural-process-style transformer.

Observation tokens -> encoder (self-attention) ; query points -> cross-attend to encoded tokens ->
mean and log-variance of the (normalised) anomaly for fof2, hmf2, mufd. Global indices are a token too.
Deliberately small: beat the tuned kernel first, then grow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

F_TOK, F_QRY, F_GLOB, F_GLO, F_SPOT = 17, 10, 6, 14, 29  # F_TOK 17 = v1..v5's 16 + log1p(rows pooled)/3 (build_samples --pool)


class MultiheadSDPA(nn.Module):
    """nn.MultiheadAttention replacement on F.scaled_dot_product_attention. With a key_padding_mask,
    nn.MultiheadAttention leaves the fused path and materialises the (B, heads, Nq, Nk) scores in fp32; here the
    padding mask is a broadcast (B,1,1,Nk) additive bias, which the memory-efficient kernel accepts, so attention
    memory is O(N) and the encoder can take 10k+ tokens. need_weights=True takes the explicit path (diagnostics)."""

    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.h, self.dk, self.dropout = heads, d // heads, dropout
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        B, Nq, d = query.shape; Nk = key.shape[1]
        q = self.q_proj(query).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, self.h, self.dk).transpose(1, 2)
        bias = None
        if key_padding_mask is not None:
            bias = torch.zeros(B, 1, 1, Nk, dtype=q.dtype, device=q.device).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
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

    def forward(self, q, memory, memory_key_padding_mask=None, need_weights=False):
        h = self.ln1(q)
        a, w = self.attn(h, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=need_weights)  # w: (B,M,N) head-averaged
        q = q + self.drop(a)
        return q + self.drop(self.ffn(self.ln2(q))), w


class AnomalyModel(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=4, dec_layers=2, dropout=0.1, query_self_attn=False, glotec=False, spots=False, f_spot=F_SPOT, f_qry=F_QRY):
        super().__init__()
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
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=pad)
        return x, pad

    def decode(self, qry, h, pad, return_attn=False):
        """return_attn: also return cross-attention weights (B,M,N'), head- and layer-averaged, for the
        attention-mass diagnostic (analysis/attention_mass.py). Column order matches encode()'s token order."""
        q = self.qry_in(qry); ws = []
        for layer in self.decoder:
            q, w = layer(q, h, memory_key_padding_mask=pad, need_weights=return_attn); ws.append(w)
        out = self.head(q)
        if return_attn:
            return out[..., :3], out[..., 3:].clamp(-6, 4), torch.stack(ws).mean(0)
        return out[..., :3], out[..., 3:].clamp(-6, 4)


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
