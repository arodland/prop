"""Muon (orthogonalized-momentum) for conv weights, AdamW for the rest.

**Training-only.** Nothing in the send/receive path imports this; an
optimizer changes how a checkpoint is *reached*, never what it means,
so unlike almost everything else in this project it has no bearing on
the on-air contract and needs no fleet coordination.

`torch.optim.Muon` is the reference implementation and this file
deliberately reproduces its arithmetic step for step -- same
Newton-Schulz (bf16, transpose-then-normalize), same nesterov momentum,
same decoupled decay at the *unadjusted* lr while the update takes the
adjusted one. `tests/test_muon.py` asserts bit-level agreement with it
on 2D parameters, which is the whole reason to mirror it rather than
improvise: torch is the oracle, and we only extend.

**What we extend is the one thing that blocks it here: torch's Muon
rejects `ndim != 2`, and this model is convolutions all the way down.**
A conv filter is `(out, in, kh, kw)`; folding the input fan into the
columns gives the `(out, in*kh*kw)` matrix Muon wants. That is the
standard treatment and is what Keller Jordan's original (MIT) does too.
Note the AGPL-3.0 port in ultralytics is *not* the only 4D-aware Muon
and must not be vendored -- this tree is Artistic-2.0, see NOTICE.

**Learning rate is the trap this file exists to defuse.** Muon's update
is unit-spectral-norm by construction rather than gradient-scaled, so
"original" scaling wants ~0.02 where AdamW wants 2e-4, and a Muon run
at AdamW's lr measures nothing at all. `adjust_lr_fn="match_rms_adamw"`
(the Kimi scaling, `0.2*sqrt(max(A, B))`) makes the update's RMS match
AdamW's, so the *same* `--lr` and the *same* cosine schedule carry over
and a comparison against an existing AdamW baseline is apples to
apples. That is why it is the default here and not torch's default.

One deliberate deviation, and it matters by ~3x on 3x3 kernels: torch
computes the adjustment from `param.shape[:2]`, which for a 4D filter
is `(out, in)` -- the wrong pair, because the matrix actually being
orthogonalized is the *reshaped* one. We pass the reshaped shape, since
the RMS argument the scaling comes from is about the matrix Muon sees.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

__all__ = ["newton_schulz", "adjust_lr", "Muon", "build_param_groups"]

# Coefficients chosen to maximize the iteration's slope at zero; the
# result is not exactly U V^T but U S' V^T with S' spread around 1,
# which is empirically fine for updates and far cheaper than an SVD.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
EPS = 1e-7


def newton_schulz(grad: torch.Tensor,
                  coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
                  steps: int = NS_STEPS, eps: float = EPS) -> torch.Tensor:
    """Orthogonalize a 2D matrix. Mirrors torch.optim's `_zeropower_via_newtonschulz`."""
    if grad.ndim != 2:
        raise ValueError("Newton-Schulz needs a 2D matrix")
    a, b, c = coefficients
    ortho = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho = ortho.T
    ortho = ortho.div(ortho.norm().clamp(min=eps))  # spectral norm <= 1
    for _ in range(steps):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if grad.size(0) > grad.size(1):
        ortho = ortho.T
    return ortho


def adjust_lr(lr: float, adjust_lr_fn: str | None,
              shape: torch.Size | tuple[int, int]) -> float:
    """LR scaling by matrix shape. `shape` is the *orthogonalized* matrix's."""
    A, B = shape[0], shape[1]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, A / B))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(A, B))
    if adjust_lr_fn == "none":
        return lr
    raise ValueError(f"unknown adjust_lr_fn {adjust_lr_fn!r}")


class Muon(Optimizer):
    """Muon on `use_muon` groups, AdamW on the rest, in a single object.

    One optimizer rather than two so `CosineAnnealingLR` and every
    `step()` / `zero_grad()` site in train.py stay exactly as they are.
    Two optimizers would mean two schedulers and several call sites to
    keep in sync, and a missed one leaves half the model unscheduled --
    which trains perfectly well and answers a different question than
    the one being asked.
    """

    def __init__(self, param_groups, lr: float = 2e-4,
                 weight_decay: float = 0.01, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = NS_STEPS,
                 ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
                 eps: float = EPS, adjust_lr_fn: str = "match_rms_adamw",
                 betas: tuple[float, float] = (0.9, 0.999),
                 adamw_eps: float = 1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps,
                        ns_coefficients=ns_coefficients, eps=eps,
                        adjust_lr_fn=adjust_lr_fn, betas=betas,
                        adamw_eps=adamw_eps, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                self._muon_group(params, group, lr, wd)
            else:
                self._adamw_group(params, group, lr, wd)
        return loss

    def _muon_group(self, params, group, lr, wd):
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            grad = p.grad
            buf.lerp_(grad, 1 - group["momentum"])
            update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
            # The 4D extension: fold the input fan into the columns.
            matrix = update.reshape(len(update), -1) if update.ndim > 2 else update
            ortho = newton_schulz(matrix, group["ns_coefficients"],
                                  group["ns_steps"], group["eps"])
            # Adjustment from the reshaped shape -- see module docstring.
            alr = adjust_lr(lr, group["adjust_lr_fn"], matrix.shape)
            p.mul_(1 - lr * wd)  # decoupled decay uses the *unadjusted* lr
            p.add_(ortho.reshape(p.shape), alpha=-alr)

    def _adamw_group(self, params, group, lr, wd):
        """Plain AdamW, matching torch.optim.AdamW step for step."""
        beta1, beta2 = group["betas"]
        for p in params:
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.lerp_(p.grad, 1 - beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            p.mul_(1 - lr * wd)
            denom = (v.sqrt() / math.sqrt(1 - beta2 ** t)).add_(group["adamw_eps"])
            p.addcdiv_(m, denom, value=-lr / (1 - beta1 ** t))


def build_param_groups(model, lr: float, weight_decay: float = 0.01,
                       adamw_lr: float | None = None) -> list[dict]:
    """Conv/transposed-conv weights to Muon, biases and GroupNorm gains to AdamW.

    A 1D parameter has no matrix structure to orthogonalize, which is
    why torch's Muon refuses it outright. With `match_rms_adamw` the two
    groups share one learning rate by design; `adamw_lr` overrides it
    for the 1D group if that ever needs decoupling.
    """
    matrices = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    others = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [
        {"params": matrices, "use_muon": True, "lr": lr,
         "weight_decay": weight_decay},
        {"params": others, "use_muon": False,
         "lr": lr if adamw_lr is None else adamw_lr,
         "weight_decay": weight_decay},
    ]
