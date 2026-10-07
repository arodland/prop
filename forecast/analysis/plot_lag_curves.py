"""Learned time-lag preference per attention head, from --lag-fourier checkpoints (E4f, E5f). Writes eval/probes/lag_curves_e4f_e5f.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
HEAD = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LAG_MAX = 48
lag = np.linspace(0, 48, 481)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, facecolor=SURF)
for r, (name, title) in enumerate([("mids_d_e4f", "E4f (no qstate)"), ("mids_d_e5f", "E5f (qstate)")]):
    sd = torch.load(f"/kass/forecast/models/{name}/best.pt", map_location="cpu", weights_only=False)["model"]
    C = sd["lag_fourier"].numpy(); R = sd["rel_bias"].numpy(); K = C.shape[2] // 2
    om = 2 * np.pi / LAG_MAX * np.arange(1, K + 1)
    for layer in range(C.shape[0]):
        ax = axes[r, layer]; ax.set_facecolor(SURF)
        for h in range(C.shape[1]):
            f = np.cos(np.outer(lag, om)) @ C[layer, h, :K] + np.sin(np.outer(lag, om)) @ C[layer, h, K:]
            f = f - f[0]
            ax.plot(lag, f, color=HEAD[h], lw=2, label=f"head {h}  ({R[layer, 0, h]:+.1f} per 1000 km)")
        for x in (12, 24, 36):
            ax.axvline(x, color=GRID, lw=1, zorder=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{title}, decoder layer {layer + 1}", color=INK, fontsize=10, loc="left", pad=34)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
        ax.set_xticks(range(0, 49, 6)); ax.tick_params(colors=INK2); ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
for ax in axes[1]:
    ax.set_xlabel("time lag, query time - token time (h)", color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel("attention logit bias, relative to lag 0", color=INK2)
fig.text(0.06, 0.965, "Learned time-lag preference of each attention head", color=INK, fontsize=12)
fig.text(0.06, 0.935, "Higher = the head attends more to ionosonde tokens at that lag. Distance coefficient in the legend: "
         "strongly negative = a local head.", color=INK2, fontsize=8)
fig.text(0.06, 0.012, "Curves wrap at 48 h (harmonics of a 48 h period), so the last few hours return to the lag-0 value; lag 48 h only occurs at lead 24 h with 24 h-old tokens.", color=INK2, fontsize=8)
fig.subplots_adjust(top=0.84, bottom=0.1, hspace=0.55, wspace=0.18, right=0.96, left=0.07)
fig.savefig("/kass/forecast/eval/probes/lag_curves_e4f_e5f.png", dpi=125)
for name in ("mids_d_e4f", "mids_d_e5f"):
    sd = torch.load(f"/kass/forecast/models/{name}/best.pt", map_location="cpu", weights_only=False)["model"]
    print(name, "distance coef per head, layer 2:", np.round(sd["rel_bias"].numpy()[1, 0], 2), " d^2:", np.round(sd["rel_bias"].numpy()[1, 1], 2))
