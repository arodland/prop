"""Plot the hourly temporal kernel: model gain at the perturbed station (probe_impulse.py gain) against the data's
own-station coefficient (anomaly_correlation.py --temporal-out). Edit `series` to add runs. Writes eval/probes/temporal_kernel.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys

P = "/kass/forecast/eval/probes"
FRESH = "--fresh" in sys.argv  # fresh deviation: probe --recent-h 1 vs data --raw --history-control
SM, SD = ("_recent", "_fresh") if FRESH else ("", "")
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def model(f):
    d = pd.read_csv(f"{P}/{f}")
    g = d.groupby("lead").gain
    return g.mean(), 1.96 * g.std() / np.sqrt(g.count())


def data(f):
    d = pd.read_csv(f"{P}/{f}")
    g = d.groupby("lag").self_coef
    return g.mean(), 1.96 * g.std() / np.sqrt(g.count())


series = [  # (label, (mean, ci), color, dashed)
    ("E2 (attention bias, no qstate)", model(f"gain_e2{SM}.csv"), "#2a78d6", False),
    ("D (qstate)", model(f"gain_d{SM}.csv"), "#eb6834", False),
    ("Data 2009–18", data(f"temporal_data_2009-2018{SD}.csv"), "#1baf7a", True),
    ("Data 2024", data(f"temporal_data_2024{SD}.csv"), "#eda100", True),
    ("E3 (attention bias + qstate)", model(f"gain_e3{SM}.csv"), "#e87ba4", False),
] + ([("E4f (lag curve, no qstate)", model("gain_e4f_recent.csv"), "#008300", False),
      ("E5f (lag curve + qstate)", model("gain_e5f_recent.csv"), "#4a3aa7", False)] if FRESH else [])

fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURF)
ax.set_facecolor(SURF)
for x in (12, 24):
    ax.axvline(x, color=GRID, lw=1, zorder=0)
for label, (m, ci), c, dashed in series:
    x = m.index.to_numpy()
    ax.fill_between(x, m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.plot(x, m, color=c, lw=2, ls=(0, (5, 3)) if dashed else "-", marker="o", ms=3.5)
ends = sorted([(m.iloc[-1], lab) for lab, (m, _), _, _ in series])
ys = []
for v, lab in ends:  # dodge end labels at least 0.04 apart
    ys.append(max(v, ys[-1] + 0.04) if ys else v)
for (v, lab), y in zip(ends, ys):
    ax.annotate(lab, (24, v), xytext=(24.4, y), textcoords="data", va="center", color=INK2, fontsize=9, annotation_clip=False)
ax.set_xlim(0, 24); ax.set_ylim(-0.06 if FRESH else 0, 1.02); ax.axhline(0, color=INK2, lw=0.8, zorder=1); ax.set_xticks(range(0, 25, 3))
ax.set_xlabel("hours after the observation (model: forecast lead; data: lag)", color=INK2)
ax.set_ylabel("weight on the station's own observation", color=INK2)
fig.text(0.07, 0.955, ("How long a fresh foF2 deviation keeps its weight at that station" if FRESH else "How long a station's foF2 anomaly keeps its weight at that station"), color=INK, fontsize=12)
fig.text(0.07, 0.885, ("Model: gain at the station's location after a +0.5 MHz nudge to its last hour of tokens only (800 impulses, Oct–Dec 2025). "
        "Data: the station's own coefficient\non its current-hour raw anomaly, predicting h hours later from all stations, holding its previous-24 h mean fixed (ridge). Bands: 95% CI."
        if FRESH else "Model: gain at the station's location after a +0.5 MHz nudge to its tokens (800 impulses, Oct–Dec 2025). "
        "Data: the station's own coefficient\npredicting its anomaly h hours later from all stations' anomalies (ridge), "
        "station month×local-hour means removed. Bands: 95% CI."), fontsize=8, color=INK2, va="bottom")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
handles = [plt.Line2D([], [], color=c, lw=2, ls=(0, (5, 3)) if d else "-") for _, _, c, d in series]
ax.legend(handles, [s[0] for s in series], frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)
fig.subplots_adjust(right=0.80, top=0.84)
fig.savefig(f"{P}/temporal_kernel{'_fresh' if FRESH else ''}.png", dpi=130)
for label, (m, _), _, _ in series:
    print(f"{label:32s}", " ".join(f"{v:.2f}" for v in m.to_numpy()[[0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24]]))
