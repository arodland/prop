"""Render the lead/distance skill JSON (analysis/skill_by_lead_distance.py) as a self-contained HTML page
with inline SVG line charts, light/dark themed.

    uv run analysis/skill_charts.py /kass/forecast/eval/skill_v4.json out.html --light "#hex,#hex,#hex,#hex" --dark "#hex,..."
Series order (fixed): model, production GP, tuned kernel, IRI.
"""
import argparse
import json

SERIES = [("model", "v4 model"), ("gp_prod", "production GP"), ("anomaly_decay", "tuned kernel"), ("iri", "IRI")]
W, H, PL, PR, PT, PB = 620, 300, 52, 110, 22, 40


def svg_lines(xs, series, xlabel, ylabel, ylim, title, xticks):
    """xs: list of x labels; series: [(key, label, [y...])]. Categorical x (evenly spaced)."""
    n = len(xs); lo, hi = ylim
    def X(i): return PL + (W - PL - PR) * (i / max(n - 1, 1))
    def Y(v): return PT + (H - PT - PB) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    out.append(f'<text x="{PL}" y="14" class="ttl">{title}</text>')
    step = 0.2 if hi - lo <= 1.2 else 0.25
    v = lo
    while v <= hi + 1e-9:
        out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/><text x="{PL-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
        v += step
    for i, xl in enumerate(xs):
        if i in xticks:
            out.append(f'<text x="{X(i):.1f}" y="{H-PB+16}" class="tick" text-anchor="middle">{xl}</text>')
    out.append(f'<text x="{(PL+W-PR)/2:.0f}" y="{H-6}" class="axis" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text transform="rotate(-90 12 {(PT+H-PB)/2:.0f})" x="12" y="{(PT+H-PB)/2:.0f}" class="axis" text-anchor="middle">{ylabel}</text>')
    labels = []
    for k, (key, label, ys) in enumerate(series):
        pts = [(X(i), Y(y)) for i, y in enumerate(ys) if y is not None]
        if not pts:
            continue
        out.append(f'<polyline class="s{k}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for (x, y), i in zip(pts, [i for i, y in enumerate(ys) if y is not None]):
            out.append(f'<circle class="s{k} m" cx="{x:.1f}" cy="{y:.1f}" r="3.5"><title>{label}, {xs[i]}: {ys[i]:.3f} MHz</title></circle>')
        labels.append((k, label, pts[-1]))
    # direct end labels, nudged apart
    labels.sort(key=lambda t: t[2][1]); last = -99
    for k, label, (x, y) in labels:
        y = max(y, last + 13); last = y
        out.append(f'<text x="{W-PR+8}" y="{y+4:.1f}" class="lab s{k}t">{label}</text>')
    out.append('</svg>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("out"); ap.add_argument("--light", required=True); ap.add_argument("--dark", required=True)
    a = ap.parse_args(); d = json.load(open(a.json))
    L = a.light.split(","); D = a.dark.split(",")
    leads = [str(h) for h in range(1, 25)]
    def ser(block, keys=SERIES): return [(k, lab, d[block][k]) for k, lab in keys if k in d[block]]
    figs = []
    figs.append(("Stations that fed the forecast (full mode)", f"{d['lead_iono_full_n']:,} rows per model", svg_lines(leads, ser("lead_iono_full"), "lead, hours", "foF2 RMSE, MHz", (0.6, 1.6), "foF2 RMSE by lead, ionosonde rows, full mode", {0, 5, 11, 17, 23})))
    figs.append(("Withheld stations (holdout)", f"{d['lead_iono_holdout_n']:,} rows per model; the GP is omitted: its non-assimilated stations are a different, selection-biased set", svg_lines(leads, ser("lead_iono_holdout", [s for s in SERIES if s[0] != "gp_prod"]), "lead, hours", "foF2 RMSE, MHz", (0.8, 1.6), "foF2 RMSE by lead, withheld ionosonde clusters", {0, 5, 11, 17, 23})))
    figs.append(("Radio-occultation points (no station is an input for anyone)", f"{d['lead_ro_n']:,} rows per model", svg_lines(leads, ser("lead_ro"), "lead, hours", "foF2 RMSE, MHz", (1.0, 2.0), "foF2 RMSE by lead, RO points", {0, 5, 11, 17, 23})))
    dl = ["<250", "250–500", "500–1000", "1000–2000", "2000–4000", ">4000"]
    figs.append(("RO points by distance to the nearest contributing ionosonde", "rows per bin: " + ", ".join(f"{n:,}" for n in d["dist_ro_n"]), svg_lines(dl, ser("dist_ro"), "distance to nearest contributing ionosonde, km", "foF2 RMSE, MHz", (1.0, 2.2), "foF2 RMSE by distance, RO points", set(range(6)))))
    figs.append(("Withheld stations by distance to the nearest contributing ionosonde", "rows per bin: " + ", ".join(f"{n:,}" for n in d["dist_iono_holdout_n"]) + "; the GP omitted as above", svg_lines(dl, ser("dist_iono_holdout", [s for s in SERIES if s[0] != "gp_prod"]), "distance to nearest contributing ionosonde, km", "foF2 RMSE, MHz", (0.6, 1.8), "foF2 RMSE by distance, withheld clusters", set(range(6)))))
    css_series = "\n".join(f".s{k}{{stroke:{L[k]};fill:none;stroke-width:2}} .s{k}.m{{fill:{L[k]};stroke:var(--surface);stroke-width:1.5}} .s{k}t{{fill:{L[k]}}}" for k in range(4))
    css_series_dark = "\n".join(f".s{k}{{stroke:{D[k]}}} .s{k}.m{{fill:{D[k]}}} .s{k}t{{fill:{D[k]}}}" for k in range(4))
    html = f"""<title>Forecast Skill by Lead and Distance</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root{{--bg:#F2F4F6;--surface:#FFFFFF;--ink:#1B2430;--muted:#5B6673;--rule:#D6DCE3;--grid:#E4E8EC}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0E141B;--surface:#161E27;--ink:#E4E9EE;--muted:#93A0AE;--rule:#2A3541;--grid:#243040}} :root:not([data-theme="light"]) {{ }} }}
:root[data-theme="dark"]{{--bg:#0E141B;--surface:#161E27;--ink:#E4E9EE;--muted:#93A0AE;--rule:#2A3541;--grid:#243040}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;margin:0}}
main{{max-width:1320px;margin:0 auto;padding:36px 20px 60px}}
h1{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:1.9rem;margin:0 0 .3rem}}
.lede{{color:var(--muted);max-width:78ch;margin:.3rem 0 1.2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:18px}}
figure{{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:14px}}
figure svg{{display:block;width:100%;height:auto}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:8px}}
figcaption b{{color:var(--ink);font-weight:500}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:.9rem;margin:0 0 14px}}
.legend span::before{{content:"";display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px;background:var(--c)}}
.ttl{{font-family:"IBM Plex Sans",sans-serif;font-size:12px;fill:var(--muted)}}
.tick{{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--muted)}}
.axis{{font-family:"IBM Plex Sans",sans-serif;font-size:11px;fill:var(--muted)}}
.lab{{font-family:"IBM Plex Sans",sans-serif;font-size:11px}}
.grid{{stroke:var(--grid);stroke-width:1}}
{css_series}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]) {{ }} {css_series_dark.replace('.s', ':root:not([data-theme="light"]) .s')}}}
{css_series_dark.replace('.s', ':root[data-theme="dark"] .s')}
</style>
<main>
<h1>Forecast Skill by Lead and Distance</h1>
<p class="lede">foF2 RMSE for the {d['model_name']} model, the production GP, the tuned anomaly kernel and IRI-2020, {d['window'][0]} to {d['window'][1]}, all out of sample for the model, paired on the same observations. Lower is better. "Contributing ionosonde" means a station with data in the 24-hour input window at that issue time and not withheld.</p>
<div class="legend">{"".join(f'<span style="--c:{L[k]}">{lab}</span>' for k,(key,lab) in enumerate(SERIES))}</div>
<div class="grid">
{"".join(f'<figure>{svg}<figcaption><b>{t}.</b> {c}</figcaption></figure>' for t, c, svg in figs)}
</div>
</main>
"""
    open(a.out, "w").write(html); print("wrote", a.out)


if __name__ == "__main__":
    main()
