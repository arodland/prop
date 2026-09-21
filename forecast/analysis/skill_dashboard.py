"""Skill dashboard for one model: foF2 RMSE by lead and by distance vs IRI / production GP / tuned kernel,
plus a source-availability summary. Writes a self-contained HTML page (inline SVG, no libraries).

    uv run analysis/skill_dashboard.py --skill /kass/forecast/eval/skill_model_v7p.json --sources sources.json --out dash.html

sources.json: {"rows": [{"label": "...", "iono": bool, "glotec": bool, "spots": bool, "holdout": +0.276, "ro": +0.229}, ...],
               "iri_holdout": 1.447, "iri_ro": 1.656}   (skill = 1 - RMSE/RMSE_iri, paired, Oct-Dec)
"""
import argparse
import json

SERIES = [("model", "v7p model", "--s1"), ("gp_prod", "production GP", "--s2"), ("anomaly_decay", "tuned kernel", "--s3"), ("iri", "IRI", "--s4")]
DIST_LABELS = ["<250", "250–500", "500–1000", "1000–2000", "2000–4000", ">4000"]


def line_chart(series, ylo, yhi, title, sub, n, cid):
    """series: list of (key, label, var, values[24]) -> svg string. x = lead hour 1..24."""
    W, H, L, R, T, B = 640, 300, 44, 96, 20, 34
    px = lambda h: L + (h - 1) / 23 * (W - L - R)
    py = lambda v: T + (yhi - v) / (yhi - ylo) * (H - T - B)
    ticks = []
    step = 0.1 if yhi - ylo <= 0.8 else 0.2
    v = ylo
    while v <= yhi + 1e-9:
        ticks.append(round(v, 2)); v += step
    g = [f'<line x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}" class="grid"/>' for t in ticks]
    yl = [f'<text x="{L - 8}" y="{py(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>' for t in ticks]
    xl = [f'<text x="{px(h):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{h}</text>' for h in (1, 6, 12, 18, 24)]
    paths, ends, dots = [], [], []
    for key, label, var, vals in series:
        pts = " ".join(f"{px(i + 1):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="2" stroke-linejoin="round"/>')
        ends.append((py(vals[-1]), label, var, vals[-1]))
        for i, v in enumerate(vals):
            dots.append(f'<circle cx="{px(i + 1):.1f}" cy="{py(v):.1f}" r="9" fill="transparent" data-l="{label}" data-h="{i + 1}" data-v="{v:.3f}"/>')
    # spread end labels so they don't collide
    ends.sort()
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < 14:
            ys[i] = ys[i - 1] + 14
    lab = [f'<text x="{W - R + 8}" y="{y + 4:.1f}" class="lab" fill="var({var})">{label} {v:.2f}</text>' for y, (_, label, var, v) in zip(ys, ends)]
    return f'''<figure class="chart" id="{cid}">
<figcaption><b>{title}</b><span>{sub} · n = {n:,}</span></figcaption>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">
{''.join(g)}{''.join(yl)}{''.join(xl)}
<text x="{L}" y="{T - 6}" class="tick">MHz</text><text x="{W - R}" y="{H - 12}" class="tick" text-anchor="end" dx="24">lead h</text>
{''.join(paths)}{''.join(lab)}{''.join(dots)}
</svg></figure>'''


def bar_chart(series, ylo, yhi, title, sub, counts, cid):
    """grouped bars by distance bin; series: list of (key, label, var, values[6] or None)."""
    W, H, L, R, T, B = 640, 300, 44, 16, 20, 46
    nb = len(DIST_LABELS); gw = (W - L - R) / nb; ns = len(series); bw = min(18, (gw - 16) / ns)
    py = lambda v: T + (yhi - v) / (yhi - ylo) * (H - T - B)
    ticks = [round(ylo + i * 0.2, 1) for i in range(int(round((yhi - ylo) / 0.2)) + 1)]
    g = [f'<line x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}" class="grid"/>' for t in ticks]
    yl = [f'<text x="{L - 8}" y="{py(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>' for t in ticks]
    xl = [f'<text x="{L + gw * (i + 0.5):.1f}" y="{H - 24}" class="tick" text-anchor="middle">{d}</text>'
          f'<text x="{L + gw * (i + 0.5):.1f}" y="{H - 10}" class="tick muted" text-anchor="middle">n {counts[i]:,}</text>' for i, d in enumerate(DIST_LABELS)]
    bars = []
    for j, (key, label, var, vals) in enumerate(series):
        for i, v in enumerate(vals):
            if v is None:
                continue
            x = L + gw * (i + 0.5) + (j - (ns - 1) / 2) * (bw + 2) - bw / 2
            bars.append(f'<rect x="{x:.1f}" y="{py(v):.1f}" width="{bw:.1f}" height="{py(ylo) - py(v):.1f}" fill="var({var})" rx="2" data-l="{label}" data-h="{DIST_LABELS[i]} km" data-v="{v:.3f}"/>')
    leg = " ".join(f'<span><i style="background:var({var})"></i>{label}</span>' for _, label, var, _ in series)
    return f'''<figure class="chart" id="{cid}">
<figcaption><b>{title}</b><span>{sub}</span></figcaption>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">
{''.join(g)}{''.join(yl)}{''.join(xl)}<text x="{L}" y="{T - 6}" class="tick">MHz</text>{''.join(bars)}
</svg><div class="legend">{leg}</div></figure>'''


def source_table(src):
    """Two skill references per row: IRI, and the model's own indices-only mode (learned climatology), which is the
    fair baseline for what an observation source adds. Bars are scaled per reference."""
    rows = []
    mx = max(max(abs(r["holdout"]), abs(r["ro"])) for r in src["rows"])
    mx2 = max(max(abs(r.get("holdout_clim") or 0), abs(r.get("ro_clim") or 0)) for r in src["rows"]) or 1
    for r in src["rows"]:
        chips = "".join(f'<i class="{"on" if r[k] else "off"}" title="{k}">{t}</i>' for k, t in (("iono", "ionosondes"), ("glotec", "GloTEC"), ("spots", "spots")))
        def bar(v, m, cls=""):
            if v is None:
                return '<div class="bar ref"><span>reference</span></div>'
            w = abs(v) / m * 100
            return f'<div class="bar {cls}"><b style="width:{w:.0f}%" class="{"neg" if v < 0 else ""}"></b><span>{v * 100:+.1f}%</span></div>'
        rows.append(f'<tr><th>{r["label"]}</th><td class="chips">{chips}</td><td>{bar(r["holdout"], mx)}</td><td>{bar(r["ro"], mx)}</td>'
                    f'<td>{bar(r.get("holdout_clim"), mx2, "clim")}</td><td>{bar(r.get("ro_clim"), mx2, "clim")}</td></tr>')
    return "".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True); ap.add_argument("--sources", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="v7p")
    a = ap.parse_args()
    j = json.load(open(a.skill)); src = json.load(open(a.sources))
    win = f"{j['window'][0]} to {j['window'][1]}"
    def ser(view, keys=SERIES):
        return [(k, lab.replace("v7p", a.name), var, j[view][k]) for k, lab, var in keys if k in j[view]]
    ho, fu, ro = j["lead_iono_holdout"], j["lead_iono_full"], j["lead_ro"]
    prim = lambda d: sum(d[k] for k in (0, 1, 2, 3, 4)) / 5  # not the bucketed primary; headline tiles use the paired skills from sources.json
    h = src["headline"]
    charts = [
        line_chart(ser("lead_iono_holdout"), 0.9, 1.5, "Held-out stations", "foF2 RMSE by lead, stations withheld from the inputs", j["lead_iono_holdout_n"], "c1"),
        line_chart(ser("lead_iono_full"), 0.7, 1.5, "All stations", "foF2 RMSE by lead; the GP is scored at its own assimilated stations", j["lead_iono_full_n"], "c2"),
        line_chart(ser("lead_ro"), 1.2, 1.9, "Radio occultation", "foF2 RMSE by lead at COSMIC-2 and PlanetiQ profiles, scored as map pixels (RO flag off), fair for every model", j["lead_ro_n"], "c3"),
        bar_chart(ser("dist_ro"), 1.0, 2.2, "Radio occultation by distance", "km to the nearest ionosonde that fed the forecast; scored as map pixels", j["dist_ro_n"], "c4"),
        bar_chart(ser("dist_iono_holdout", [s for s in SERIES if s[0] != "gp_prod"]), 0.6, 1.8, "Held-out stations by distance", "km to the nearest contributing ionosonde (GP omitted: distance 0 at its own stations)", j["dist_iono_holdout_n"], "c5"),
    ]
    html = f'''<title>{a.name} Forecast Skill</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{color-scheme:light;--bg:#F2F4F6;--surface:#FFFFFF;--ink:#1B2430;--muted:#5B6673;--rule:#D6DCE3;--grid:#E4E8EC;--tip:#1B2430;--tipink:#FFFFFF;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#1a7f4b;--bad:#c2410c;--chip:#E4E8EC}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#0E141B;--surface:#161E27;--ink:#E4E9EE;--muted:#93A0AE;--rule:#2A3541;--grid:#243040;--tip:#E4E9EE;--tipink:#0E141B;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good:#3fbf7f;--bad:#f0884d;--chip:#243040}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0E141B;--surface:#161E27;--ink:#E4E9EE;--muted:#93A0AE;--rule:#2A3541;--grid:#243040;--tip:#E4E9EE;--tipink:#0E141B;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good:#3fbf7f;--bad:#f0884d;--chip:#243040}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
main{{max-width:1360px;margin:0 auto;padding:32px 28px 56px}}
header{{display:flex;flex-wrap:wrap;gap:12px 32px;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--rule);padding-bottom:16px;margin-bottom:20px}}
h1{{font:600 30px/1.15 "Source Serif 4",Georgia,serif;margin:0;text-wrap:balance}}
header p{{margin:0;color:var(--muted);max-width:60ch}}
.eyebrow{{font:500 12px/1 "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:22px}}
.tile{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}}
.tile .v{{font:500 30px/1.1 "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;margin:6px 0 2px}}
.tile .v.good{{color:var(--good)}} .tile .v.bad{{color:var(--bad)}}
.tile small{{color:var(--muted);display:block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
.chart{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px 10px;margin:0;min-width:0}}
figcaption{{display:flex;flex-direction:column;gap:2px;margin-bottom:6px}} figcaption b{{font-weight:600}} figcaption span{{color:var(--muted);font-size:13px}}
svg{{width:100%;height:auto;display:block;font:12px "IBM Plex Sans",sans-serif}}
svg .grid{{stroke:var(--grid);stroke-width:1}} svg .tick{{fill:var(--muted);font-size:11px;font-family:"IBM Plex Mono",monospace}} svg .tick.muted{{opacity:.7}}
svg .lab{{font:500 12px "IBM Plex Sans",sans-serif}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted);padding:6px 2px 0}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
section{{margin-top:28px}} h2{{font:600 20px/1.2 "Source Serif 4",Georgia,serif;margin:0 0 4px}} section>p{{color:var(--muted);margin:0 0 12px;max-width:70ch}}
table{{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--rule);border-radius:6px;overflow:hidden}}
th,td{{text-align:left;padding:9px 12px;border-top:1px solid var(--rule);vertical-align:middle}} thead th{{border-top:0;color:var(--muted);font-weight:500;font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-family:"IBM Plex Mono",monospace}}
tbody th{{font-weight:500;white-space:nowrap}}
.chips i{{display:inline-block;font:500 11px/1 "IBM Plex Mono",monospace;font-style:normal;padding:4px 7px;border-radius:3px;margin-right:4px;background:var(--chip);color:var(--ink)}}
.chips i.off{{opacity:.35;text-decoration:line-through}}
.bar{{display:flex;align-items:center;gap:8px;min-width:180px}} .bar b{{display:block;height:10px;background:var(--s1);border-radius:2px;min-width:2px}} .bar b.neg{{background:var(--bad)}}
.bar span{{font:500 13px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
.bar.clim b{{background:var(--s3)}} .bar.ref span{{color:var(--muted);font-style:italic}}
.wrap{{overflow-x:auto}}
#tip{{position:fixed;pointer-events:none;background:var(--tip);color:var(--tipink);font:12px "IBM Plex Mono",monospace;padding:6px 8px;border-radius:4px;display:none;z-index:9}}
footer{{margin-top:32px;color:var(--muted);font-size:13px;max-width:80ch}}
@media (prefers-reduced-motion:no-preference){{.bar b{{transition:width .3s}}}}
</style>
<main>
<header>
 <div><div class="eyebrow">Ionospheric forecast · foF2 · {win}</div><h1>{a.name} Forecast Skill</h1></div>
 <p>{src["blurb"]}</p>
</header>
<div class="tiles">
 <div class="tile"><small>Held-out stations vs production GP</small><div class="v {"good" if h["holdout_gp"] > 0 else "bad"}">{h["holdout_gp"] * 100:+.1f}%</div><small>{h["holdout_gp_ci"]}</small></div>
 <div class="tile"><small>All stations vs production GP</small><div class="v {"good" if h["full_gp"] > 0 else "bad"}">{h["full_gp"] * 100:+.1f}%</div><small>{h["full_gp_ci"]}</small></div>
 <div class="tile"><small>Radio occultation vs IRI</small><div class="v {"good" if h["ro_iri"] > 0 else "bad"}">{h["ro_iri"] * 100:+.1f}%</div><small>{h["ro_iri_ci"]}</small></div>
 <div class="tile"><small>σ coverage, ±1σ / ±2σ</small><div class="v">{h["cov1"]:.0%} / {h["cov2"]:.0%}</div><small>ideal 68% / 95%; GP {h["gp_cov1"]:.0%} / {h["gp_cov2"]:.0%}</small></div>
</div>
<div class="grid">{charts[0]}{charts[1]}{charts[2]}</div>
<section><h2>Skill by distance from the nearest input</h2><p>Error growth with distance from the nearest contributing ionosonde is the spatial-extrapolation test. Hover any bar or point for the value.</p>
<div class="grid">{charts[3]}{charts[4]}</div></section>
<section><h2>What each source is worth</h2><p>{src["source_blurb"]}</p>
<div class="wrap"><table><thead><tr><th>inputs at forecast time</th><th>sources</th><th>held-out stations, vs IRI</th><th>radio occultation (map pixels), vs IRI</th><th>held-out stations, vs indices only</th><th>radio occultation (map pixels), vs indices only</th></tr></thead>
<tbody>{source_table(src)}</tbody></table></div></section>
<footer>{src["footer"]}</footer>
</main>
<div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-v]').forEach(el=>{{
 el.addEventListener('mousemove',e=>{{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=el.dataset.l+' · '+el.dataset.h+' · '+el.dataset.v+' MHz';}});
 el.addEventListener('mouseleave',()=>tip.style.display='none');
}});
</script>'''
    open(a.out, "w").write(html)
    print("wrote", a.out, len(html), "bytes")


if __name__ == "__main__":
    main()
