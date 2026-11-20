"""Build a self-describing HTML report from out/benchmark.json and out/gallery/manifest.json.

    uv run python scripts/gallery_page.py [--gallery out/gallery] [--bench out/benchmark.json]

Writes <gallery>/report.html; images are referenced relative to it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

CSS = """
/* layout: single reading column, wide figures may scroll inside their own frame */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --fg: #17202b; --muted: #5b6676;
  --line: #d9dee6; --accent: #1f6f8b; --good: #2e7d4f; --bad: #b4442c;
  --display: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11161d; --surface: #182029; --fg: #e3e8ef; --muted: #98a3b3;
  --line: #2b3542; --accent: #6cc0dc; --good: #6fcf97; --bad: #f08a6c; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11161d; --surface: #182029; --fg: #e3e8ef; --muted: #98a3b3;
  --line: #2b3542; --accent: #6cc0dc; --good: #6fcf97; --bad: #f08a6c; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.2; }
h1 { font-size: 2rem; font-weight: 600; }
h2 { font-size: 1.4rem; font-weight: 600; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; font-weight: 600; color: var(--muted); }
p { margin: 0; max-width: 72ch; }
.lede { color: var(--muted); font-size: 1.05rem; }
code, .mono { font-family: var(--mono); font-size: 0.9em; }
.eyebrow { font: 600 0.75rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); }
nav { display: flex; flex-wrap: wrap; gap: 8px; }
nav a { color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 12px; font-size: 0.9rem; background: var(--surface); }
nav a:hover, nav a:focus-visible { border-color: var(--accent); outline: none; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px;
  background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
  font-size: 0.88rem; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
th { font-family: var(--mono); font-weight: 600; font-size: 0.78rem; color: var(--muted);
  background: var(--bg); position: sticky; top: 0; }
td:first-child, th:first-child, td.l, th.l { text-align: left; }
tr.wf td { font-weight: 600; }
.good { color: var(--good); } .bad { color: var(--bad); }
section { display: grid; gap: 14px; }
.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px 24px; }
.legend div { min-width: 0; }
.legend b { font-family: var(--mono); font-size: 0.85rem; }
figure { margin: 0; display: grid; gap: 6px; }
figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px;
  background: #fff; }
figcaption { color: var(--muted); font-size: 0.88rem; max-width: 90ch; }
.facts { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); font-size: 0.88rem; }
.facts span b { color: var(--fg); font-weight: 600; }
"""

CAPTIONS = {
    "warp": "Top: the test image, pulled back through the raw field and each decoded field "
    "(compression ratio under the name). Middle: zoom where a crack crosses a chunk face, chunk "
    "faces in yellow. Bottom: discontinuity labels on the raw-warped image, and each codec's "
    "warped image minus the raw-field warp on one colour scale (frame ticks mark chunk faces).",
    "field": "Displacement as hue/brightness and as vectors; ground-truth discontinuities with "
    "the samples the encoder promoted to tight or verbatim; det J of the raw field (det J &lt; 0 "
    "outlined), the change in det J after compression with any violations circled, and |∇u|.",
    "error": "|decoded − raw| per codec on a 0–ε scale; the error along one row crossing cracks "
    "and chunk faces; and the seam detector — mean neighbour-to-neighbour error jump over the "
    "whole array at each x. A systematic seam is a spike at every dashed line.",
    "stack": "The field resliced across the stack/time axis: x-displacement, discontinuity "
    "labels (misaligned sections are whole rows), and warpfield's error.",
    "roundtrip": "Image warped forward through the decoded field and back through the inverse "
    "computed from it. Bright pixels in the third panel are where no inverse exists (crack gaps "
    "and fold overlaps); the fourth uses the raw forward field and is bounded by the third "
    "plus ε.",
}


def kind(fname: str) -> str:
    return fname.split("_")[0].split(".")[0]


def fmt_ratio(v):
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


def zfp_matched(sweep: list[dict], names: list[str]) -> list[str]:
    """zfp with its tolerance loosened until it spends warpfield's error budget."""
    out = [
        '<section id="zfp"><h2>zfp at matched error</h2>',
        "<p>zfp's <span class='mono'>tolerance</span> is a loose upper bound, so at "
        "tolerance 0.25 it spends far less error than warpfield. Here the tolerance is raised "
        "until zfp's <i>realised</i> error matches warpfield at ε = 0.25: first its maximum "
        "error (the guarantee), then its RMS error (the typical error, where zfp is "
        "favoured because its errors are mostly tiny with rare large ones). "
        "zfp's tolerance only takes effect in powers of two.</p>",
        '<div class="scroll"><table><thead><tr><th class="l">preset</th>'
        "<th>warpfield ratio</th><th>max / RMS err</th><th>seam</th><th>det J viol.</th>"
        "<th>zfp, same max err: tol</th><th>ratio</th><th>max / RMS err</th><th>seam</th>"
        "<th>det J viol.</th>"
        "<th>zfp, same RMS err: tol</th><th>ratio</th><th>max err</th><th>seam</th>"
        "<th>det J viol.</th></tr></thead><tbody>",
    ]
    for name in names:
        rs = [r for r in sweep if r["preset"] == name]
        w = next((r for r in rs if r["codec"] == "warpfield" and r["knob"] == 0.25), None)
        z = [r for r in rs if r["codec"] == "zfp"]
        if not w or not z:
            continue
        zm = max(
            (r for r in z if r["max_err"] <= w["max_err"] + 1e-6), key=lambda r: r["ratio"]
        )
        zr = min(z, key=lambda r: abs(np.log(r["rms_err"] / w["rms_err"])))
        cells = [
            f"<td class='l'>{html.escape(name)}</td>",
            f"<td class='good'>{fmt_ratio(w['ratio'])}</td>",
            f"<td>{w['max_err']:.2f} / {w['rms_err']:.3f}</td>",
            f"<td>{w['seam_ratio']:.2f}</td>",
            f"<td>{w['jac']}</td>",
            f"<td>{zm['knob']:g}</td>",
            f"<td class='bad'>{fmt_ratio(zm['ratio'])}</td>",
            f"<td>{zm['max_err']:.2f} / {zm['rms_err']:.3f}</td>",
            f"<td>{zm['seam_ratio']:.2f}</td>",
            f"<td>{zm['jac']}</td>",
            f"<td>{zr['knob']:g}</td>",
            f"<td>{fmt_ratio(zr['ratio'])}</td>",
            f"<td class='bad'>{zr['max_err']:.2f}</td>",
            f"<td>{zr['seam_ratio']:.2f}</td>",
            f"<td>{zr['jac']}</td>",
        ]
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div></section>")
    return out


def build(gallery: Path, bench: Path, sweep_path: Path | None = None) -> Path:
    man = json.loads((gallery / "manifest.json").read_text())
    rows = json.loads(bench.read_text()) if bench.exists() else []
    sweep = (
        json.loads(sweep_path.read_text())
        if sweep_path is not None and sweep_path.exists()
        else []
    )
    e = html.escape
    out = [
        "<title>warpfield codec gallery</title>",
        '<meta name="claude:viewport-fit-sides" content="cover">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:'
        'wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">',
        f"<style>{CSS}</style>",
        '<main class="wrap">',
        "<header style='display:grid;gap:10px'>",
        '<span class="eyebrow">zarr extension draft · experimental</span>',
        "<h1>warpfield: error-bounded, seam-free compression of alignment fields</h1>",
        '<p class="lede">Synthetic registration fields with tears, folds (true jumps), steep bands and '
        "misaligned sections, compressed chunk by chunk and used to warp a test image. Every "
        "decoded sample is within ε of the original; chunk faces are coded at ε/4 so chunk "
        "boundaries leave no seam; the Jacobian pass keeps cells invertible inside each chunk."
        "</p></header>",
        "<nav>"
        + "".join(f'<a href="#{e(m["preset"])}">{e(m["preset"])}</a>' for m in man)
        + '<a href="#zfp">zfp matched</a><a href="#bench">benchmark</a></nav>',
    ]
    # summary at eps = 0.25
    s = [r for r in rows if abs(r["eps"] - 0.25) < 1e-9]
    if s:
        out.append("<section><h2>At ε = 0.25 voxel</h2>")
        out.append(
            "<p>Ratios are raw float32 bytes over stored bytes, all codecs chunked identically. "
            "Seam ratio is the mean error jump across chunk faces divided by the mean jump "
            "inside chunks (1 = invisible). Jacobian violations count cells whose det J fell "
            "below 0.1 or changed sign; those left by warpfield all straddle a chunk face.</p>"
        )
        out.append('<div class="scroll"><table><thead><tr><th class="l">preset</th>')
        codecs = ["warpfield", "baseline", "zfp"]
        for c in codecs:
            out.append(f"<th>{c} ratio</th>")
        out.append(
            "<th>warpfield max err</th><th>warpfield seam</th><th>baseline seam</th>"
            "<th>warpfield det J viol.</th><th>no-Jacobian det J viol.</th></tr></thead><tbody>"
        )
        for m in man:
            by = {r["codec"]: r for r in s if r["preset"] == m["preset"]}
            if "warpfield" not in by:
                continue
            w = by["warpfield"]
            cells = [f'<td class="l"><a href="#{e(m["preset"])}">{e(m["preset"])}</a></td>']
            for c in codecs:
                cells.append(f"<td>{fmt_ratio(by[c]['ratio']) if c in by else '–'}</td>")
            cells.append(f"<td>{w['max_err']:.3f}</td>")
            cells.append(f"<td class='good'>{w['seam_ratio']:.2f}</td>")
            b = by.get("baseline")
            cells.append(f"<td class='bad'>{b['seam_ratio']:.2f}</td>" if b else "<td>–</td>")
            cells.append(
                f"<td>{w['jac_violations']} "
                f"<span class='mono' style='color:var(--muted)'>"
                f"({w['jac_violations_cross_chunk']} at faces)</span></td>"
            )
            nj = by.get("no-jacobian")
            cells.append(f"<td>{nj['jac_violations'] if nj else '–'}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
        out.append("</tbody></table></div></section>")
    out.append(
        "<section><h2>Reading the figures</h2><div class='legend'>"
        "<div><b>raw field</b><p>ground truth</p></div>"
        "<div><b>warpfield</b><p>this codec: ε = 0.25, faces ε/4, ENO, det J ≥ 0.1 kept</p></div>"
        "<div><b>baseline</b><p>same predictor and quantizer without face classes, SZ3-style "
        "extrapolation at chunk edges</p></div>"
        "<div><b>zfp</b><p>zfp fixed-accuracy, tolerance ε, per chunk and component</p></div>"
        "<div><b>labels</b><p><span style='color:#e61a1a'>slip crack</span>, "
        "<span style='color:#ff8c00'>opening crack</span>, "
        "<span style='color:#00bfd9'>steep band</span>, "
        "<span style='color:#d900d9'>fold (overlap jump)</span>, "
        "<span style='color:#5959ff'>over-compression (continuous, det J &lt; 0)</span>, "
        "<span style='color:#c9a800'>misaligned section</span></p></div>"
        "</div></section>"
    )
    if sweep:
        out += zfp_matched(sweep, [m["preset"] for m in man])
    for m in man:
        out.append(f'<section id="{e(m["preset"])}">')
        out.append(f"<h2>{e(m['preset'])} — {e(m['description'])}</h2>")
        facts = [
            ("shape", m["shape"]),
            ("chunks", m["chunks"]),
            ("axis_roles", ", ".join(m["axis_roles"])),
        ]
        rat = ", ".join(f"{k} {fmt_ratio(v)}" for k, v in m["ratios"].items() if v)
        out.append(
            "<div class='facts'>"
            + "".join(f"<span>{k} <b class='mono'>{e(str(v))}</b></span>" for k, v in facts)
            + f"<span>ratios <b>{e(rat)}</b></span></div>"
        )
        for v in m["views"]:
            out.append(f"<h3>{e(v['title'])}</h3>")
            for f in v["files"]:
                out.append(
                    f"<figure><img loading='lazy' src='{e(m['preset'])}/{e(f)}' "
                    f"alt='{e(m['preset'])} {e(v['title'])} {kind(f)}'>"
                    f"<figcaption>{CAPTIONS[kind(f)]}</figcaption></figure>"
                )
        for f in m.get("extra", []):
            out.append(
                f"<figure><img loading='lazy' src='{e(m['preset'])}/{e(f)}' "
                f"alt='{e(m['preset'])} {kind(f)}'>"
                f"<figcaption>{CAPTIONS[kind(f)]}</figcaption></figure>"
            )
        out.append("</section>")
    if rows:
        out.append('<section id="bench"><h2>Full benchmark</h2>')
        out.append(
            "<p>All presets, ε ∈ {0.05, 0.1, 0.25, 0.5}. zfp's fixed-accuracy mode stays far "
            "inside its tolerance, which is why its ratios are low. Speeds are the pure-numpy "
            "reference implementation on one core.</p>"
        )
        cols = [
            ("preset", "preset"),
            ("ε", "eps"),
            ("codec", "codec"),
            ("ratio", "ratio"),
            ("bits/value", "bits_per_value"),
            ("max err", "max_err"),
            ("max err on disc.", "max_err_on_discontinuities"),
            ("seam", "seam_ratio"),
            ("det J viol.", "jac_violations"),
            ("at faces", "jac_violations_cross_chunk"),
            ("verbatim", "verbatim"),
            ("tight", "tight"),
            ("enc MB/s", "encode_MBps"),
            ("dec MB/s", "decode_MBps"),
        ]
        out.append('<div class="scroll" style="max-height:520px"><table><thead><tr>')
        out.append(
            "".join(
                f"<th class='{'l' if i < 3 else ''}'>{c}</th>" for i, (c, _) in enumerate(cols)
            )
        )
        out.append("</tr></thead><tbody>")
        for r in rows:
            tds = []
            for i, (_, k) in enumerate(cols):
                v = r[k]
                if k == "ratio":
                    v = fmt_ratio(v)
                elif isinstance(v, float):
                    v = f"{v:.3f}" if v < 10 else f"{v:.1f}"
                tds.append(f"<td class='{'l' if i < 3 else ''}'>{e(str(v))}</td>")
            cls = " class='wf'" if r["codec"] == "warpfield" else ""
            out.append(f"<tr{cls}>" + "".join(tds) + "</tr>")
        out.append("</tbody></table></div></section>")
    out.append("</main>")
    path = gallery / "report.html"
    path.write_text("\n".join(out))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=Path, default=Path("out/gallery"))
    ap.add_argument("--bench", type=Path, default=Path("out/benchmark.json"))
    ap.add_argument("--sweep", type=Path, default=Path("out/zfp_sweep.json"))
    a = ap.parse_args()
    print(build(a.gallery, a.bench, a.sweep))
