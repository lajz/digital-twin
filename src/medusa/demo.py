"""Build a self-contained demo.html for a run: model-vs-reality renders, scrubbable
across training iterations. Stdlib only; images embedded as base64 so it is portable.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _iter_images(run_dir: Path) -> list[tuple[int, Path]]:
    demo_dir = run_dir / "demo"
    out: list[tuple[int, Path]] = []
    for idir in sorted(run_dir.glob("iter_*")):
        n = int(idir.name.split("_")[1])
        spatial = demo_dir / f"iter_{n:02d}.png"
        series = idir / "forecast.png"
        if spatial.exists():
            out.append((n, spatial))
        elif series.exists():
            out.append((n, series))
    return out


def build_demo(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    imgs = _iter_images(run_dir)
    if not imgs:
        return None

    meta = json.loads((run_dir / "meta.json").read_text())
    scard = {}
    if (run_dir / "scorecard.json").exists():
        scard = json.loads((run_dir / "scorecard.json").read_text())
    trace = {}
    if (run_dir / "trace.jsonl").exists():
        for line in (run_dir / "trace.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trace[r["iter"]] = r

    frames = []
    for n, p in imgs:
        r = trace.get(n, {})
        per = r.get("per_observable") or {}
        per_txt = "  ".join(f"{k}:{v:.3f}" for k, v in per.items())
        frames.append(
            {
                "iter": n,
                "img": _img(p),
                "family": r.get("family") or "-",
                "smape": r.get("holdout_smape"),
                "status": r.get("status", "-"),
                "per": per_txt,
            }
        )

    task = meta.get("task", {}).get("name", "?")
    best = scard.get("best_holdout_smape")
    subtitle = (
        f"{meta.get('dataset','?')} &middot; task: {task} &middot; "
        f"{len(frames)} iterations &middot; best holdout sMAPE "
        f"{best:.3f}" if isinstance(best, (int, float)) else f"{meta.get('dataset','?')} &middot; task: {task}"
    )

    html = _TEMPLATE.replace("__SUBTITLE__", subtitle).replace(
        "__FRAMES__", json.dumps(frames)
    ).replace("__DATASET__", meta.get("dataset", "run"))
    out = run_dir / "demo.html"
    out.write_text(html)
    return out


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>medusa demo &mdash; __DATASET__</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background:#0e0f13; color:#e8e8ea; }
 .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
 h1 { font-size: 20px; margin: 0 0 2px; }
 .sub { color:#9aa; margin-bottom: 18px; }
 .stage { background:#16171d; border:1px solid #26272e; border-radius:10px; padding:14px; }
 img { width:100%; display:block; border-radius:6px; background:#fff; }
 .bar { display:flex; align-items:center; gap:14px; margin:14px 0 4px; }
 input[type=range] { flex:1; }
 .meta { font-variant-numeric: tabular-nums; color:#cdd; }
 .k { color:#7db8e8; }
 button { background:#26272e; color:#e8e8ea; border:1px solid #383a44; border-radius:6px;
          padding:4px 10px; cursor:pointer; }
 .cap { color:#9aa; font-size:12px; margin-top:8px; }
</style></head><body><div class="wrap">
<h1>medusa &mdash; model vs reality, over training</h1>
<div class="sub">__SUBTITLE__</div>
<div class="stage">
  <img id="frame" alt="model vs reality">
  <div class="bar">
    <button id="prev">&larr;</button>
    <input type="range" id="slider" min="0" value="0" step="1">
    <button id="next">&rarr;</button>
  </div>
  <div class="meta" id="meta"></div>
  <div class="cap">Top row: reality (segmented microscopy). Bottom row: the twin's rollout
  from the first frame. Columns span the movie; red labels are held-out (never fit).</div>
</div>
<script>
const F = __FRAMES__;
const img = document.getElementById('frame'), s = document.getElementById('slider'),
      meta = document.getElementById('meta');
s.max = F.length - 1;
function show(i){
  i = Math.max(0, Math.min(F.length-1, i|0)); s.value = i;
  const f = F[i];
  img.src = f.img;
  const sm = (f.smape==null) ? '&mdash;' : (+f.smape).toFixed(4);
  meta.innerHTML = `iteration <span class="k">${f.iter}</span> / ${F.length}
    &nbsp;&nbsp; family <span class="k">${f.family}</span>
    &nbsp;&nbsp; holdout sMAPE <span class="k">${sm}</span>
    &nbsp;&nbsp; ${f.status}` + (f.per ? `<br><span style="color:#889">${f.per}</span>` : '');
}
s.addEventListener('input', e => show(+e.target.value));
document.getElementById('prev').onclick = () => show(+s.value - 1);
document.getElementById('next').onclick = () => show(+s.value + 1);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') show(+s.value - 1);
  if (e.key === 'ArrowRight') show(+s.value + 1);
});
show(F.length - 1);
</script></div></body></html>"""
