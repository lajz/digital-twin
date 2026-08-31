"""Real-time 2D player: animate real microscopy frames beside a twin's rollout.

Produces one self-contained HTML file (canvas animation, no libraries) that plays through
the whole time-lapse -- reality on the left, the twin on the right -- with a scrubber, a
play/pause/speed control, and the fit/holdout boundary marked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medusa.data.build import Dataset


def _pack_frames(frames) -> list[list]:
    """One entry per frame: [x[], y[], angle[], length[], width] rounded for size."""
    out = []
    for f in frames:
        out.append([
            np.round(f.x, 1).tolist(),
            np.round(f.y, 1).tolist(),
            np.round(f.angle, 3).tolist(),
            np.round(f.length, 1).tolist(),
            round(float(np.median(f.width)) if len(f) else 1.0, 2),
        ])
    return out


def build_player(
    dataset: Dataset,
    twin_source: str,
    params: dict,
    out_path: str | Path,
    *,
    title: str = "",
    family: str = "twin",
    seed: int = 0,
) -> Path:
    from medusa.harness.evaluate import spatial_rollout
    from medusa.spatial.summarize import summarize_frame

    real, twin = spatial_rollout(twin_source, params, dataset, seed=seed)
    n = min(len(real), len(twin))

    allx = np.concatenate([f.x for f in real.frames] + [f.x for f in twin.frames])
    ally = np.concatenate([f.y for f in real.frames] + [f.y for f in twin.frames])
    pad = 3.0
    world = [float(allx.min() - pad), float(allx.max() + pad),
             float(ally.min() - pad), float(ally.max() + pad)]

    t_split = dataset.split["t_split_s"]
    split_frame = int(np.searchsorted(real.time_s[:n], t_split, side="right"))

    real_nn = [summarize_frame(f)["mean_nn_dist_um"] for f in real.frames[:n]]

    data = {
        "meta": {
            "family": family,
            "params": {k: round(float(v), 4) for k, v in params.items()},
            "times_h": np.round(real.time_s[:n] / 3600.0, 3).tolist(),
            "world": world,
            "splitFrame": split_frame,
            "dataset": dataset.name,
        },
        "real": _pack_frames(real.frames[:n]),
        "twin": _pack_frames(twin.frames[:n]),
        "realNN": [round(v, 2) for v in real_nn],
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
        "__TITLE__", title or f"{dataset.name}: reality vs twin"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root{
    --ground:#eef1f4;--panel:#ffffff;--ink:#132029;--muted:#5c6d78;--faint:#8a9aa4;
    --line:#d7dfe3;--reality:#12303f;--twin:#2f86bf;--holdout:#bd2a1e;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --ground:#0a0f14;--panel:#0f171e;--ink:#dde7ec;--muted:#8ea0ab;--faint:#5f727d;
    --line:#213038;--reality:#9ecbe4;--twin:#5fa8d3;--holdout:#e0654a;
  }}
  :root[data-theme="dark"]{
    --ground:#0a0f14;--panel:#0f171e;--ink:#dde7ec;--muted:#8ea0ab;--faint:#5f727d;
    --line:#213038;--reality:#9ecbe4;--twin:#5fa8d3;--holdout:#e0654a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
    font:400 15px/1.55 "IBM Plex Sans",system-ui,sans-serif;}
  .wrap{max-width:1040px;margin:0 auto;padding:36px 20px 64px;}
  .mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;}
  .eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;
    text-transform:uppercase;color:var(--twin);}
  h1{font-family:"Newsreader",Georgia,serif;font-weight:400;font-size:clamp(28px,4.4vw,40px);
    letter-spacing:-.01em;margin:.3em 0 .5em;text-wrap:balance;}
  .stage{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
  @media (max-width:720px){.stage{grid-template-columns:1fr;}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:12px;overflow:hidden;}
  .panel h2{margin:0 0 8px;font:500 12px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
    text-transform:uppercase;}
  .panel.real h2{color:var(--reality);} .panel.twin h2{color:var(--twin);}
  canvas{display:block;width:100%;height:auto;background:#fff;border-radius:7px;
    border:1px solid var(--line);}
  .count{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--muted);
    margin-top:7px;display:flex;justify-content:space-between;}
  .bar{display:flex;align-items:center;gap:14px;margin-top:20px;}
  button{font-family:"IBM Plex Mono",monospace;background:var(--panel);color:var(--ink);
    border:1px solid var(--line);border-radius:8px;height:40px;min-width:44px;padding:0 12px;
    cursor:pointer;font-size:15px;}
  button:hover{border-color:var(--twin);color:var(--twin);}
  button:focus-visible{outline:2px solid var(--twin);outline-offset:2px;}
  .track{flex:1;position:relative;height:40px;}
  input[type=range]{width:100%;position:absolute;top:9px;left:0;margin:0;accent-color:var(--twin);}
  .split{position:absolute;top:0;bottom:14px;width:2px;background:var(--holdout);opacity:.7;}
  .split::after{content:"holdout";position:absolute;top:-15px;left:4px;font:400 10px/1 "IBM Plex Mono",monospace;
    color:var(--holdout);white-space:nowrap;}
  .readout{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--muted);
    margin-top:12px;display:flex;gap:22px;flex-wrap:wrap;}
  .readout b{color:var(--ink);}
  .speeds{display:flex;gap:4px;}
  .speeds button{height:28px;min-width:34px;font-size:12px;padding:0 6px;}
  .speeds button[aria-pressed=true]{border-color:var(--twin);color:var(--twin);background:color-mix(in srgb,var(--twin) 12%,transparent);}
  p.note{color:var(--muted);font-size:13px;max-width:70ch;margin-top:18px;}
  .legend{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:5px;}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">

<div class="wrap">
  <div class="eyebrow">medusa &nbsp;/&nbsp; L2 &middot; real time</div>
  <h1 id="h1"></h1>
  <div class="stage">
    <div class="panel real"><h2>reality &mdash; segmented microscopy</h2>
      <canvas id="cReal"></canvas>
      <div class="count"><span>cells <b id="nReal" class="mono"></b></span><span id="phaseReal"></span></div>
    </div>
    <div class="panel twin"><h2 id="twinName">twin &mdash; rollout</h2>
      <canvas id="cTwin"></canvas>
      <div class="count"><span>cells <b id="nTwin" class="mono"></b></span><span id="phaseTwin"></span></div>
    </div>
  </div>

  <div class="bar">
    <button id="play" aria-label="play / pause">&#9654;</button>
    <button id="restart" aria-label="restart">&#8635;</button>
    <div class="track">
      <div class="split" id="split"></div>
      <input type="range" id="scrub" min="0" value="0" step="1">
    </div>
    <div class="speeds" id="speeds"></div>
  </div>

  <div class="readout">
    <span>t = <b id="tNow" class="mono"></b> h</span>
    <span>frame <b id="fNow" class="mono"></b></span>
    <span id="phaseNow"></span>
  </div>
  <p class="note">
    <span class="legend" style="background:var(--reality)"></span>reality is the
    CellProfiler+Omnipose segmentation of a real <i>E.&nbsp;coli</i> monolayer;
    <span class="legend" style="background:var(--twin)"></span>the twin is an agent-written
    rod simulator, seeded from the first real frame and rolled forward &mdash; it is
    calibrated only on the fit phase and never sees the held-out frames it is drawn against
    here. Same world coordinates in both panels.
  </p>
</div>

<script>
const D = __DATA__;
const M = D.meta, R = D.real, T = D.twin, NF = R.length;
document.getElementById('h1').textContent = M.dataset.replace(/-/g,' ') + ': reality vs twin';
document.getElementById('twinName').innerHTML = 'twin &mdash; ' + M.family;

const [x0,x1,y0,y1] = M.world;
const wW = x1 - x0, wH = y1 - y0;
const cssW = 460, cssH = Math.round(cssW * wH / wW);
const dpr = Math.min(window.devicePixelRatio || 1, 2);

function setup(cv){
  cv.style.aspectRatio = wW + ' / ' + wH;
  cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  const g = cv.getContext('2d');
  g.scale(cv.width / wW, cv.height / wH);
  return g;
}
const gReal = setup(document.getElementById('cReal'));
const gTwin = setup(document.getElementById('cTwin'));
const scalePx = (document.getElementById('cReal').width / dpr) / wW; // css px per micron

function draw(g, frame, color){
  g.save();
  g.clearRect(0,0,wW,wH);
  g.translate(-x0,-y0);
  g.strokeStyle = color; g.lineCap = 'round';
  const [xs,ys,as,ls,w] = frame;
  g.lineWidth = Math.max(w, 0.7);
  g.globalAlpha = 0.92;
  g.beginPath();
  for (let i=0;i<xs.length;i++){
    const half = Math.max((ls[i]-w)/2, 0.15);
    const c = Math.cos(as[i])*half, s = Math.sin(as[i])*half;
    g.moveTo(xs[i]-c, ys[i]-s); g.lineTo(xs[i]+c, ys[i]+s);
  }
  g.stroke();
  g.restore();
}

const css = getComputedStyle(document.documentElement);
function render(fi){
  const cR = css.getPropertyValue('--reality').trim();
  const cT = css.getPropertyValue('--twin').trim();
  draw(gReal, R[fi], cR);
  draw(gTwin, T[fi], cT);
  document.getElementById('nReal').textContent = R[fi][0].length;
  document.getElementById('nTwin').textContent = T[fi][0].length;
  document.getElementById('tNow').textContent = M.times_h[fi].toFixed(2);
  document.getElementById('fNow').textContent = (fi+1) + ' / ' + NF;
  const held = fi >= M.splitFrame;
  const phase = held ? 'held-out' : 'fit';
  const col = held ? 'var(--holdout)' : 'var(--muted)';
  for (const id of ['phaseReal','phaseTwin','phaseNow']){
    const el = document.getElementById(id); el.textContent = phase; el.style.color = col;
  }
  scrub.value = fi;
}

// timeline split marker
const split = document.getElementById('split');
split.style.left = (M.splitFrame / (NF-1) * 100) + '%';
const scrub = document.getElementById('scrub');
scrub.max = NF - 1;

let fi = 0, playing = false, fps = 10, acc = 0, last = 0, raf = 0;
function loop(ts){
  if (!playing) return;
  if (!last) last = ts;
  acc += (ts - last) / 1000; last = ts;
  const step = 1 / fps;
  while (acc >= step){
    acc -= step; fi++;
    if (fi >= NF){ fi = NF-1; stop(); break; }
  }
  render(fi);
  raf = requestAnimationFrame(loop);
}
function start(){
  if (fi >= NF-1) fi = 0;
  playing = true; last = 0; acc = 0;
  document.getElementById('play').innerHTML = '&#9612;&#9612;';
  raf = requestAnimationFrame(loop);
}
function stop(){
  playing = false; cancelAnimationFrame(raf);
  document.getElementById('play').innerHTML = '&#9654;';
}
document.getElementById('play').onclick = () => playing ? stop() : start();
document.getElementById('restart').onclick = () => { stop(); fi = 0; render(0); };
scrub.addEventListener('input', e => { stop(); fi = +e.target.value; render(fi); });
document.addEventListener('keydown', e => {
  if (e.key === ' '){ e.preventDefault(); playing ? stop() : start(); }
  if (e.key === 'ArrowRight'){ stop(); fi = Math.min(NF-1, fi+1); render(fi); }
  if (e.key === 'ArrowLeft'){ stop(); fi = Math.max(0, fi-1); render(fi); }
});

const speeds = [0.5, 1, 2, 4];
document.getElementById('speeds').innerHTML = speeds.map(s =>
  `<button data-fps="${s*10}" aria-pressed="${s===1}">${s}&times;</button>`).join('');
document.querySelectorAll('.speeds button').forEach(b => b.onclick = () => {
  fps = +b.dataset.fps;
  document.querySelectorAll('.speeds button').forEach(x =>
    x.setAttribute('aria-pressed', x === b));
});

render(0);
if (matchMedia('(prefers-reduced-motion: no-preference)').matches) start();
</script>
"""
