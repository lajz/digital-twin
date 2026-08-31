"""Build the medusa L2 demo artifact from runs/sample/."""
import base64, json, pathlib

RUN = pathlib.Path(__file__).resolve().parent.parent / "runs" / "sample"
OUT = pathlib.Path(__file__).parent / "medusa-demo.html"

rows = [json.loads(l) for l in (RUN / "trace.jsonl").read_text().splitlines() if l.strip()]
scard = json.loads((RUN / "scorecard.json").read_text())
meta = json.loads((RUN / "meta.json").read_text())

OBS = ["population_count", "total_length_um", "colony_radius_um",
       "colony_aspect", "nematic_order", "mean_nn_dist_um"]
OBS_LABEL = {
    "population_count": "cell count", "total_length_um": "total rod length",
    "colony_radius_um": "radius of gyration", "colony_aspect": "colony aspect",
    "nematic_order": "nematic order", "mean_nn_dist_um": "neighbour spacing",
}

frames = []
for r in rows:
    n = r["iter"]
    img = RUN / "demo" / f"iter_{n:02d}.png"
    per = r.get("per_observable") or {}
    frames.append({
        "iter": n,
        "img": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode() if img.exists() else "",
        "family": r.get("family") or "-",
        "smape": r.get("holdout_smape"),
        "valid": bool(r.get("is_valid")),
        "per": {k: per.get(k) for k in OBS},
    })

best_i = min((f for f in frames if f["valid"] and f["smape"] is not None),
             key=lambda f: f["smape"])["iter"]

portfolio = [
    {"rank": 1, "family": "stochastic-rod-population", "iter": 6, "smape": 0.143,
     "note": "matches count, colony extent and neighbour packing; misses local alignment"},
    {"rank": 2, "family": "overdamped-rods", "iter": 1, "smape": 0.201,
     "note": "reproduces local alignment (nematic order) but packs too loosely"},
    {"rank": 3, "family": "pore-channel-network", "iter": 12, "smape": 0.279,
     "note": "channels between clusters; growth rate a touch slow"},
]

DATA = {
    "frames": frames, "best": best_i, "portfolio": portfolio,
    "scard": {k: scard[k] for k in ("best_holdout_smape", "distinct_plausible_families",
                                    "n_iters", "usd_cost", "improvement_auc")},
    "obs": OBS, "obsLabel": OBS_LABEL,
    "split": {"fit": meta["split"]["n_fit"], "holdout": meta["split"]["n_holdout"]},
}

HTML = r"""<title>Twinning a Microcolony</title>
<style>
  :root{
    --ground:#eef1f4; --surface:#ffffff; --surface-2:#f4f7f8; --card:#ffffff;
    --ink:#132029; --muted:#5c6d78; --faint:#8a9aa4; --line:#d7dfe3;
    --reality:#1b4965; --twin:#2f86bf; --holdout:#bd2a1e; --good:#237a5b; --amber:#b26a12;
    --shadow: 0 1px 2px rgba(19,32,41,.06), 0 12px 32px -12px rgba(19,32,41,.18);
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    --ground:#0a0f14; --surface:#111a21; --surface-2:#0d151b; --card:#0f171e;
    --ink:#dde7ec; --muted:#8ea0ab; --faint:#5f727d; --line:#213038;
    --reality:#7fb4d4; --twin:#5fa8d3; --holdout:#e0654a; --good:#4cbd94; --amber:#d69a4a;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 20px 44px -18px rgba(0,0,0,.7);
  }}
  :root[data-theme="dark"]{
    --ground:#0a0f14; --surface:#111a21; --surface-2:#0d151b; --card:#0f171e;
    --ink:#dde7ec; --muted:#8ea0ab; --faint:#5f727d; --line:#213038;
    --reality:#7fb4d4; --twin:#5fa8d3; --holdout:#e0654a; --good:#4cbd94; --amber:#d69a4a;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 20px 44px -18px rgba(0,0,0,.7);
  }
  *{box-sizing:border-box}
  body{background:var(--ground); color:var(--ink);
    font:400 16px/1.6 "IBM Plex Sans", system-ui, sans-serif;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:920px; margin:0 auto; padding:56px 24px 80px;}
  .mono{font-family:"IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric:tabular-nums;}
  .eyebrow{font-family:"IBM Plex Mono", monospace; font-size:12px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--twin);}
  h1{font-family:"Newsreader", Georgia, serif; font-weight:400; font-size:clamp(34px,6vw,52px);
    line-height:1.08; letter-spacing:-.01em; margin:.28em 0 .3em; text-wrap:balance;}
  h1 em{font-style:italic; color:var(--reality);}
  .dek{font-size:19px; color:var(--muted); max-width:60ch; text-wrap:pretty;}
  h2{font-family:"Newsreader", Georgia, serif; font-weight:500; font-size:24px;
    margin:56px 0 14px; letter-spacing:-.01em;}

  .stats{display:flex; flex-wrap:wrap; gap:2px 30px; margin:30px 0 40px;
    padding:18px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line);}
  .stat{display:flex; flex-direction:column;}
  .stat b{font-family:"IBM Plex Mono", monospace; font-size:22px; font-weight:500;
    letter-spacing:-.01em;}
  .stat span{font-size:12.5px; color:var(--faint); text-transform:uppercase; letter-spacing:.06em;}

  .viewer{background:var(--surface); border:1px solid var(--line); border-radius:14px;
    padding:16px; box-shadow:var(--shadow);}
  .specimen{background:#fff; border-radius:9px; overflow:hidden; border:1px solid var(--line);}
  .specimen img{display:block; width:100%; height:auto;}
  .cap{display:flex; justify-content:space-between; align-items:baseline; gap:12px;
    margin:13px 4px 4px; flex-wrap:wrap;}
  .cap .fam{font-family:"IBM Plex Mono", monospace; font-size:14px;}
  .cap .sm{font-family:"IBM Plex Mono", monospace; font-size:14px; color:var(--muted);}
  .cap .sm b{color:var(--ink);}

  .controls{display:flex; align-items:center; gap:12px; margin-top:12px;}
  button.nav{font-family:"IBM Plex Mono",monospace; background:var(--surface-2);
    color:var(--ink); border:1px solid var(--line); border-radius:7px; height:34px;
    min-width:38px; padding:0 10px; cursor:pointer; font-size:14px;}
  button.nav:hover{border-color:var(--twin); color:var(--twin);}
  button.nav:focus-visible{outline:2px solid var(--twin); outline-offset:2px;}
  .scrub{flex:1;}
  .scrub svg{display:block; width:100%; height:78px; overflow:visible;}
  .scrub rect.bar{cursor:pointer;}
  .axis{font-family:"IBM Plex Mono",monospace; font-size:10px; fill:var(--faint);}

  .detail{margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:6px 26px;}
  @media (max-width:560px){ .detail{grid-template-columns:1fr;} }
  .metric{display:grid; grid-template-columns:120px 1fr 46px; align-items:center; gap:10px;
    padding:5px 0; font-size:13px;}
  .metric .name{color:var(--muted);}
  .metric .track{height:7px; background:var(--surface-2); border-radius:4px; overflow:hidden;}
  .metric .fill{height:100%; border-radius:4px;}
  .metric .val{font-family:"IBM Plex Mono",monospace; font-size:12px; text-align:right; color:var(--muted);}

  p{max-width:64ch; color:var(--ink);}
  p.sub{color:var(--muted);}
  .pf{display:flex; flex-direction:column; gap:10px;}
  .pf .row{display:grid; grid-template-columns:26px 1fr auto; gap:14px; align-items:baseline;
    padding:14px 16px; background:var(--surface); border:1px solid var(--line); border-radius:10px;}
  .pf .rank{font-family:"Newsreader",serif; font-size:20px; color:var(--faint);}
  .pf .fam{font-family:"IBM Plex Mono",monospace; font-size:13.5px;}
  .pf .note{color:var(--muted); font-size:13.5px; grid-column:2; margin-top:3px;}
  .pf .sc{font-family:"IBM Plex Mono",monospace; font-size:14px;}
  a{color:var(--twin);}
  footer{margin-top:56px; padding-top:20px; border-top:1px solid var(--line);
    color:var(--faint); font-size:13px;}
  .key{display:inline-block; padding:1px 6px; border-radius:4px; font-size:11px;
    font-family:"IBM Plex Mono",monospace; border:1px solid var(--line); color:var(--muted);}
  @media (prefers-reduced-motion:no-preference){
    .specimen img{transition:opacity .12s;}
  }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">

<div class="wrap">
  <div class="eyebrow">medusa &nbsp;/&nbsp; L2 spatial twin</div>
  <h1>Twinning a <em>microcolony</em></h1>
  <p class="dek">A coding agent writes an agent-based simulator of a growing
  <i>E.&nbsp;coli</i> colony, a harness scores its rollout against real phase-contrast
  microscopy, and the score feeds the next attempt. Twelve iterations; the twin is
  seeded from the first real frame and never sees the held-out future.</p>

  <div class="stats mono" id="stats"></div>

  <div class="viewer">
    <div class="specimen"><img id="frame" alt="the twin's rollout beside the real microscopy"></div>
    <div class="cap">
      <span class="fam">iteration <b id="cap-iter"></b> &nbsp;&middot;&nbsp; <span id="cap-fam"></span></span>
      <span class="sm">combined holdout sMAPE <b id="cap-smape"></b></span>
    </div>
    <div class="controls">
      <button class="nav" id="prev" aria-label="previous iteration">&larr;</button>
      <button class="nav" id="play" aria-label="play">&#9654;</button>
      <button class="nav" id="next" aria-label="next iteration">&rarr;</button>
      <div class="scrub"><svg id="scrub" viewBox="0 0 740 78" preserveAspectRatio="none"
        role="slider" aria-label="training iteration" tabindex="0"></svg></div>
    </div>
    <div class="detail" id="detail"></div>
    <p class="sub" style="font-size:12.5px;margin:14px 4px 2px">
      Top row: reality (segmented microscopy). Bottom row: the twin's rollout from the
      first frame. Columns span the 2.5&nbsp;h movie; red labels mark held-out frames.
      Bars above: per-iteration match quality; the line tracks the best twin so far.</p>
  </div>

  <h2>Reading the score</h2>
  <p>Each candidate is judged on six spatial summary-statistic time series over the
  holdout window &mdash; cell count, total rod length, radius of gyration, colony aspect
  ratio, nematic (alignment) order, and mean nearest-neighbour distance &mdash; combined
  as a weighted <span class="mono">sMAPE</span>. Never on pixels; never on frames the
  agent was shown. A candidate is thrown out if it crashes, is non-deterministic for a
  fixed seed, runs over budget, implies a biologically implausible growth rate, or
  returns a first frame that isn't the given initial condition.</p>

  <h2>The portfolio</h2>
  <p class="sub">The loop keeps the best candidate per mechanistic family. The top three
  disagree in an informative way &mdash; each mechanism captures a different facet of the
  real spatial structure.</p>
  <div class="pf" id="pf"></div>

  <footer>
    Data: <i>E. coli</i> K-12 monolayer microcolony, from Ahmadi et&nbsp;al. 2024,
    <i>&ldquo;A benchmarked comparison of software packages for time-lapse image
    processing of monolayer bacterial population dynamics&rdquo;</i> (Microbiology
    Spectrum; CC-BY&nbsp;4.0) &mdash; 100 frames at 90&nbsp;s, CellProfiler+Omnipose
    segmentation, pixel size self-calibrated to the ~1&nbsp;&micro;m cell width.
    Model: <span class="mono">deepseek-v4-flash</span>.
    <span class="key">&larr;</span> <span class="key">&rarr;</span> to step, click a bar to jump.
  </footer>
</div>

<script>
const D = __DATA__;
const F = D.frames, N = F.length;
let cur = D.best - 1, playing = false, timer = null;

// stats strip
document.getElementById('stats').innerHTML = [
  ['best holdout sMAPE', (+D.scard.best_holdout_smape).toFixed(3)],
  ['model families', D.scard.distinct_plausible_families],
  ['iterations', D.scard.n_iters],
  ['fit / holdout frames', D.split.fit + ' / ' + D.split.holdout],
  ['cost (USD)', '$' + (+D.scard.usd_cost).toFixed(3)],
].map(([l,v]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join('');

// portfolio
document.getElementById('pf').innerHTML = D.portfolio.map(p => `
  <div class="row">
    <div class="rank">${p.rank}</div>
    <div><span class="fam">${p.family}</span>
      <div class="note">${p.note} &nbsp;&middot;&nbsp; iteration ${p.iter}</div></div>
    <div class="sc">${p.smape.toFixed(3)}</div>
  </div>`).join('');

// scrubber
const svg = document.getElementById('scrub');
const W = 740, H = 78, pad = 8, bw = (W - pad*2) / N;
function q(s){ return s==null ? 0 : Math.max(0, 1 - Math.min(s, 1)); }   // taller = better
let bestSoFar = [], b = Infinity;
F.forEach(f => { if (f.valid && f.smape!=null) b = Math.min(b, f.smape); bestSoFar.push(b); });

function drawScrub(){
  const barMax = H - 20;
  let bars = F.map((f,i) => {
    const h = 4 + q(f.smape) * (barMax - 4);
    const isBest = (i === D.best - 1);
    const col = !f.valid ? 'var(--line)' : isBest ? 'var(--good)'
              : (i === cur ? 'var(--twin)' : 'var(--faint)');
    const op = (i === cur) ? 1 : .55;
    return `<rect class="bar" x="${pad + i*bw + 1.5}" y="${H-16-h}" width="${bw-3}"
       height="${h}" rx="1.5" fill="${col}" opacity="${op}" data-i="${i}"></rect>`
     + (i === cur ? `<rect x="${pad+i*bw}" y="2" width="${bw}" height="${H-12}" rx="2"
          fill="none" stroke="var(--twin)" stroke-dasharray="2 2" opacity=".8"></rect>` : '');
  }).join('');
  // best-so-far step line
  let pts = bestSoFar.map((s,i) => {
    const x = pad + i*bw + bw/2, y = H - 16 - (4 + q(s)*(barMax-4));
    return `${x},${y}`;
  });
  let line = '';
  for (let i=1;i<pts.length;i++){
    const [x0,y0] = pts[i-1].split(','), [x1] = pts[i].split(',');
    line += `<polyline points="${x0},${y0} ${x1},${y0} ${pts[i]}" fill="none"
       stroke="var(--reality)" stroke-width="1.4" opacity=".9"></polyline>`;
  }
  let axis = F.map((f,i) => `<text class="axis" x="${pad+i*bw+bw/2}" y="${H-3}"
     text-anchor="middle">${f.iter}</text>`).join('');
  svg.innerHTML = bars + line + axis;
  svg.querySelectorAll('rect.bar').forEach(r =>
    r.addEventListener('click', e => select(+e.target.dataset.i)));
}

const fillColor = v => v==null ? 'var(--line)'
  : v < 0.15 ? 'var(--good)' : v < 0.4 ? 'var(--twin)'
  : v < 0.8 ? 'var(--amber)' : 'var(--holdout)';

function render(){
  const f = F[cur];
  document.getElementById('frame').src = f.img;
  document.getElementById('cap-iter').textContent = f.iter;
  document.getElementById('cap-fam').textContent = f.family;
  document.getElementById('cap-smape').textContent = f.smape==null ? '—' : (+f.smape).toFixed(3);
  document.getElementById('detail').innerHTML = D.obs.map(k => {
    const v = f.per[k];
    const w = v==null ? 0 : Math.min(v, 1.5) / 1.5 * 100;
    return `<div class="metric"><span class="name">${D.obsLabel[k]}</span>
      <span class="track"><span class="fill" style="width:${w}%;background:${fillColor(v)}"></span></span>
      <span class="val">${v==null ? '—' : v.toFixed(2)}</span></div>`;
  }).join('');
  svg.setAttribute('aria-valuenow', f.iter);
  drawScrub();
}
function select(i){ cur = (i + N) % N; render(); }
document.getElementById('prev').onclick = () => { stop(); select(cur - 1); };
document.getElementById('next').onclick = () => { stop(); select(cur + 1); };
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft'){ stop(); select(cur - 1); }
  if (e.key === 'ArrowRight'){ stop(); select(cur + 1); }
});
const playBtn = document.getElementById('play');
function stop(){ playing = false; clearInterval(timer); playBtn.innerHTML = '&#9654;'; }
function start(){
  playing = true; playBtn.innerHTML = '&#9612;&#9612;';
  timer = setInterval(() => {
    if (cur >= N - 1){ stop(); return; }
    select(cur + 1);
  }, 900);
}
playBtn.onclick = () => playing ? stop() : start();

render();
</script>
"""

OUT.write_text(HTML.replace("__DATA__", json.dumps(DATA)))
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")
