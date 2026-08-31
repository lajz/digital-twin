"""Write the one-generation checkpoint, the final report, and a small meta.html."""

from __future__ import annotations

import json
from pathlib import Path


def _s(score) -> str:
    return (f"mean best sMAPE **{score.mean_best_smape:.3f}**, worst "
            f"{score.worst_best_smape:.3f}, {score.mean_distinct_families:.1f} families, "
            f"${score.mean_usd_cost:.3f}/dataset, valid {score.mean_valid_rate:.0%}")


def write_checkpoint(meta_dir: Path, base, seed, seed_val, seed_val_rerun, noise_band,
                     digest: str, child, child_entry, msg: str) -> Path:
    sv = seed_val.mean_best_smape
    sr = seed_val_rerun.mean_best_smape
    cv = child_entry.val.mean_best_smape
    delta = sv - cv                     # positive = child better
    feasible = child_entry.val.feasible

    if not feasible:
        verdict = ("**PROPOSAL PATH HAS ISSUES** — the child is infeasible "
                   f"({child_entry.val.mean_distinct_families:.1f} families, "
                   f"valid {child_entry.val.mean_valid_rate:.0%}). Fix the constraints / "
                   "prompt before trusting the search.")
    elif noise_band > 0.02:
        verdict = (f"**NOISE-LIMITED** — re-running the seed unchanged moved META_VAL by "
                   f"{noise_band:.3f}. Any single-generation gain smaller than ~{2*noise_band:.3f} "
                   "is not real. Widen the suite, average 2+ inner seeds per dataset, or "
                   "drop the inner temperature before running a long search.")
    elif delta > 2 * noise_band + 1e-4:
        verdict = (f"**HILL-CLIMB IS WORKING** — child improves META_VAL by {delta:.3f}, "
                   f"clear of the {noise_band:.3f} noise band. Safe to run "
                   "`medusa meta --generations 8`.")
    elif delta < -2 * noise_band:
        verdict = ("**ONE STEP DOWN** — child is worse, but that's expected sometimes; the "
                   "archive keeps the seed. If several generations only go down, revisit the "
                   "objective or the component menu.")
    else:
        verdict = (f"**INCONCLUSIVE** — child moved META_VAL by {delta:+.3f}, within the "
                   f"{noise_band:.3f} noise band. One generation isn't enough signal; run "
                   "3-5 more and watch the trend, or make the digest sharper.")

    md = f"""# meta checkpoint — one generation

## Signal vs noise (META_VAL, mean best sMAPE)

| | score | vs seed |
|---|---|---|
| seed | {sv:.4f} | — |
| seed re-run (nothing changed) | {sr:.4f} | {sv - sr:+.4f}  ← **noise band {noise_band:.4f}** |
| child (`{child.touched}`) | {cv:.4f} | {delta:+.4f} |

## Verdict

{verdict}

## What the meta-agent changed

- touched: `{child.touched}` — {child.rationale}

{child.diff_summary(base)}

child feasible: {feasible} · child train: {_s(child_entry.train)} · child val: {_s(child_entry.val)}

## The reflection it acted on

{digest}

---
Full artefacts under `{meta_dir}/`. Re-run with `medusa meta --generations 8` only once
the verdict above is green.
"""
    p = meta_dir / "meta_checkpoint.md"
    p.write_text(md)
    return p


def write_report(meta_dir: Path, base, archive, seed_val, seed_val_rerun, noise_band) -> Path:
    best = archive.best_on_val()
    rows = archive.leaderboard()
    lines = ["# meta-loop report", ""]
    lines.append(f"Genomes evaluated: {len(archive.entries)} · "
                 f"seed META_VAL noise band: {noise_band:.4f}")
    lines.append("")
    lines.append("## Pareto front (META_VAL)")
    for e in sorted(archive.front(), key=lambda e: e.val.mean_best_smape):
        g = e.genome
        lines.append(f"- `{g.genome_id}` gen {g.generation}, touched "
                     f"`{g.touched or 'seed'}` — {_s(e.val)}")
    lines.append("")
    if best is not None:
        lines.append("## Best on META_VAL, checked once on META_TEST")
        lines.append(f"- genome `{best.genome.genome_id}` (touched `{best.genome.touched or 'seed'}`)")
        lines.append(f"- TRAIN: {_s(best.train)}")
        lines.append(f"- VAL:   {_s(best.val)}")
        if best.test is not None:
            lines.append(f"- TEST:  {_s(best.test)}")
            gap = best.test.mean_best_smape - best.val.mean_best_smape
            flag = "  ⚠️ possible overfit to the meta-suite" if gap > 0.05 else ""
            lines.append(f"- VAL→TEST gap: {gap:+.3f}{flag}")
        seed_gain = seed_val.mean_best_smape - best.val.mean_best_smape
        lines.append("")
        lines.append(f"**Net vs the committed baseline: {seed_gain:+.3f} META_VAL mean sMAPE** "
                     f"(noise band {noise_band:.3f}).")
    lines.append("")
    lines.append("## Leaderboard")
    for r in rows:
        g = r["genome"]
        lines.append(f"- `{g['genome_id']}` gen {g['generation']} `{g['touched'] or 'seed'}`: "
                     f"val mean {r['val']['mean_best_smape']:.3f}, "
                     f"train mean {r['train']['mean_best_smape']:.3f}")
    p = meta_dir / "meta_report.md"
    p.write_text("\n".join(lines))
    _write_html(meta_dir, archive, seed_val, noise_band)
    return p


def _write_html(meta_dir: Path, archive, seed_val, noise_band) -> None:
    gens = []
    for e in archive.entries:
        gens.append({
            "id": e.genome.genome_id, "gen": e.genome.generation,
            "touched": e.genome.touched or "seed", "rationale": e.genome.rationale,
            "train": round(e.train.mean_best_smape, 4),
            "val": round(e.val.mean_best_smape, 4),
            "test": round(e.test.mean_best_smape, 4) if e.test else None,
            "cost": round(e.val.mean_usd_cost, 4),
            "families": round(e.val.mean_distinct_families, 2),
            "feasible": e.val.feasible,
            "front": e in archive.front(),
        })
    data = {"noise_band": round(noise_band, 4), "seed_val": round(seed_val.mean_best_smape, 4),
            "gens": gens}
    (meta_dir / "meta.html").write_text(_HTML.replace("__DATA__", json.dumps(data)))


_HTML = """<title>medusa meta-loop</title>
<style>
 :root{--bg:#0f1216;--fg:#e7e9ec;--mut:#8b95a1;--line:#262b31;--good:#4cbd94;--acc:#5fa8d3;--warn:#d69a4a}
 @media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--fg:#161a1d;--mut:#5a636c;--line:#dde1e6}}
 body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0}
 .wrap{max-width:900px;margin:0 auto;padding:32px 20px 64px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:var(--mut);margin-bottom:22px}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
 tr.front td{background:color-mix(in srgb,var(--good) 10%,transparent)}
 .id{font-family:ui-monospace,monospace;color:var(--acc)}
 .bad{color:var(--warn)} .dim{color:var(--mut)}
 .band{margin:14px 0 26px;padding:12px 14px;border:1px solid var(--line);border-radius:8px}
</style>
<div class="wrap">
 <h1>medusa meta-loop</h1>
 <div class="sub" id="sub"></div>
 <div class="band" id="band"></div>
 <table><thead><tr><th>gen</th><th>touched</th><th>train</th><th>val</th><th>test</th>
  <th>families</th><th>$/ds</th><th></th></tr></thead><tbody id="rows"></tbody></table>
 <p class="dim" id="rats" style="margin-top:22px"></p>
</div>
<script>
const D=__DATA__;
document.getElementById('sub').textContent=
  `${D.gens.length} genomes · seed META_VAL ${D.seed_val} · noise band ±${D.noise_band}`;
document.getElementById('band').innerHTML=
  `A single-generation META_VAL gain below <b>${(2*D.noise_band).toFixed(4)}</b> `+
  `(2× the noise band) is not distinguishable from run-to-run variance.`;
document.getElementById('rows').innerHTML=D.gens.map(g=>{
  const t=g.test==null?'<span class=dim>—</span>':g.test;
  return `<tr class="${g.front?'front':''}"><td>${g.gen}</td>`+
   `<td><span class=id>${g.touched}</span></td><td>${g.train}</td>`+
   `<td><b>${g.val}</b></td><td>${t}</td><td>${g.families}</td><td>${g.cost}</td>`+
   `<td>${g.feasible?'':'<span class=bad>infeasible</span>'}</td></tr>`;
}).join('');
document.getElementById('rats').innerHTML=D.gens.filter(g=>g.rationale).map(g=>
  `<b>${g.touched}</b>: ${g.rationale}`).join('<br>');
</script>"""
