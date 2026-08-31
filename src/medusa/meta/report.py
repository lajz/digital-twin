"""Write the one-generation checkpoint, the final report, and the meta.html dashboard."""

from __future__ import annotations

import json
from pathlib import Path


# --- meta.html dashboard: read the whole meta_dir tree -------------------------

_STAGE_ORDER = ("seed_train", "seed_val", "seed_val_rerun", "best_test")
_DOT = {"ok": "good", "no_code_block": "bad", "crashed": "bad"}


def _stage_runs(stage_dir: Path) -> list[Path]:
    return sorted(p.parent for p in stage_dir.glob("*/*/trace.jsonl"))


def _run_rows(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines() if l.strip()]


def _iter_strip(rows: list[dict]) -> dict:
    dots, best, run_best = [], [], None
    for r in rows:
        st = str(r.get("status", "?"))
        kind = "good" if r.get("is_valid") else ("mid" if st.startswith("constraint") else "bad")
        dots.append({"kind": kind, "status": st[:60], "family": r.get("family"),
                     "smape": r.get("holdout_smape")})
        s = r.get("holdout_smape")
        if r.get("is_valid") and isinstance(s, (int, float)):
            run_best = s if run_best is None else min(run_best, s)
        best.append(run_best)
    return {"dots": dots, "best_curve": best,
            "final_best": run_best, "n": len(rows),
            "families": len({d["family"] for d in dots if d["kind"] == "good" and d["family"]})}


def _collect(meta_dir: Path) -> dict:
    meta_dir = Path(meta_dir)
    cfg = json.loads((meta_dir / "meta_config.json").read_text()) if (meta_dir / "meta_config.json").exists() else {}
    seed = json.loads((meta_dir / "seed.json").read_text()) if (meta_dir / "seed.json").exists() else {}

    stages = []
    stage_dirs = [d for d in sorted(meta_dir.iterdir())
                  if d.is_dir() and (d.name in _STAGE_ORDER or d.name.startswith("gen_"))]
    for sd in stage_dirs:
        runs = _stage_runs(sd)
        if not runs:
            continue
        per = {}
        for run in runs:
            meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
            per[meta.get("dataset", run.name)] = _iter_strip(_run_rows(run))
        stages.append({"stage": sd.name, "datasets": per})

    gens = []
    for gd in sorted(meta_dir.glob("gen_[0-9][0-9]")):
        out = {}
        if (gd / "outcome.json").exists():
            out = json.loads((gd / "outcome.json").read_text())
        gens.append({
            "gen": gd.name, "digest": (gd / "digest.md").read_text() if (gd / "digest.md").exists() else "",
            "rejected": out.get("rejected"), "touched": out.get("touched"),
            "diff": out.get("diff"),
            "val_mean": (out.get("val") or {}).get("mean_best_smape"),
            "train_mean": (out.get("train") or {}).get("mean_best_smape"),
        })

    return {
        "meta_dir": meta_dir.name,
        "noise_band": seed.get("noise_band"),
        "seed_val": (seed.get("val") or {}).get("mean_best_smape"),
        "seed_val_rerun": (seed.get("val_rerun") or {}).get("mean_best_smape"),
        "seed_per": (seed.get("val") or {}).get("per_dataset", {}),
        "rerun_per": (seed.get("val_rerun") or {}).get("per_dataset", {}),
        "config": cfg,
        "stages": stages,
        "gens": gens,
    }


def build_meta_html(meta_dir: str | Path) -> Path:
    meta_dir = Path(meta_dir)
    (meta_dir / "meta.html").write_text(
        _DASHBOARD.replace("__DATA__", json.dumps(_collect(meta_dir), default=str))
    )
    return meta_dir / "meta.html"


def _s(score) -> str:
    return (f"mean best sMAPE **{score.mean_best_smape:.3f}**, worst "
            f"{score.worst_best_smape:.3f}, {score.mean_distinct_families:.1f} families, "
            f"${score.mean_usd_cost:.3f}/dataset, valid {score.mean_valid_rate:.0%}")


def _per_table(seed_per: dict, rerun_per: dict) -> str:
    names = sorted(set(seed_per) | set(rerun_per))
    rows = ["| dataset | seed best | re-run best | Δ (noise) | valid seed→re-run |",
            "|---|---|---|---|---|"]
    for n in names:
        a, b = seed_per.get(n, {}), rerun_per.get(n, {})
        sb, rb = a.get("best"), b.get("best")
        d = (f"{abs(sb - rb):.3f}" if isinstance(sb, (int, float)) and isinstance(rb, (int, float))
             else "—")
        rows.append(f"| {n} | {_f(sb)} | {_f(rb)} | {d} | "
                    f"{_f(a.get('valid'))} → {_f(b.get('valid'))} |")
    return "\n".join(rows)


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def write_checkpoint(meta_dir: Path, base, seed, seed_val, seed_val_rerun, noise_band, *,
                     seed_val_per: dict, seed_rerun_per: dict, digest: str,
                     child, child_entry, reject: str | None) -> Path:
    sv = seed_val.mean_best_smape
    sr = seed_val_rerun.mean_best_smape

    if reject is not None or child_entry is None:
        cv_line = f"| child | — (rejected: {reject}) | — |"
        delta = None
        verdict = (f"**PROPOSAL DIDN'T LAND** — {reject}. The meta-agent's reply couldn't "
                   "be turned into a valid single-component change. Fix the output-format "
                   "instructions / parser, then re-run `--generations 1`.")
    else:
        cv = child_entry.val.mean_best_smape
        delta = sv - cv
        cv_line = f"| child (`{child.touched}`) | {cv:.4f} | {delta:+.4f} |"
        if not child_entry.val.feasible:
            verdict = (f"**CHILD INFEASIBLE** — {child_entry.val.mean_distinct_families:.1f} "
                       f"families / valid {child_entry.val.mean_valid_rate:.0%} on VAL. "
                       "The change hurt robustness; tighten the component menu or the digest.")
        elif noise_band > 0.02:
            verdict = (f"**NOISE-LIMITED** — re-running the seed unchanged moved META_VAL by "
                       f"{noise_band:.3f}; a real single-generation gain must exceed "
                       f"~{2*noise_band:.3f}. Child moved it {delta:+.3f}. Per-dataset table "
                       "below shows where the variance lives — pull that dataset out of VAL, "
                       "lower the inner temperature further, or average 2+ inner seeds.")
        elif delta > 2 * noise_band + 1e-4:
            verdict = (f"**HILL-CLIMB IS WORKING** — child improves META_VAL by {delta:.3f}, "
                       f"clear of the {noise_band:.3f} noise band. Safe to run "
                       "`medusa meta --generations 8`.")
        elif delta < -2 * noise_band:
            verdict = ("**ONE STEP DOWN** — child is worse; the archive keeps the seed. Fine "
                       "for one step, but if the digest points somewhere useful and the "
                       "change still hurts, the objective may be off.")
        else:
            verdict = (f"**INCONCLUSIVE** — child moved META_VAL {delta:+.3f}, inside the "
                       f"{noise_band:.3f} noise band. Sharpen the digest or reduce noise; "
                       "one generation isn't enough signal yet.")

    diff = child.diff_summary(base) if child is not None else "(none)"
    ratl = child.rationale if child is not None else "—"
    touched = child.touched if child is not None else "—"
    child_scores = (f"child train: {_s(child_entry.train)} · child val: {_s(child_entry.val)}"
                    if child_entry is not None else "")

    md = f"""# meta checkpoint — one generation

## Signal vs noise (META_VAL, mean best sMAPE)

| | score | vs seed |
|---|---|---|
| seed | {sv:.4f} | — |
| seed re-run (nothing changed) | {sr:.4f} | {sv - sr:+.4f}  ← **noise band {noise_band:.4f}** |
{cv_line}

### Where the noise lives (per VAL dataset)

{_per_table(seed_val_per, seed_rerun_per)}

## Verdict

{verdict}

## What the meta-agent changed

- touched: `{touched}` — {ratl}

{diff}

{child_scores}

## The reflection it acted on

{digest}

---
Full artefacts under `{meta_dir}/`.
"""
    p = meta_dir / "meta_checkpoint.md"
    p.write_text(md)
    try:
        build_meta_html(meta_dir)
    except Exception:  # pragma: no cover - dashboard is best-effort
        pass
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
    build_meta_html(meta_dir)
    return p


_DASHBOARD = r"""<title>medusa meta-loop</title>
<style>
 :root{--bg:#0e1116;--fg:#e7e9ec;--mut:#8b95a1;--line:#242a31;--card:#151a20;
   --good:#4cbd94;--mid:#d69a4a;--bad:#e0654a;--acc:#5fa8d3}
 @media(prefers-color-scheme:light){:root{--bg:#f5f7f9;--fg:#141a1d;--mut:#5a636c;
   --line:#dde1e6;--card:#fff}}
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,sans-serif;margin:0}
 .wrap{max-width:1000px;margin:0 auto;padding:34px 20px 80px}
 h1{font-size:20px;margin:0 0 3px} h2{font-size:14px;text-transform:uppercase;
   letter-spacing:.06em;color:var(--mut);margin:34px 0 12px;font-weight:600}
 .sub{color:var(--mut);margin-bottom:20px;font-variant-numeric:tabular-nums}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
 .band{margin-bottom:24px} .band b{color:var(--fg)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
 .mono{font-family:ui-monospace,monospace}
 .ds{margin:10px 0 18px}
 .ds h3{margin:0 0 6px;font-size:13px;font-family:ui-monospace,monospace;color:var(--acc)}
 .strip{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
 .lane{display:flex;flex-direction:column;gap:4px}
 .lane .lbl{font-size:11px;color:var(--mut)}
 .dots{display:flex;gap:3px}
 .dot{width:13px;height:13px;border-radius:3px;cursor:default}
 .dot.good{background:var(--good)} .dot.mid{background:var(--mid)} .dot.bad{background:var(--bad)}
 .num{font-family:ui-monospace,monospace;font-size:12px}
 .digest{white-space:pre-wrap;font:12.5px/1.5 ui-monospace,monospace;color:var(--fg)}
 .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin:0 4px}
</style>
<div class="wrap">
 <h1>medusa meta-loop</h1>
 <div class="sub" id="sub"></div>
 <div class="card band" id="band"></div>

 <h2>Signal vs noise — seed on META_VAL, run twice unchanged</h2>
 <table id="noise"><thead><tr><th>dataset</th><th>seed best</th><th>re-run best</th>
   <th>Δ (noise)</th><th>valid rate</th></tr></thead><tbody></tbody></table>

 <h2>Every inner run — <span class="swatch" style="background:var(--good)"></span>valid
   <span class="swatch" style="background:var(--mid)"></span>constraint
   <span class="swatch" style="background:var(--bad)"></span>failed · one square per iteration</h2>
 <div id="stages"></div>

 <h2>Generations</h2>
 <div id="gens"></div>
</div>
<script>
const D = __DATA__;
const f = x => (x==null ? "—" : (+x).toFixed(4));

document.getElementById("sub").textContent =
  `${D.meta_dir} · seed META_VAL ${f(D.seed_val)} → re-run ${f(D.seed_val_rerun)} `+
  `· noise band ±${f(D.noise_band)}`;
document.getElementById("band").innerHTML = D.noise_band==null ? "run in progress…" :
  `A single-generation META_VAL improvement below <b>${(2*D.noise_band).toFixed(4)}</b> `+
  `(2× the noise band) can't be told apart from run-to-run variance. The hill-climb `+
  `only has signal once a child clears that.`;

const names = [...new Set([...Object.keys(D.seed_per), ...Object.keys(D.rerun_per)])].sort();
document.querySelector("#noise tbody").innerHTML = names.map(n=>{
  const a=D.seed_per[n]||{}, b=D.rerun_per[n]||{};
  const d = (a.best!=null&&b.best!=null) ? Math.abs(a.best-b.best).toFixed(3) : "—";
  const hot = d!=="—" && +d > 0.03 ? ' style="color:var(--bad)"' : '';
  return `<tr><td class=mono>${n}</td><td class=num>${f(a.best)}</td>`+
    `<td class=num>${f(b.best)}</td><td class=num${hot}>${d}</td>`+
    `<td class=num>${f(a.valid)} → ${f(b.valid)}</td></tr>`;
}).join("");

const dot = d => `<span class="dot ${d.kind}" title="${d.status}${d.family?' · '+d.family:''}${d.smape!=null?' · '+(+d.smape).toFixed(3):''}"></span>`;
document.getElementById("stages").innerHTML = D.stages.map(s=>{
  const dsList = Object.entries(s.datasets).map(([name,r])=>
    `<div class="ds"><h3>${name}</h3><div class="dots">${r.dots.map(dot).join("")}</div>`+
    `<div class="num" style="color:var(--mut);margin-top:3px">best ${r.final_best==null?"—":(+r.final_best).toFixed(3)} · ${r.families} famil${r.families==1?"y":"ies"} · ${r.n} iters</div></div>`
  ).join("");
  return `<div style="margin-bottom:20px"><div class="lbl mono" style="color:var(--mut);margin-bottom:2px">${s.stage}</div>${dsList}</div>`;
}).join("");

document.getElementById("gens").innerHTML = D.gens.length ? D.gens.map(g=>{
  const head = g.rejected
    ? `<b style="color:var(--bad)">${g.gen} — rejected:</b> ${g.rejected}`
    : `<b>${g.gen}</b> touched <span class=mono style="color:var(--acc)">${g.touched}</span> `+
      `— train ${f(g.train_mean)} · val ${f(g.val_mean)}`;
  return `<div class="card" style="margin-bottom:12px">${head}`+
    (g.diff?`<div class="num" style="color:var(--mut);margin:6px 0">${g.diff.replace(/\n/g,"<br>")}</div>`:"")+
    `<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--mut)">reflection</summary>`+
    `<div class="digest">${(g.digest||"").replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</div></details></div>`;
}).join("") : "<div class='card' style='color:var(--mut)'>none yet</div>";
</script>"""
