# medusa

An **agentic feedback loop that generates digital twins**.

A DeepSeek-driven coding agent writes a *mechanistic* model of a real system plus its
own calibration code, a deterministic harness scores that model against real
observational data, and the score feeds back into the next iteration. Over many rounds
the loop returns a **ranked portfolio of distinct model families**.

**Demo #1** (this repo) twins the simplest real system: a well-mixed single-cell
microbial population — a bacterial growth curve — from public microscopy time-lapse data
reduced to `population_count` over time.

```
prompt ──▶ DeepSeek v4-flash ──▶ twin.py ──▶ harness (sandbox + score) ──▶ feedback
   ▲                                                                          │
   └──────────────────── archive summary + last attempt ◀─────────────────────┘
```

## Quickstart

```bash
uv sync

# build a dataset -> data/processed/  (synthetic by default; see "Real data" below)
uv run medusa build

# exercise the whole loop with canned twins -- no API key, no cost
uv run medusa run --dry-run --iters 4

# real run (needs DEEPSEEK_API_KEY in .env or the environment)
uv run medusa run --iters 20

# score the loop itself across the benchmark suite
uv run medusa bench --dry-run

# reprint the latest run's scorecard + portfolio
uv run medusa report
```

Output lands in `runs/<timestamp>-<dataset>/`:

| file | contents |
|---|---|
| `iter_NN/` | the prompt, raw agent message, `twin.py`, `metrics.json`, `forecast.png` |
| `trace.jsonl` | one machine-readable line per iteration |
| `leaderboard.json` | every candidate, ranked |
| `portfolio/` | best twin per family, top 3 — code, params, forecast plot, `portfolio.md` |
| `scorecard.json` | `LoopMetrics` — how the *loop* did (see below) |

## The twin contract

Every generated `twin.py` must define a `Twin` class with:

- `FAMILY` — slug naming the mechanistic family (used to group the portfolio)
- `PARAMS` — `{name: (prior_low, prior_high, unit)}`
- `fit(obs) -> dict` — the agent's own calibration, returns params within bounds
- `predict(params, time_s) -> np.ndarray` — deterministic forecast

Enforced by `medusa/contract/checker.py` before scoring: stdlib + `numpy` + `scipy`
only (no `sklearn`/`torch`/`pandas`/IO/network), mechanistic parameters only,
deterministic `predict`, `< 5 s` runtime. `medusa/contract/reference_twin.py` is a
hand-written logistic twin that serves as the baseline the agent must beat.

## Scoring

Per candidate (`medusa/harness/`): calibrate on the **fit window** (first 60% of the
series), forecast the **holdout window** (never shown to the agent), then compute
`holdout_smape` (primary rank key), `holdout_mase`, `fit_r2`, `aic`, and a
**plausibility** score — the model's implied doubling time vs. the datasheet's
ground-truth range. A candidate is disqualified if it crashes, is non-deterministic,
runs over budget, or is biologically implausible.

## The loop is built to score — and later improve — itself

`recursive self-improvement` is the end goal. The same pattern applies one level up:

| level | agent generates | scored against | fitness |
|---|---|---|---|
| inner (built) | `twin.py` | one dataset's holdout | `holdout_smape` + plausibility |
| meta (Milestone 7) | a `LoopConfig` | the benchmark suite | aggregate `LoopMetrics` |

So today: **every** loop knob lives in `config.LoopConfig` (prompt text included —
`agent/loop.py` has no literals of its own), every run emits `trace.jsonl`, and
`harness/scorecard.py::loop_scorecard()` reduces a run to `LoopMetrics`
(`best_holdout_smape`, `iters_to_target`, `distinct_plausible_families`,
`improvement_auc`, token cost, …). `medusa bench` runs the inner loop across
`bench/suite.py` and aggregates. The meta-loop is then additive: an agent proposes
`LoopConfig` diffs, `bench` scores them.

## Real data

**`ipb-ecoli`** is wired up end to end:

```bash
uv run medusa fetch --dataset ipb-ecoli     # ~65 KB CSV -> data/raw/
uv run medusa run   --dataset ipb-ecoli --iters 12
```

Real *E. coli* K-12 microcolony on an agar pad — 100 phase-contrast frames at 90 s
intervals, segmented by CellProfiler+Omnipose, reduced to cell count per frame
(110 → 1160 cells). From Ahmadi et al. 2024, *"A benchmarked comparison of software
packages for time-lapse image processing of monolayer bacterial population dynamics"*
(Microbiology Spectrum;
[github.com/ingallslab/ImageProcessing-Benchmarking](https://github.com/ingallslab/ImageProcessing-Benchmarking),
CC-BY 4.0). It's one automated pipeline's count, not a manual ground truth, and the
crop starts mid-growth (no lag/saturation phase) — so it's a growth-rate problem.
See `runs/sample/` for a real run: best held-out sMAPE **0.018**, implied doubling
time 30 min.

Other shortlisted datasets (`uv run medusa fetch` with no args prints how to get them):
DeLTA 2.0 agar-pad microcolony, Cell Tracking Challenge bacterial 2D sets, Tanouchi
et al. 2015. Drop any raw tracking CSV in `data/raw/` and reduce it with
`medusa.data.fetch.reduce_csv(...)` (point it at the time / count / area columns).

## Limitations

- **Sandboxing is best-effort**: candidate `twin.py` runs in a subprocess with a
  timeout, an AST import allowlist, a temp CWD and a scrubbed env — not a hard security
  boundary. Fine for local research; do not run untrusted configs unattended.
- DeepSeek pricing in the scorecard is a placeholder (`DEEPSEEK_PRICE_IN/OUT` to override).
- `deepseek-v4-flash` reasons by default; the client sends `thinking: {type: disabled}`
  unless `LoopConfig.thinking` is set (which also raises the token budget to fit the
  reasoning trace).
- Family de-duplication trusts the agent-declared `FAMILY` string.
- Demo #1 is well-mixed only — no spatial structure.
