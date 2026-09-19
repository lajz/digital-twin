# medusa

An **agentic feedback loop that generates digital twins**.

A DeepSeek-driven coding agent writes a *mechanistic* model of a real system plus its
own calibration code, a deterministic harness scores that model against real
observational data, and the score feeds back into the next iteration. Over many rounds
the loop returns a **ranked portfolio of distinct model families**.

The loop is **domain-agnostic** — a `Domain` adapter (`src/medusa/domains/`) supplies the
only three subject-specific pieces: the data reduction, what's scored (`Task`), and the
agent's instructions. `uv run medusa domains` lists them.

**Bacterial microcolony** (real public microscopy), at three fidelity levels:

| level | domain | twin state | scored on |
|---|---|---|---|
| **L0** population | `ipb-ecoli` | scalar `N(t)` | count vs time |
| **L1** size-structured | `ipb-ecoli-structured` | cell-size distribution | count, biomass, mean length, CV |
| **L2** spatial | `ipb-ecoli-spatial` | every cell a rod `(x, y, θ, ℓ)` | count, biomass, radius of gyration, aspect, nematic order, neighbour distance — **plus side-by-side renders** |

**A SaaS business** (`saas-seed`, synthetic): the twin models customer acquisition /
churn / expansion / headcount / cash, is handed the company's *planned* levers
(marketing spend, capital raised) over the whole horizon, and must produce a forecast
that both fits the history and **obeys the accounting identities exactly** —
`cash[t] = cash[t-1] + net_income[t] + capital_raised[t]`, `customers[t] ≤ customers[t-1]
+ new_customers[t]`. Those are hard constraints (`Domain.constraints`): a twin that
breaks one is rejected, unscored — free plausibility, no tuning.

Model fidelity has to match data fidelity: each level/domain feeds the agent more of the
data and asks for a more mechanistic model.

```
prompt ──▶ DeepSeek v4-flash ──▶ twin.py ──▶ harness (sandbox + score) ──▶ feedback
   ▲                                                                          │
   └──────────────────── archive summary + last attempt ◀─────────────────────┘
```

## Quickstart

```bash
uv sync

# exercise the whole loop with canned twins -- no API key, no cost
uv run medusa run --dry-run --iters 4

uv run medusa domains                # list registered domain adapters

# real runs (need DEEPSEEK_API_KEY in .env or the environment)
uv run medusa run --dataset ipb-ecoli          --iters 12   # L0
uv run medusa run --dataset ipb-ecoli-structured --iters 12  # L1
uv run medusa run --dataset ipb-ecoli-spatial  --iters 10    # L2  -> runs/<ts>/demo.html
uv run medusa run --dataset saas-seed          --iters 12   # a SaaS business

uv run medusa bench --dry-run        # score the loop itself across the suite
uv run medusa report                 # reprint the latest run's scorecard + portfolio
```

Every run writes `runs/<ts>/demo.html` — a self-contained page that scrubs through the
training iterations showing the twin's forecast (L0/L1) or its rollout beside the real
microscopy (L2).

Output lands in `runs/<timestamp>-<dataset>/`:

| file | contents |
|---|---|
| `iter_NN/` | the prompt, raw agent message, `twin.py`, `metrics.json`, `forecast.png` |
| `trace.jsonl` | one machine-readable line per iteration |
| `leaderboard.json` | every candidate, ranked |
| `portfolio/` | best twin per family, top 3 — code, params, forecast plot, `portfolio.md` |
| `scorecard.json` | `LoopMetrics` — how the *loop* did (see below) |

## Development

```bash
git config core.hooksPath .githooks   # one-time: install the pre-push hook
```

`git push` then runs `pytest` (blocking) followed by `medusa-review` (advisory,
never blocks): a DeepSeek pass over `git diff --merge-base <base> HEAD` covering
correctness/tests/simplification and a separate security pass focused on the
harness sandbox, path handling under `runs/`/`.cache/`, and secret handling. No
`DEEPSEEK_API_KEY` ⇒ the AI review is silently skipped; the test gate still runs.

```bash
uv run medusa-review                     # same as the hook's advisory pass
uv run medusa-review --base origin/feedback-loop-digital-twin --min medium
uv run medusa-review --security-only
uv run medusa-review --fail-on high      # exit non-zero on a high finding
```

Escape hatches: `SKIP_REVIEW_GATE=1 git push` skips both checks, `SKIP_AI_REVIEW=1
git push` skips only the AI pass, `git push --no-verify` skips the hook entirely.
Config precedence is defaults < `.medusa-review.json` (committed) < environment <
CLI flags; `MEDUSA_REVIEW_MODEL` / `MEDUSA_REVIEW_BASE_URL` override the model, and
`MEDUSA_REVIEW_API_KEY` gives it a separate key/budget from the main loop's
`DEEPSEEK_API_KEY`. Local-only for now — no GitHub Action / inline PR comments yet.

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
runs over budget, is implausible, or **violates a domain constraint** (an identity that
must hold — a cash balance, a conservation law).

## Adding a domain

Everything subject-specific lives behind one interface (`medusa/domains/base.py`):

```python
from medusa.domains import register
from medusa.domains.base import Domain, flow_balance, non_negative

register(Domain(
    name="my-thing",
    task=Task(name="my-thing", observables=(...), weights={...},
              plausibility={...}, exogenous=(...), period_label="mo"),
    build=my_build_fn,               # raw data -> Dataset (observations + split)
    system_prompt=MY_SYSTEM_PROMPT,  # the agent's instructions for this subject
    kind="business",
    constraints=(non_negative("x"), flow_balance("cash", "net_income")),
))
```

The loop, sandbox, archive, portfolio, scorecard, renderers and `medusa play/demo` are
all generic. `medusa/domains/{ecoli,saas,synthetic_growth}.py` are the built-ins.

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

Real-run results:
- **L0** `ipb-ecoli` — best held-out sMAPE 0.018, implied doubling time 30 min
- **L2** `ipb-ecoli-spatial` — best combined held-out sMAPE **0.143**
  (`stochastic-rod-population`), 6 model families, matching count / radius of gyration /
  packing on held-out frames; `runs/sample/` is this run (`medusa demo runs/sample` for
  the side-by-side viewer).

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
