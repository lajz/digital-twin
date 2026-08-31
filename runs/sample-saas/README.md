# Sample run — digital twin of a SaaS business

A real `deepseek-v4-flash` run on the **`saas-seed`** domain (synthetic seed-stage SaaS):

```bash
uv run medusa run --dataset saas-seed --iters 12 --min-families 4
```

Same feedback loop as the E. coli twins — a different subject. The twin models customer
acquisition / churn / expansion / headcount / cash. It is handed the company's **plans**
(`marketing_spend`, `capital_raised`) over the whole horizon as exogenous inputs, fit on
the first 29 months, and scored on the held-out final 19 — MRR, customers, net income,
cash, headcount, new customers.

## Result (`scorecard.json`)

- best combined holdout sMAPE **0.076** (`cohort-retention`, iteration 7) — MRR 0.034,
  customers 0.030, cash 0.046 on held-out months
- **4 distinct model families** (cohort-retention, funnel-conversion,
  ltv-cac-steady-state, bass-diffusion-churn), `iters_to_target` = 7, ~$0.031
- recovered `acq_per_dollar` 0.0109 (true 0.011), `arpa0` $92 (true $90), and — from the
  fit window alone — **inferred the month-30 price increase** at 1.28× (true 1.2×)
- **zero constraint violations**: every valid twin computed cash as
  `cash[t-1] + net_income[t] + capital_raised[t]` and kept net customer adds ≤ gross
  adds — the accounting identities are hard constraints, so a twin that fits cash as a
  free curve is rejected unscored

`best_forecast.png` shows the twin catching all three exogenous events — the Series A
(month 12), the price step (month 30), and the marketing budget cut (month 36).

Trimmed to `trace.jsonl` + `scorecard.json` + `portfolio/` + per-iteration
`twin.py` / `metrics.json` / `agent_msg.md`.
