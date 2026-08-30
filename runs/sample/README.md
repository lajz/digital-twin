# Sample run

A real `deepseek-v4-flash` run committed as a reference of the `runs/` layout:

```bash
uv run medusa run --iters 12
```

on the `synthetic-ecoli-fast` dataset (`--iters 12`, stopped early at iteration 6
once 3+ plausible families were found and the best score plateaued below target).

Highlights (`scorecard.json`):

- best holdout sMAPE **0.060** (`baranyi-robust-fixed`, a Baranyi-Roberts ODE) —
  recovered mu_max 1.95 /h (true 2.0), K 40 427 (true 40 000)
- 5 distinct plausible model families explored, `iters_to_target` = 2
- ~$0.011, 21.9k prompt + 11.8k response tokens

Note iteration 1 (`baranyi-robust`) forecast well but was **disqualified** for
exceeding the 5 s runtime budget (`differential_evolution`); iteration 2 is the agent
rewriting its own calibration to be faster after seeing that feedback.

Each `iter_NN/` keeps the prompt, the agent message, `twin.py`, `metrics.json`, and a
`forecast.png`; `portfolio/` holds the ranked top 3.
