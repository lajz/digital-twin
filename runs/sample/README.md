# Sample run (dry-run)

Committed reference of the `runs/` layout. Produced by:

```bash
uv run medusa run --dry-run --iters 4
```

`--dry-run` replaces the DeepSeek API with the canned twins in
`medusa/agent/canned.py` (logistic, gompertz, baranyi-ode, richards), so the scores
here reflect those fixed models on `synthetic-ecoli-fast`, not a real agent search.
Token counts and `usd_cost` in `scorecard.json` are synthetic.

Per-iteration `forecast.png` and `prompt.md` were removed to keep the repo light; a
live run writes them under each `iter_NN/`.
