# Sample run — real E. coli microscopy data

A real `deepseek-v4-flash` run on the **`ipb-ecoli`** dataset:

```bash
uv run medusa fetch --dataset ipb-ecoli
uv run medusa run   --dataset ipb-ecoli --iters 12
```

*E. coli* K-12 microcolony on an agar pad — 100 phase-contrast frames at 90 s,
CellProfiler+Omnipose cell counts (110 → 1160 cells over ~2.5 h). Stopped early at
iteration 10 once 3+ plausible families were found and the best score plateaued.

Highlights (`scorecard.json`):

- best held-out sMAPE **0.018** (`two-stage-lag-adaptation`, a Baranyi-style ODE with
  an adaptation term) — fit R² 0.995
- **implied doubling time 30 min** — squarely in the literature range for E. coli on
  agar pads (the plausibility anchor)
- 9 distinct plausible model families explored, `iters_to_target` = 3
- ~$0.018, 36.5k prompt + 18.1k response tokens

Notable: iteration 1 (`logistic`) crashed on the real data; iterations 5/8/9 proposed
families that collapsed to near-constant forecasts (sMAPE ≈ 1.23) and ranked last. The
top 3 are all lag/adaptation variants — they capture the mild deceleration in the real
curve that a plain exponential (iter 2, sMAPE 0.34) misses.

Per-iteration `forecast.png` and `prompt.md` were stripped to keep the repo light; a
live run writes them under each `iter_NN/`.
