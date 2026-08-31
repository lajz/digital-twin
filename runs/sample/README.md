# Sample run &mdash; L2 spatial twin of a real E. coli microcolony

A real `deepseek-v4-flash` run on **`ipb-ecoli-spatial`**:

```bash
uv run medusa fetch --dataset ipb-ecoli-spatial     # ~43 MB per-object CSV -> data/raw/
uv run medusa run   --dataset ipb-ecoli-spatial --iters 12 --min-families 4
uv run medusa demo runs/sample                       # rebuild demo.html
```

The twin is an **agent-based rod simulator**: every cell is a capsule (x, y, angle,
length); cells elongate, divide, and push each other. It is calibrated on the fit-window
frames, **initialised from the real first frame**, and rolled forward. It is scored on
spatial summary-statistic time series over the **holdout** window only (cell count,
total rod length, radius of gyration, colony aspect, nematic order, mean
nearest-neighbour distance) &mdash; never on pixels, never on held-out frames.

## Result (`scorecard.json`)

- best combined holdout sMAPE **0.143** (`stochastic-rod-population`, iteration 6) &mdash;
  edging past the hand-written reference simulator (0.146)
- that twin matches count (0.12), radius of gyration (0.009), aspect (0.04) and
  nearest-neighbour packing (0.07); it is weakest on nematic order (0.55)
- **6 distinct plausible families**, valid rate 1.0, no crashes, improvement AUC 0.83
- ~$0.041, 125k tokens

The portfolio shows the tradeoff: iteration 1 (`overdamped-rods`) reproduces local
alignment (nematic 0.10) but not packing (0.54); iteration 6 reproduces packing (0.07)
but not alignment (0.55). Different mechanisms capture different facets of the real
spatial structure &mdash; which is what the ranked portfolio is for.

## Files

- `demo/iter_NN.png` &mdash; per-iteration side-by-side: reality (top) vs the twin's
  rollout (bottom), columns spanning the movie, holdout frames labelled red
- `portfolio/NN_<family>/` &mdash; the top-3 twins: `twin.py`, params, metrics, comparison
- `iter_NN/{twin.py, agent_msg.md, metrics.json}` &mdash; every candidate
- `trace.jsonl`, `leaderboard.json`, `scorecard.json`

`rollout.npz`, per-iter `comparison.png`, `prompt.md` and the ~3 MB embedded `demo.html`
were stripped to keep the repo light; `uv run medusa demo runs/sample` regenerates them.
