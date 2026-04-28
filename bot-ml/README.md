# Bot ML Evidence

This directory contains the bot machine-learning files from the playable Business Empire app. It includes the policy reports, benchmark logic, and trained model artifact.

## What's here

- **`reports/`** — benchmark reports
  - `bot-ml-summary.md`: seat-rotated 400-game benchmark, medium baseline vs hard tuned vs expert ML
  - `benchmark-audit.md`: the story of the first benchmark failure and how it was repaired
  - `bot-decision-basis.md`: contract for what each policy can and cannot do
- **`artifacts/`** — optimized policy and ML model JSONs the playable app consumes
  - `optimized-hard-policy.json`: tuned heuristic weights (currently the strongest policy)
  - `optimized-expert-model.json`: 10-feature standardized linear scorer
  - `expert-model-report.json`: feature importance and validation metrics
- **`source-snapshot/`** — five TypeScript files for the bot decision code
  - `beBotPolicy.ts`: deterministic heuristic and ML scoring entry point
  - `beBotModel.generated.ts`: code-generated wrapper around the linear scorer
  - `beBotPolicy.spec.ts`: deterministic-replay tests
  - `be-bot-selfplay.ts`: seat-rotated benchmark harness
  - `be-bot-optimize.ts`: offline tuning script

Card art for the Streamlit app lives in `app/streamlit_assets/cards/` at the repository root.

## Why not include the full app

The full app contains native build configs and unrelated screens. This folder keeps the files that explain the bot decisions without requiring the full Expo project.

## Headline result

| Policy | Win rate | Avg score | Score diff vs same-seat medium | Invalid actions |
|---|---:|---:|---:|---:|
| randomDiagnostic | 2.3% | 8.95 | −18.20 | 0 |
| medium (naive baseline) | 25.0% | 27.15 | 0.00 | 0 |
| hard (tuned heuristic) | 61.3% | 35.66 | +8.51 | 0 |
| expert (ML scorer) | 39.8% | 29.52 | +2.37 | 0 |

Interpretation: hard outperforms expert. Both beat the baseline. In this small action-ranking problem, the tuned heuristic performed better than the linear ML scorer.
