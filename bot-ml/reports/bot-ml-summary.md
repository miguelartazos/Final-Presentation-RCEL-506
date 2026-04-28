# Bot ML Seat-Rotated Benchmark Summary

Generated: 2026-04-28T04:22:28.537Z
Rules version: be-mvp-rules-v3
Card data version: app-card-registry
Model version: expert-linear-v0.3-seat-rotated-mc40
Benchmark status: passed

## Fair Comparison

| Policy | Games | Win rate | Avg rank | Avg score | Score diff vs same-seat medium | Invalid actions | Effective actions | Safe no-op rate | Finance action rate | Avg game length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| randomDiagnostic | 400 | 2.3% | 3.76 | 8.95 | -18.20 | 0 | 4595 | 24.8% | 18.8% | 61.3 |
| easy | 400 | 1.0% | 3.61 | 14.63 | -12.52 | 0 | 5381 | 12.1% | 22.3% | 61.2 |
| medium | 400 | 25.0% | 2.46 | 27.15 | 0.00 | 0 | 5593 | 7.5% | 24.9% | 60.5 |
| hard | 400 | 61.3% | 1.60 | 35.66 | 8.51 | 0 | 3617 | 41.2% | 0.2% | 61.5 |
| expert | 400 | 39.8% | 2.05 | 29.52 | 2.37 | 0 | 5432 | 8.1% | 24.8% | 59.1 |

## Interpretation

- Medium is the naive baseline: rule-based, legal-action-aware, and reproducible.
- Random diagnostic remains in the report only to catch benchmark or game-economy problems.
- ML does not invent moves. The game engine generates legal actions first, then expert ranks those candidates.
- Hard is accepted only if it beats medium on held-out seat-rotated validation.
- Expert did not beat hard by average rank in this benchmark, so the tuned heuristic is the stronger current policy.

## Current Gate Snapshot

- Hard score diff vs medium: 8.51
- Expert score diff vs medium: 2.37
- Random diagnostic outperformed medium: no
- Top expert feature: actionIncome
