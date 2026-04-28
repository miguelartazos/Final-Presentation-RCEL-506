# Balance Pipeline Summary

## Board

- Baseline `overall_balance`: 0.5637
- Final verified `overall_balance`: 0.5688
- Final `centro_vs_barrio`: 2.3993
- Accepted board changes: Adjust spatial feature plaza_adjacent 0.35 -> 0.30
- Candidate dynamic zone gap: 0.3067 -> 0.1484
- Candidate weakest zone: Barrio Comercial -> Plaza Central
- Candidate weakest-zone win rate: 0.3333 -> 0.4516
- Graph frame betweenness gap: 0.3944 -> 0.3944
- Graph closeness spread: 0.1113 -> 0.1113
- Representative graph artifacts: `src/optimizer_outputs/board/board_slot_graph_metrics.csv`, `src/optimizer_outputs/board/board_frame_graph_metrics.csv`, `src/optimizer_outputs/board/board_graph_report.json`

## Card Recommendations

- Baseline `balance_score` (lower is better): 0.1726
- Final `balance_score`: 0.1726
- Accepted recommendation count: 0

## Board-Aware Tournament

- `balance_score`: 0.2035
- `max_abs_deviation`: 0.4167
- Highest observed starting-zone win rates are exported to `src/optimizer_outputs/game/win_rate_by_starting_zone.csv`
- Candidate before/after frame and zone comparisons are exported to `src/optimizer_outputs/game/candidate_zone_comparison.csv` and `src/optimizer_outputs/game/candidate_frame_comparison.csv`

## ML Surrogate

- Target label: `win_bias`
- Mean-baseline MAE: 0.0180
- Out-of-fold MAE: 0.0163
- Out-of-fold R²: 0.0905
- Top residual outliers are exported to `src/optimizer_outputs/game/card_strength_surrogate.csv`
