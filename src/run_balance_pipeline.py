"""
run_balance_pipeline.py — unified board + game balance export runner
====================================================================
Writes board search artifacts, board-aware game telemetry, and a
unified markdown/json summary under optimizer_outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

from balance_optimizer import BalanceConfig, evaluate_current_balance, export_recommendations
from board_autobalancer import BoardSearchConfig, export_board_recommendations, optimize_board
from card_parser import parse_all_cards
from game_engine_v3 import DEFAULT_TOURNAMENT_STRATEGIES, GameConfig, GameEngine, evaluate_strategy_tournament
from ml_balance_model import export_card_strength_surrogate, fit_card_strength_surrogate


OUTPUT_DIR = Path(__file__).resolve().parent / "optimizer_outputs"
BOARD_DIR = OUTPUT_DIR / "board"
GAME_DIR = OUTPUT_DIR / "game"
UNIFIED_DIR = OUTPUT_DIR / "unified"


def _to_serializable(value):
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    return value


def _records(frame, limit: int):
    return json.loads(frame.head(limit).to_json(orient="records", force_ascii=False))


def _comparison_df(before_df, after_df, key_columns: list[str], value_columns: list[str]):
    before = before_df.copy()
    after = after_df.copy()
    for column in value_columns:
        if column not in before.columns:
            before[column] = 0.0
        if column not in after.columns:
            after[column] = 0.0
    merged = before.merge(
        after,
        on=key_columns,
        how="outer",
        suffixes=("_before", "_after"),
    ).fillna(0)
    for column in value_columns:
        merged[f"{column}_delta"] = merged[f"{column}_after"] - merged[f"{column}_before"]
    return merged


def run_pipeline(
    board_search_config: BoardSearchConfig | None = None,
    tournament_simulations: int = 12,
    card_baseline_simulations: int = 24,
    strategies: list[str] | None = None,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GAME_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    board_search_config = board_search_config or BoardSearchConfig(
        baseline_boards=24,
        search_boards=10,
        final_verification_boards=48,
        max_accepted_changes=2,
        minimum_improvement=0.002,
        allow_global_fee_changes=False,
    )
    strategies = strategies or list(DEFAULT_TOURNAMENT_STRATEGIES[:6])

    board_result = optimize_board(
        config=board_search_config
    )
    export_board_recommendations(board_result, BOARD_DIR)
    board_graph_report = json.loads((BOARD_DIR / "board_graph_report.json").read_text(encoding="utf-8"))

    cards_df = parse_all_cards()

    card_result = evaluate_current_balance(
        cards_df,
        BalanceConfig(
            baseline_simulations=card_baseline_simulations,
            candidate_pool_size=6,
        ),
    )
    export_recommendations(card_result, GAME_DIR / "card_recommendations")

    board_engine = GameEngine(cards_df, GameConfig(board_enabled=True))
    tournament = evaluate_strategy_tournament(
        engine=board_engine,
        strategies=strategies,
        n_simulations=tournament_simulations,
        base_seed=20260409,
        include_card_usage=True,
        include_matchup_results=True,
        verbose=False,
    )

    tournament["matchups"].to_csv(GAME_DIR / "board_aware_matchups.csv", index=False)
    tournament["strategy_stats"].to_csv(GAME_DIR / "board_aware_strategy_stats.csv", index=False)
    tournament["card_usage"].to_csv(GAME_DIR / "board_aware_card_usage.csv", index=False)
    tournament["board_telemetry"]["win_rate_by_starting_zone"].to_csv(
        GAME_DIR / "win_rate_by_starting_zone.csv", index=False
    )
    tournament["board_telemetry"]["slot_usage"].to_csv(GAME_DIR / "slot_usage.csv", index=False)
    tournament["board_telemetry"]["frame_usage"].to_csv(GAME_DIR / "frame_usage.csv", index=False)
    tournament["board_telemetry"]["industry_zone_matrix"].to_csv(
        GAME_DIR / "industry_zone_matrix.csv", index=False
    )
    tournament["board_telemetry"]["summary"].to_csv(GAME_DIR / "board_telemetry_summary.csv", index=False)
    surrogate_report = fit_card_strength_surrogate(cards_df, tournament)
    export_card_strength_surrogate(surrogate_report, GAME_DIR)

    game_report = {
        "metric_direction": {
            "balance_score": "lower_is_better",
            "max_abs_deviation": "lower_is_better",
        },
        "balance_score": tournament["balance_score"],
        "max_abs_deviation": tournament["max_abs_deviation"],
        "strategies": tournament["strategies"],
        "strategy_stats": _to_serializable(tournament["strategy_stats"]),
        "board_telemetry": {
            key: _to_serializable(value)
            for key, value in tournament["board_telemetry"].items()
        },
        "card_strength_surrogate": {
            "model_type": surrogate_report["model_type"],
            "target_column": surrogate_report["target_column"],
            "baseline_metrics": surrogate_report["baseline_metrics"],
            "oof_metrics": surrogate_report["oof_metrics"],
            "top_positive_residuals": _records(surrogate_report["predictions"], 5),
        },
    }
    (GAME_DIR / "board_aware_game_report.json").write_text(
        json.dumps(game_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    candidate_before_zone = board_result.baseline_dynamic_metrics["board_telemetry"]["win_rate_by_starting_zone"].copy()
    candidate_after_zone = board_result.final_dynamic_metrics["board_telemetry"]["win_rate_by_starting_zone"].copy()
    candidate_before_frame = board_result.baseline_dynamic_metrics["board_telemetry"]["frame_usage"].copy()
    candidate_after_frame = board_result.final_dynamic_metrics["board_telemetry"]["frame_usage"].copy()
    candidate_before_slot = board_result.baseline_dynamic_metrics["board_telemetry"]["slot_usage"].copy()
    candidate_after_slot = board_result.final_dynamic_metrics["board_telemetry"]["slot_usage"].copy()

    zone_comparison = _comparison_df(candidate_before_zone, candidate_after_zone, ["starting_zone"], ["games", "wins", "win_rate"])
    frame_comparison = _comparison_df(candidate_before_frame, candidate_after_frame, ["frame_name"], ["times_used"])
    slot_comparison = _comparison_df(candidate_before_slot, candidate_after_slot, ["slot_id"], ["times_used"])

    zone_comparison.to_csv(GAME_DIR / "candidate_zone_comparison.csv", index=False)
    frame_comparison.to_csv(GAME_DIR / "candidate_frame_comparison.csv", index=False)
    slot_comparison.to_csv(GAME_DIR / "candidate_slot_comparison.csv", index=False)

    candidate_verification = {
        "accepted_changes": [change.description for change in board_result.accepted_changes],
        "baseline_dynamic": {
            key: value
            for key, value in board_result.baseline_dynamic_metrics.items()
            if key != "board_telemetry"
        },
        "final_dynamic": {
            key: value
            for key, value in board_result.final_dynamic_metrics.items()
            if key != "board_telemetry"
        },
        "zone_gap_worsened": board_result.final_dynamic_metrics["zone_gap"] > board_result.baseline_dynamic_metrics["zone_gap"],
        "weakest_zone_before": board_result.baseline_dynamic_metrics["weakest_zone"],
        "weakest_zone_after": board_result.final_dynamic_metrics["weakest_zone"],
    }
    (GAME_DIR / "candidate_verification_report.json").write_text(
        json.dumps(candidate_verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    unified_summary = {
        "board": {
            "baseline": board_result.baseline_metrics["grand_mean"],
            "final_verification": board_result.final_metrics["grand_mean"],
            "graph_metrics": {
                "baseline": board_result.baseline_metrics.get("graph_grand_mean", {}),
                "final_verification": board_result.final_metrics.get("graph_grand_mean", {}),
                "representative_report": board_graph_report,
            },
            "accepted_changes": [change.description for change in board_result.accepted_changes],
            "candidate_dynamic_before": {
                "zone_gap": board_result.baseline_dynamic_metrics["zone_gap"],
                "weakest_zone": board_result.baseline_dynamic_metrics["weakest_zone"],
                "weakest_zone_win_rate": board_result.baseline_dynamic_metrics["weakest_zone_win_rate"],
            },
            "candidate_dynamic_after": {
                "zone_gap": board_result.final_dynamic_metrics["zone_gap"],
                "weakest_zone": board_result.final_dynamic_metrics["weakest_zone"],
                "weakest_zone_win_rate": board_result.final_dynamic_metrics["weakest_zone_win_rate"],
            },
        },
        "card_recommendations": {
            "baseline_balance_score": card_result.baseline_metrics["balance_score"],
            "final_balance_score": card_result.final_metrics["balance_score"],
            "accepted_changes": [
                {
                    "card_id": change.card_id,
                    "field": change.field,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                }
                for change in card_result.accepted_changes
            ],
        },
        "board_aware_game": {
            "balance_score": tournament["balance_score"],
            "max_abs_deviation": tournament["max_abs_deviation"],
            "top_starting_zones": _to_serializable(
                tournament["board_telemetry"]["win_rate_by_starting_zone"].head(5)
            ),
        },
        "ml_surrogate": {
            "model_type": surrogate_report["model_type"],
            "target_column": surrogate_report["target_column"],
            "baseline_metrics": surrogate_report["baseline_metrics"],
            "oof_metrics": surrogate_report["oof_metrics"],
            "top_positive_residuals": _records(surrogate_report["predictions"], 5),
        },
        "graph_metrics": {
            "board_graph_report": board_graph_report,
            "final_board_graph_summary": board_result.final_metrics.get("graph_grand_mean", {}),
        },
    }
    (UNIFIED_DIR / "balance_summary.json").write_text(
        json.dumps(unified_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_md = f"""# Balance Pipeline Summary

## Board

- Baseline `overall_balance`: {board_result.baseline_metrics['grand_mean']['mean_overall_balance']:.4f}
- Final verified `overall_balance`: {board_result.final_metrics['grand_mean']['mean_overall_balance']:.4f}
- Final `centro_vs_barrio`: {board_result.final_metrics['grand_mean']['mean_centro_vs_barrio']:.4f}
- Accepted board changes: {', '.join(change.description for change in board_result.accepted_changes) or 'none'}
- Candidate dynamic zone gap: {board_result.baseline_dynamic_metrics['zone_gap']:.4f} -> {board_result.final_dynamic_metrics['zone_gap']:.4f}
- Candidate weakest zone: {board_result.baseline_dynamic_metrics['weakest_zone']} -> {board_result.final_dynamic_metrics['weakest_zone']}
- Candidate weakest-zone win rate: {board_result.baseline_dynamic_metrics['weakest_zone_win_rate']:.4f} -> {board_result.final_dynamic_metrics['weakest_zone_win_rate']:.4f}
- Graph frame betweenness gap: {board_result.baseline_metrics.get('graph_grand_mean', {}).get('frame_betweenness_gap', 0.0):.4f} -> {board_result.final_metrics.get('graph_grand_mean', {}).get('frame_betweenness_gap', 0.0):.4f}
- Graph closeness spread: {board_result.baseline_metrics.get('graph_grand_mean', {}).get('frame_closeness_std', 0.0):.4f} -> {board_result.final_metrics.get('graph_grand_mean', {}).get('frame_closeness_std', 0.0):.4f}
- Representative graph artifacts: `{BOARD_DIR / 'board_slot_graph_metrics.csv'}`, `{BOARD_DIR / 'board_frame_graph_metrics.csv'}`, `{BOARD_DIR / 'board_graph_report.json'}`

## Card Recommendations

- Baseline `balance_score` (lower is better): {card_result.baseline_metrics['balance_score']:.4f}
- Final `balance_score`: {card_result.final_metrics['balance_score']:.4f}
- Accepted recommendation count: {len(card_result.accepted_changes)}

## Board-Aware Tournament

- `balance_score`: {tournament['balance_score']:.4f}
- `max_abs_deviation`: {tournament['max_abs_deviation']:.4f}
- Highest observed starting-zone win rates are exported to `{GAME_DIR / 'win_rate_by_starting_zone.csv'}`
- Candidate before/after frame and zone comparisons are exported to `{GAME_DIR / 'candidate_zone_comparison.csv'}` and `{GAME_DIR / 'candidate_frame_comparison.csv'}`

## ML Surrogate

- Target label: `{surrogate_report['target_column']}`
- Mean-baseline MAE: {surrogate_report['baseline_metrics']['mae']:.4f}
- Out-of-fold MAE: {surrogate_report['oof_metrics']['mae']:.4f}
- Out-of-fold R²: {surrogate_report['oof_metrics']['r2']:.4f}
- Top residual outliers are exported to `{GAME_DIR / 'card_strength_surrogate.csv'}`
"""
    (UNIFIED_DIR / "balance_summary.md").write_text(summary_md, encoding="utf-8")

    return unified_summary


def main() -> None:
    summary = run_pipeline()
    print("Unified balance pipeline complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
