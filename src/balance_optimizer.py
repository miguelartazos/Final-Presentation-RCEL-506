"""
balance_optimizer.py — Monte Carlo auto-balancer for Business Empire
====================================================================
Uses simulator metrics and searches for small,
design-constrained stat edits that improve overall balance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from card_parser import compute_derived_features, parse_all_cards
from game_engine_v3 import (
    DEFAULT_TOURNAMENT_STRATEGIES,
    GameEngine,
    _stable_seed,
    evaluate_strategy_tournament,
)
from ml_balance_model import export_card_strength_surrogate, fit_card_strength_surrogate


OUTPUT_DIR = Path(__file__).resolve().parent / "optimizer_outputs"


@dataclass
class BalanceConfig:
    """Configuration for the Monte Carlo balance search."""

    strategies: tuple[str, ...] = tuple(DEFAULT_TOURNAMENT_STRATEGIES)
    baseline_simulations: int = 300
    search_simulations: int = 120
    final_verification_simulations: int = 1000
    editable_fields: tuple[str, ...] = ("cost", "income", "valuation_points", "exit_value", "income_scaled")
    step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "cost": 1.0,
            "income": 1.0,
            "valuation_points": 1.0,
            "exit_value": 1.0,
            "income_scaled": 1.0,
        }
    )
    candidate_pool_size: int = 12
    max_accepted_edits: int = 5
    stop_balance_score: float = 0.10
    minimum_improvement: float = 0.005
    base_seed: int = 20260316
    baseline_seed: int | None = None
    search_seed: int | None = None
    final_verification_seed: int | None = None
    max_relative_change: float = 0.20
    max_vp_change: float = 2.0
    max_fields_per_card: int = 2
    zero_value_absolute_cap: float = 1.0


@dataclass
class CandidateChange:
    """A single local stat change evaluated by the optimizer."""

    card_id: str
    card_name: str
    field: str
    old_value: float
    new_value: float
    delta: float
    risk_score: float
    rationale: str
    step: int = 0
    before_balance_score: float | None = None
    after_balance_score: float | None = None
    before_max_abs_deviation: float | None = None
    after_max_abs_deviation: float | None = None
    estimated_after_balance_score: float | None = None
    estimated_after_max_abs_deviation: float | None = None
    estimated_improvement: float | None = None
    confirmed_after_balance_score: float | None = None
    confirmed_after_max_abs_deviation: float | None = None
    confirmed_improvement: float | None = None
    confirmation_passed: bool = False
    search_evaluation_seed: int | None = None
    confirmation_seed: int | None = None
    evaluation_seed: int | None = None


@dataclass
class OptimizationResult:
    """Structured optimizer output for notebooks and exports."""

    config: BalanceConfig
    original_cards_df: pd.DataFrame
    optimized_cards_df: pd.DataFrame
    baseline_metrics: dict[str, Any]
    optimized_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    risky_cards: pd.DataFrame
    candidate_changes: list[CandidateChange]
    accepted_changes: list[CandidateChange]
    search_trace: pd.DataFrame
    run_summary: dict[str, Any]


def _prepare_cards_df(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric fields and recompute derived features."""
    df = cards_df.copy(deep=True)
    defaults = {
        "tags": [[] for _ in range(len(df))],
        "requirements": [[] for _ in range(len(df))],
        "synergies": [[] for _ in range(len(df))],
        "upkeep": 0,
        "time_delay": 0,
        "effort": 0,
        "likelihood": 0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    numeric_cols = [
        "cost",
        "income",
        "valuation_points",
        "upkeep",
        "time_delay",
        "effort",
        "likelihood",
        "staff_min",
        "staff_opt",
        "income_scaled",
        "income_opt",
        "exit_value",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return compute_derived_features(df)


def _resolve_comparison_seed(config: BalanceConfig, phase: str) -> int:
    """Resolve an explicit comparison seed for a given evaluation phase."""
    phase_seeds = {
        "baseline": config.baseline_seed,
        "search": config.search_seed,
        "final_verification": config.final_verification_seed,
    }
    configured = phase_seeds.get(phase)
    if configured is not None:
        return configured
    return _stable_seed(config.base_seed, phase)


def _metrics_to_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert tournament metrics into JSON-friendly summary data."""
    return {
        "balance_score": metrics["balance_score"],
        "max_abs_deviation": metrics["max_abs_deviation"],
        "n_simulations": metrics["n_simulations"],
        "base_seed": metrics["base_seed"],
        "comparison_seed": metrics.get("comparison_seed"),
        "strategies": metrics["strategies"],
        "strategy_stats": metrics["strategy_stats"].to_dict(orient="records"),
        "top_cards": metrics["card_usage"].head(15).to_dict(orient="records"),
    }


def _evaluate_cards(
    cards_df: pd.DataFrame,
    config: BalanceConfig,
    n_simulations: int,
    comparison_seed: int,
    include_results: bool = True,
    include_card_usage: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a seeded tournament evaluation for the current card set."""
    prepared = _prepare_cards_df(cards_df)
    engine = GameEngine(prepared)
    metrics = evaluate_strategy_tournament(
        engine=engine,
        strategies=list(config.strategies),
        n_simulations=n_simulations,
        base_seed=comparison_seed,
        include_card_usage=include_card_usage,
        include_matchup_results=include_results,
        verbose=verbose,
    )
    metrics["comparison_seed"] = comparison_seed
    return metrics


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series."""
    filled = series.fillna(0.0).astype(float)
    minimum = float(filled.min())
    maximum = float(filled.max())
    if maximum == minimum:
        return pd.Series([0.0] * len(filled), index=filled.index)
    return (filled - minimum) / (maximum - minimum)


def _build_risky_cards(cards_df: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    """Score business cards by combined outlier strength and tournament dominance."""
    prepared = _prepare_cards_df(cards_df)
    business = prepared[prepared["type"].astype(str).str.strip().str.title() == "Business"].copy()

    usage = metrics["card_usage"][
        ["card_id", "usage_rate", "win_bias", "win_deck_rate", "loss_deck_rate"]
    ].rename(columns={"card_id": "id"})
    merged = business.merge(usage, on="id", how="left")
    for column in ["usage_rate", "win_bias", "win_deck_rate", "loss_deck_rate"]:
        merged[column] = merged[column].fillna(0.0)

    surrogate_columns = [
        "id",
        "predicted_win_bias",
        "win_bias_residual",
        "positive_win_bias_residual",
        "positive_residual_norm",
        "negative_residual_norm",
        "outlier_direction",
    ]
    try:
        surrogate_report = fit_card_strength_surrogate(prepared, metrics)
        surrogate = surrogate_report["predictions"][surrogate_columns].copy()
    except Exception:
        surrogate = pd.DataFrame({"id": merged["id"]})
        surrogate["predicted_win_bias"] = 0.0
        surrogate["win_bias_residual"] = 0.0
        surrogate["positive_win_bias_residual"] = 0.0
        surrogate["positive_residual_norm"] = 0.0
        surrogate["negative_residual_norm"] = 0.0
        surrogate["outlier_direction"] = "unavailable"

    merged = merged.merge(surrogate, on="id", how="left")
    for column in [
        "predicted_win_bias",
        "win_bias_residual",
        "positive_win_bias_residual",
        "positive_residual_norm",
        "negative_residual_norm",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["outlier_direction"] = merged["outlier_direction"].fillna("unavailable")

    merged["positive_win_bias"] = merged["win_bias"].clip(lower=0.0)
    # v2 risk metrics (Hormozi retired from scoring)
    merged["effective_roi_norm"] = _normalize(merged["effective_roi"] if "effective_roi" in merged.columns else pd.Series(0.0, index=merged.index))
    merged["hormozi_v2_norm"] = _normalize(merged["hormozi_v2"] if "hormozi_v2" in merged.columns else pd.Series(0.0, index=merged.index))
    merged["usage_norm"] = _normalize(merged["usage_rate"])
    merged["win_bias_norm"] = _normalize(merged["positive_win_bias"])
    payback = merged["payback_breaks"] if "payback_breaks" in merged.columns else pd.Series(1.0, index=merged.index)
    merged["payback_inv_norm"] = _normalize(1.0 / payback.clip(lower=0.1))

    merged["risk_score"] = (
        0.25 * merged["effective_roi_norm"]
        + 0.20 * merged["hormozi_v2_norm"]
        + 0.15 * merged["usage_norm"]
        + 0.15 * merged["win_bias_norm"]
        + 0.10 * merged["payback_inv_norm"]
        + 0.15 * merged["positive_residual_norm"]
    ).round(4)

    def build_rationale(row: pd.Series) -> str:
        components = {
            "high effective ROI": row["effective_roi_norm"],
            "high VP per launch cost": row["hormozi_v2_norm"],
            "high overall usage": row["usage_norm"],
            "high usage in winning decks": row["win_bias_norm"],
            "fast payback": row["payback_inv_norm"],
            "beats ML expectation": row["positive_residual_norm"],
        }
        ordered = sorted(components.items(), key=lambda item: item[1], reverse=True)
        reasons = [label for label, value in ordered if value >= 0.65][:2]
        if not reasons and ordered:
            reasons = [ordered[0][0]]
        return " + ".join(reasons)

    merged["rationale"] = merged.apply(build_rationale, axis=1)
    columns = [
        "id",
        "name",
        "industry",
        "tier",
        "cost",
        "income",
        "valuation_points",
        "effective_roi",
        "hormozi_v2",
        "usage_rate",
        "win_deck_rate",
        "loss_deck_rate",
        "win_bias",
        "predicted_win_bias",
        "win_bias_residual",
        "positive_win_bias_residual",
        "positive_residual_norm",
        "negative_residual_norm",
        "outlier_direction",
        "risk_score",
        "rationale",
        "effective_roi_norm",
        "hormozi_v2_norm",
        "usage_norm",
        "win_bias_norm",
        "payback_inv_norm",
    ]
    # Only include columns that exist
    columns = [c for c in columns if c in merged.columns]
    return merged[columns].sort_values(
        ["risk_score", "usage_rate", "win_bias"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _coerce_result(
    cards_df: pd.DataFrame,
    metrics: OptimizationResult | dict[str, Any],
    config: BalanceConfig,
) -> OptimizationResult:
    """Convert raw tournament metrics into an OptimizationResult shell."""
    prepared = _prepare_cards_df(cards_df)

    if isinstance(metrics, OptimizationResult):
        return metrics

    risky_cards = _build_risky_cards(prepared, metrics)
    return OptimizationResult(
        config=config,
        original_cards_df=prepared.copy(deep=True),
        optimized_cards_df=prepared.copy(deep=True),
        baseline_metrics=metrics,
        optimized_metrics=metrics,
        final_metrics=metrics,
        risky_cards=risky_cards,
        candidate_changes=[],
        accepted_changes=[],
        search_trace=pd.DataFrame(),
        run_summary={},
    )


def _current_edit_state(result: OptimizationResult) -> dict[str, dict[str, Any]]:
    """Track cumulative edits per card across accepted changes."""
    state: dict[str, dict[str, Any]] = {}
    for change in result.accepted_changes:
        entry = state.setdefault(
            change.card_id,
            {"fields_touched": set(), "cumulative": {}},
        )
        entry["fields_touched"].add(change.field)
        entry["cumulative"][change.field] = entry["cumulative"].get(change.field, 0.0) + change.delta
    return state


def _within_guardrails(
    proposed_value: float,
    original_value: float,
    field: str,
    fields_touched: set[str],
    config: BalanceConfig,
) -> bool:
    """Check whether a proposed local edit respects the design constraints."""
    if proposed_value < 0:
        return False

    if field not in fields_touched and len(fields_touched) >= config.max_fields_per_card:
        return False

    cumulative_change = abs(proposed_value - original_value)
    if field in ("valuation_points", "exit_value"):
        return cumulative_change <= config.max_vp_change

    if original_value > 0:
        limit = original_value * config.max_relative_change
    else:
        limit = config.zero_value_absolute_cap
    return cumulative_change <= limit


def _apply_candidate_change(cards_df: pd.DataFrame, change: CandidateChange) -> pd.DataFrame:
    """Return a fresh card table with one accepted candidate applied."""
    updated = cards_df.copy(deep=True)
    mask = updated["id"] == change.card_id
    if mask.sum() != 1:
        raise ValueError(f"Could not uniquely locate card {change.card_id}")
    updated.loc[mask, change.field] = change.new_value
    return _prepare_cards_df(updated)


def _change_to_recommendation_row(change: CandidateChange) -> dict[str, Any]:
    """Serialize one accepted change for CSV/JSON export."""
    return {
        "step": change.step,
        "card_id": change.card_id,
        "card_name": change.card_name,
        "field": change.field,
        "old_value": change.old_value,
        "new_value": change.new_value,
        "delta": change.delta,
        "risk_score": round(change.risk_score, 4),
        "before_balance_score": change.before_balance_score,
        "estimated_after_balance_score": change.estimated_after_balance_score,
        "confirmed_after_balance_score": change.confirmed_after_balance_score,
        "after_balance_score": change.after_balance_score,
        "before_max_abs_deviation": change.before_max_abs_deviation,
        "estimated_after_max_abs_deviation": change.estimated_after_max_abs_deviation,
        "confirmed_after_max_abs_deviation": change.confirmed_after_max_abs_deviation,
        "after_max_abs_deviation": change.after_max_abs_deviation,
        "estimated_improvement": change.estimated_improvement,
        "confirmed_improvement": change.confirmed_improvement,
        "confirmation_passed": change.confirmation_passed,
        "search_evaluation_seed": change.search_evaluation_seed,
        "confirmation_seed": change.confirmation_seed,
        "rationale": change.rationale,
    }


def _empty_recommendations_df() -> pd.DataFrame:
    """Return an empty recommendations frame with the expected columns."""
    return pd.DataFrame(
        columns=[
            "step",
            "card_id",
            "card_name",
            "field",
            "old_value",
            "new_value",
            "delta",
            "risk_score",
            "before_balance_score",
            "estimated_after_balance_score",
            "confirmed_after_balance_score",
            "after_balance_score",
            "before_max_abs_deviation",
            "estimated_after_max_abs_deviation",
            "confirmed_after_max_abs_deviation",
            "after_max_abs_deviation",
            "estimated_improvement",
            "confirmed_improvement",
            "confirmation_passed",
            "search_evaluation_seed",
            "confirmation_seed",
            "rationale",
        ]
    )


def evaluate_current_balance(
    cards_df: pd.DataFrame,
    config: BalanceConfig | None = None,
) -> OptimizationResult:
    """Evaluate the current card set and build the initial risky-card ranking."""
    config = config or BalanceConfig()
    prepared = _prepare_cards_df(cards_df)
    baseline_metrics = _evaluate_cards(
        prepared,
        config,
        n_simulations=config.baseline_simulations,
        comparison_seed=_resolve_comparison_seed(config, "baseline"),
        include_results=True,
        include_card_usage=True,
    )
    risky_cards = _build_risky_cards(prepared, baseline_metrics)
    base_result = OptimizationResult(
        config=config,
        original_cards_df=prepared.copy(deep=True),
        optimized_cards_df=prepared.copy(deep=True),
        baseline_metrics=baseline_metrics,
        optimized_metrics=baseline_metrics,
        final_metrics=baseline_metrics,
        risky_cards=risky_cards,
        candidate_changes=[],
        accepted_changes=[],
        search_trace=pd.DataFrame(),
        run_summary={
            "candidate_evaluations": 0,
            "confirmation_attempts": 0,
            "accepted_edits": 0,
            "no_confirmed_changes": True,
        },
    )
    base_result.candidate_changes = generate_candidate_changes(prepared, base_result, config)
    return base_result


def generate_candidate_changes(
    cards_df: pd.DataFrame,
    metrics: OptimizationResult | dict[str, Any],
    config: BalanceConfig | None = None,
) -> list[CandidateChange]:
    """Generate local stat edits for the top risky business cards."""
    config = config or BalanceConfig()
    result = _coerce_result(cards_df, metrics, config)
    current_cards = _prepare_cards_df(cards_df)
    original_business = result.original_cards_df[
        result.original_cards_df["type"].astype(str).str.strip().str.title() == "Business"
    ].set_index("id")
    current_business = current_cards[
        current_cards["type"].astype(str).str.strip().str.title() == "Business"
    ].set_index("id")
    edit_state = _current_edit_state(result)

    candidates: list[CandidateChange] = []
    risky_pool = result.risky_cards.head(config.candidate_pool_size)

    for _, row in risky_pool.iterrows():
        card_id = row["id"]
        fields_touched = set(edit_state.get(card_id, {}).get("fields_touched", set()))
        current_row = current_business.loc[card_id]
        original_row = original_business.loc[card_id]

        for field in config.editable_fields:
            step = float(config.step_sizes[field])
            current_value = float(current_row[field])
            original_value = float(original_row[field])
            for direction in (-1.0, 1.0):
                new_value = round(current_value + (step * direction), 4)
                if not _within_guardrails(
                    proposed_value=new_value,
                    original_value=original_value,
                    field=field,
                    fields_touched=fields_touched,
                    config=config,
                ):
                    continue

                candidates.append(
                    CandidateChange(
                        card_id=card_id,
                        card_name=str(row["name"]),
                        field=field,
                        old_value=current_value,
                        new_value=new_value,
                        delta=round(new_value - current_value, 4),
                        risk_score=float(row["risk_score"]),
                        rationale=str(row["rationale"]),
                    )
                )

    candidates.sort(
        key=lambda change: (
            -change.risk_score,
            change.card_id,
            change.field,
            change.delta,
        )
    )
    return candidates


def optimize_balance(
    cards_df: pd.DataFrame,
    config: BalanceConfig | None = None,
) -> OptimizationResult:
    """Run the greedy Monte Carlo search and return the accepted edits."""
    config = config or BalanceConfig()
    baseline = evaluate_current_balance(cards_df, config)
    current_cards = baseline.optimized_cards_df.copy(deep=True)
    current_metrics = baseline.baseline_metrics
    current_result = baseline

    search_seed = _resolve_comparison_seed(config, "search")
    baseline_seed = _resolve_comparison_seed(config, "baseline")
    final_seed = _resolve_comparison_seed(config, "final_verification")

    accepted_changes: list[CandidateChange] = []
    search_rows: list[dict[str, Any]] = []
    candidate_evaluations = 0
    confirmation_attempts = 0

    while len(accepted_changes) < config.max_accepted_edits:
        if current_metrics["balance_score"] <= config.stop_balance_score:
            break

        candidates = generate_candidate_changes(current_cards, current_result, config)
        if not candidates:
            break

        search_baseline_metrics = _evaluate_cards(
            current_cards,
            config,
            n_simulations=config.search_simulations,
            comparison_seed=search_seed,
            include_results=False,
            include_card_usage=False,
        )

        evaluations: list[CandidateChange] = []
        for candidate in candidates:
            candidate_evaluations += 1
            updated_cards = _apply_candidate_change(current_cards, candidate)
            estimated_metrics = _evaluate_cards(
                updated_cards,
                config,
                n_simulations=config.search_simulations,
                comparison_seed=search_seed,
                include_results=False,
                include_card_usage=False,
            )

            candidate.search_evaluation_seed = search_seed
            candidate.evaluation_seed = search_seed
            candidate.before_balance_score = current_metrics["balance_score"]
            candidate.before_max_abs_deviation = current_metrics["max_abs_deviation"]
            candidate.estimated_after_balance_score = estimated_metrics["balance_score"]
            candidate.estimated_after_max_abs_deviation = estimated_metrics["max_abs_deviation"]
            candidate.estimated_improvement = round(
                search_baseline_metrics["balance_score"] - estimated_metrics["balance_score"],
                4,
            )
            evaluations.append(candidate)
            search_rows.append(
                {
                    "step": len(accepted_changes) + 1,
                    "evaluation_stage": "search_estimate",
                    "card_id": candidate.card_id,
                    "card_name": candidate.card_name,
                    "field": candidate.field,
                    "old_value": candidate.old_value,
                    "new_value": candidate.new_value,
                    "delta": candidate.delta,
                    "risk_score": candidate.risk_score,
                    "rationale": candidate.rationale,
                    "comparison_seed": search_seed,
                    "before_balance_score": search_baseline_metrics["balance_score"],
                    "before_max_abs_deviation": search_baseline_metrics["max_abs_deviation"],
                    "estimated_after_balance_score": candidate.estimated_after_balance_score,
                    "estimated_after_max_abs_deviation": candidate.estimated_after_max_abs_deviation,
                    "estimated_improvement": candidate.estimated_improvement,
                    "confirmed_after_balance_score": None,
                    "confirmed_after_max_abs_deviation": None,
                    "confirmed_improvement": None,
                    "confirmation_passed": None,
                    "accepted": False,
                }
            )

        positive_estimates = [candidate for candidate in evaluations if candidate.estimated_improvement is not None and candidate.estimated_improvement > 0]
        if not positive_estimates:
            break

        positive_estimates.sort(
            key=lambda change: (
                change.estimated_after_balance_score,
                change.estimated_after_max_abs_deviation,
                abs(change.delta),
                change.card_id,
                change.field,
            )
        )

        accepted_this_round = False
        for candidate in positive_estimates:
            confirmation_attempts += 1
            confirmed_cards = _apply_candidate_change(current_cards, candidate)
            confirmed_metrics = _evaluate_cards(
                confirmed_cards,
                config,
                n_simulations=config.baseline_simulations,
                comparison_seed=baseline_seed,
                include_results=True,
                include_card_usage=True,
            )

            candidate.step = len(accepted_changes) + 1
            candidate.confirmation_seed = baseline_seed
            candidate.confirmed_after_balance_score = confirmed_metrics["balance_score"]
            candidate.confirmed_after_max_abs_deviation = confirmed_metrics["max_abs_deviation"]
            candidate.confirmed_improvement = round(
                current_metrics["balance_score"] - confirmed_metrics["balance_score"],
                4,
            )
            candidate.after_balance_score = candidate.confirmed_after_balance_score
            candidate.after_max_abs_deviation = candidate.confirmed_after_max_abs_deviation
            candidate.confirmation_passed = candidate.confirmed_improvement >= config.minimum_improvement

            search_rows.append(
                {
                    "step": candidate.step,
                    "evaluation_stage": "confirmation",
                    "card_id": candidate.card_id,
                    "card_name": candidate.card_name,
                    "field": candidate.field,
                    "old_value": candidate.old_value,
                    "new_value": candidate.new_value,
                    "delta": candidate.delta,
                    "risk_score": candidate.risk_score,
                    "rationale": candidate.rationale,
                    "comparison_seed": baseline_seed,
                    "before_balance_score": current_metrics["balance_score"],
                    "before_max_abs_deviation": current_metrics["max_abs_deviation"],
                    "estimated_after_balance_score": candidate.estimated_after_balance_score,
                    "estimated_after_max_abs_deviation": candidate.estimated_after_max_abs_deviation,
                    "estimated_improvement": candidate.estimated_improvement,
                    "confirmed_after_balance_score": candidate.confirmed_after_balance_score,
                    "confirmed_after_max_abs_deviation": candidate.confirmed_after_max_abs_deviation,
                    "confirmed_improvement": candidate.confirmed_improvement,
                    "confirmation_passed": candidate.confirmation_passed,
                    "accepted": candidate.confirmation_passed,
                }
            )

            if not candidate.confirmation_passed:
                continue

            accepted_changes.append(candidate)
            current_cards = confirmed_cards
            current_metrics = confirmed_metrics
            current_result = OptimizationResult(
                config=config,
                original_cards_df=baseline.original_cards_df.copy(deep=True),
                optimized_cards_df=current_cards.copy(deep=True),
                baseline_metrics=baseline.baseline_metrics,
                optimized_metrics=current_metrics,
                final_metrics=current_metrics,
                risky_cards=_build_risky_cards(current_cards, current_metrics),
                candidate_changes=[],
                accepted_changes=accepted_changes.copy(),
                search_trace=pd.DataFrame(search_rows),
                run_summary={},
            )
            accepted_this_round = True
            break

        if not accepted_this_round:
            break

    final_metrics = _evaluate_cards(
        current_cards,
        config,
        n_simulations=config.final_verification_simulations,
        comparison_seed=final_seed,
        include_results=True,
        include_card_usage=True,
    )

    run_summary = {
        "candidate_evaluations": candidate_evaluations,
        "confirmation_attempts": confirmation_attempts,
        "accepted_edits": len(accepted_changes),
        "no_confirmed_changes": len(accepted_changes) == 0,
        "search_seed": search_seed,
        "baseline_seed": baseline_seed,
        "final_verification_seed": final_seed,
    }

    final_result = OptimizationResult(
        config=config,
        original_cards_df=baseline.original_cards_df.copy(deep=True),
        optimized_cards_df=current_cards.copy(deep=True),
        baseline_metrics=baseline.baseline_metrics,
        optimized_metrics=current_result.optimized_metrics,
        final_metrics=final_metrics,
        risky_cards=baseline.risky_cards.copy(deep=True),
        candidate_changes=generate_candidate_changes(current_cards, current_result, config),
        accepted_changes=accepted_changes,
        search_trace=pd.DataFrame(search_rows),
        run_summary=run_summary,
    )
    return final_result


def export_recommendations(
    result: OptimizationResult,
    out_dir: str | Path,
) -> None:
    """Export optimizer artifacts for the class deliverable."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    recommendations = pd.DataFrame(
        [_change_to_recommendation_row(change) for change in result.accepted_changes]
    )
    if recommendations.empty:
        recommendations = _empty_recommendations_df()

    search_trace = result.search_trace.copy()
    if search_trace.empty:
        search_trace = pd.DataFrame(
            columns=[
                "step",
                "evaluation_stage",
                "card_id",
                "card_name",
                "field",
                "old_value",
                "new_value",
                "delta",
                "risk_score",
                "rationale",
                "comparison_seed",
                "before_balance_score",
                "before_max_abs_deviation",
                "estimated_after_balance_score",
                "estimated_after_max_abs_deviation",
                "estimated_improvement",
                "confirmed_after_balance_score",
                "confirmed_after_max_abs_deviation",
                "confirmed_improvement",
                "confirmation_passed",
                "accepted",
            ]
        )

    recommendations.to_csv(out_path / "balance_recommendations.csv", index=False)
    search_trace.to_csv(out_path / "search_trace.csv", index=False)

    payload = {
        "config": asdict(result.config),
        "run_summary": result.run_summary,
        "baseline": _metrics_to_snapshot(result.baseline_metrics),
        "optimized": _metrics_to_snapshot(result.optimized_metrics),
        "final_verification": _metrics_to_snapshot(result.final_metrics),
        "accepted_changes": recommendations.to_dict(orient="records"),
        "risky_cards": result.risky_cards.head(result.config.candidate_pool_size).to_dict(orient="records"),
    }
    if result.run_summary.get("no_confirmed_changes"):
        payload["message"] = "No candidate edit cleared confirmation at the configured threshold."

    (out_path / "balance_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    surrogate_report = fit_card_strength_surrogate(result.optimized_cards_df, result.final_metrics)
    export_card_strength_surrogate(surrogate_report, out_path)


def main() -> None:
    """Run the default optimizer workflow and export its artifacts."""
    print("Loading card data...")
    cards_df = parse_all_cards()

    print("\nRunning Monte Carlo auto-balancer...")
    result = optimize_balance(cards_df, BalanceConfig())
    export_recommendations(result, OUTPUT_DIR)

    print("\nAccepted changes:")
    if not result.accepted_changes:
        print("  No confirmed edits found.")
    else:
        for change in result.accepted_changes:
            print(
                f"  Step {change.step}: {change.card_id} {change.field} "
                f"{change.old_value:.1f} -> {change.new_value:.1f} "
                f"(estimate {change.estimated_after_balance_score:.4f}, "
                f"confirmed {change.confirmed_after_balance_score:.4f})"
            )

    print("\nBalance summary:")
    print(
        f"  Baseline balance score: {result.baseline_metrics['balance_score']:.4f}\n"
        f"  Confirmed optimized:    {result.optimized_metrics['balance_score']:.4f}\n"
        f"  Final verification:     {result.final_metrics['balance_score']:.4f}\n"
        f"  Baseline max deviation: {result.baseline_metrics['max_abs_deviation']:.4f}\n"
        f"  Final max deviation:    {result.final_metrics['max_abs_deviation']:.4f}"
    )
    print(
        f"\nCandidates evaluated: {result.run_summary['candidate_evaluations']}\n"
        f"Confirmation attempts: {result.run_summary['confirmation_attempts']}\n"
        f"Accepted edits:        {result.run_summary['accepted_edits']}"
    )
    print(f"\nArtifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
