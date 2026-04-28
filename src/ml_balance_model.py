"""
ml_balance_model.py — supervised surrogate for card balance diagnostics
======================================================================
Uses simulation outputs as labels and fits a small ridge-regression
surrogate over card attributes. The simulator remains the source of
target metrics; this model is a small explanatory model that highlights cards
performing better or worse than their stats profile suggests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from card_parser import compute_derived_features


OUTPUT_DIR = Path(__file__).resolve().parent / "optimizer_outputs" / "game"

NUMERIC_FEATURES = (
    "cost",
    "income",
    "valuation_points",
    "upkeep",
    "exit_value",
    "staff_min",
    "staff_opt",
    "income_scaled",
    "income_opt",
    "total_launch_cost",
    "avg_income",
    "max_income",
    "effective_roi",
    "payback_breaks",
    "synergy_output",
    "synergy_input",
    "synergy_links",
    "complexity_score",
    "tag_count",
    "requirement_count",
    "synergy_count",
    "time_delay",
    "effort",
    "likelihood",
)
DEFAULT_CATEGORICAL_FEATURES = ("industry", "tier", "mode", "tempo")


@dataclass(frozen=True)
class SurrogateConfig:
    """Configuration for the card-strength surrogate model."""

    target_column: str = "win_bias"
    alpha: float = 1.5
    n_folds: int = 5
    seed: int = 20260414
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - minimum) / (maximum - minimum)


def _records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return json.loads(frame.head(limit).to_json(orient="records", force_ascii=False))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    total_var = float(np.sum(np.square(y_true - y_true.mean())))
    residual_var = float(np.sum(np.square(errors)))
    r2 = 0.0 if total_var <= 1e-12 else float(1.0 - (residual_var / total_var))
    if len(y_true) <= 1 or np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "correlation": round(corr, 6),
    }


def _prepare_business_training_frame(cards_df: pd.DataFrame, metrics: dict[str, Any], target_column: str) -> pd.DataFrame:
    prepared = compute_derived_features(cards_df.copy(deep=True))
    business = prepared[prepared["type"].astype(str).str.strip().str.title() == "Business"].copy()

    usage = metrics["card_usage"][
        ["card_id", "usage_rate", "win_deck_rate", "loss_deck_rate", "win_bias"]
    ].rename(columns={"card_id": "id"})
    training = business.merge(usage, on="id", how="left")
    for column in ("usage_rate", "win_deck_rate", "loss_deck_rate", "win_bias"):
        training[column] = pd.to_numeric(training[column], errors="coerce").fillna(0.0)

    if target_column not in training.columns:
        raise KeyError(f"Target column '{target_column}' not available in merged training frame")

    return training.reset_index(drop=True)


def _build_design_matrix(training: pd.DataFrame, config: SurrogateConfig) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns = [column for column in NUMERIC_FEATURES if column in training.columns]
    numeric = (
        training[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    categorical_columns = [column for column in config.categorical_features if column in training.columns]
    categorical = pd.DataFrame(index=training.index)
    if categorical_columns:
        categorical = pd.get_dummies(
            training[categorical_columns].fillna("Unknown").astype(str),
            prefix=categorical_columns,
            dtype=float,
        )

    features = pd.concat([numeric, categorical], axis=1)
    if features.empty:
        raise ValueError("No surrogate-model features were available")

    non_constant = features.columns[features.nunique(dropna=False) > 1]
    features = features[non_constant].astype(float)
    return features, list(features.columns)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    feature_mean = X.mean(axis=0)
    feature_scale = X.std(axis=0, ddof=0)
    safe_scale = np.where(feature_scale <= 1e-12, 1.0, feature_scale)

    X_scaled = (X - feature_mean) / safe_scale
    y_mean = float(y.mean())
    y_centered = y - y_mean

    ridge = (X_scaled.T @ X_scaled) + (alpha * np.eye(X_scaled.shape[1], dtype=float))
    weights_scaled = np.linalg.solve(ridge, X_scaled.T @ y_centered)
    weights = weights_scaled / safe_scale
    intercept = y_mean - float(feature_mean @ weights)

    return {
        "intercept": intercept,
        "weights": weights,
        "weights_scaled": weights_scaled,
        "feature_mean": feature_mean,
        "feature_scale": safe_scale,
    }


def _predict(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    intercept = float(model["intercept"])
    weights = np.asarray(model["weights"], dtype=float)
    return intercept + (X @ weights)


def _make_folds(n_rows: int, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_rows < 2:
        raise ValueError("Need at least two rows to build surrogate-model folds")
    fold_count = max(2, min(n_folds, n_rows))
    indices = np.arange(n_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold.astype(int) for fold in np.array_split(indices, fold_count) if len(fold) > 0]


def fit_card_strength_surrogate(
    cards_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: SurrogateConfig | None = None,
) -> dict[str, Any]:
    """
    Fit a ridge-regression surrogate that predicts simulation-derived
    card performance from card attributes.
    """
    config = config or SurrogateConfig()
    training = _prepare_business_training_frame(cards_df, metrics, config.target_column)
    features, feature_columns = _build_design_matrix(training, config)

    X = features.to_numpy(dtype=float)
    y = training[config.target_column].to_numpy(dtype=float)

    folds = _make_folds(len(training), config.n_folds, config.seed)
    oof_predictions = np.zeros(len(training), dtype=float)
    fold_assignments = np.full(len(training), -1, dtype=int)

    for fold_idx, validation_idx in enumerate(folds):
        train_mask = np.ones(len(training), dtype=bool)
        train_mask[validation_idx] = False
        model = _fit_ridge(X[train_mask], y[train_mask], alpha=config.alpha)
        oof_predictions[validation_idx] = _predict(model, X[validation_idx])
        fold_assignments[validation_idx] = fold_idx

    full_model = _fit_ridge(X, y, alpha=config.alpha)
    full_predictions = _predict(full_model, X)
    mean_baseline = np.full(len(training), float(y.mean()), dtype=float)

    residual = y - oof_predictions
    positive_residual = np.clip(residual, a_min=0.0, a_max=None)
    negative_residual = np.clip(-residual, a_min=0.0, a_max=None)

    prediction_frame = training[
        [
            "id",
            "name",
            "industry",
            "tier",
            "cost",
            "income",
            "valuation_points",
            "effective_roi",
            "usage_rate",
            "win_deck_rate",
            "loss_deck_rate",
            "win_bias",
        ]
    ].copy()
    prediction_frame["predicted_win_bias"] = oof_predictions
    prediction_frame["win_bias_residual"] = residual
    prediction_frame["positive_win_bias_residual"] = positive_residual
    prediction_frame["negative_win_bias_residual"] = negative_residual
    prediction_frame["positive_residual_norm"] = _normalize(pd.Series(positive_residual)).to_numpy()
    prediction_frame["negative_residual_norm"] = _normalize(pd.Series(negative_residual)).to_numpy()
    prediction_frame["full_fit_prediction"] = full_predictions
    prediction_frame["fold"] = fold_assignments
    prediction_frame["outlier_direction"] = np.where(
        prediction_frame["win_bias_residual"] >= 0.015,
        "overperforming",
        np.where(prediction_frame["win_bias_residual"] <= -0.015, "underperforming", "in-line"),
    )
    prediction_frame = prediction_frame.sort_values(
        ["positive_win_bias_residual", "win_bias", "usage_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": np.asarray(full_model["weights"], dtype=float),
            "standardized_coefficient": np.asarray(full_model["weights_scaled"], dtype=float),
        }
    )
    coefficient_frame["abs_standardized_coefficient"] = coefficient_frame["standardized_coefficient"].abs()
    coefficient_frame = coefficient_frame.sort_values(
        ["abs_standardized_coefficient", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    report = {
        "config": asdict(config),
        "model_type": "ridge_regression",
        "target_column": config.target_column,
        "n_business_cards": int(len(training)),
        "baseline_metrics": _regression_metrics(y, mean_baseline),
        "oof_metrics": _regression_metrics(y, oof_predictions),
        "full_fit_metrics": _regression_metrics(y, full_predictions),
        "notes": [
            "Simulator-derived win_bias remains the ground truth.",
            "The surrogate is used for diagnosis and prioritization, not for accepting balance edits.",
            "Out-of-fold predictions are reported to avoid in-sample leakage on the small card set.",
        ],
        "predictions": prediction_frame,
        "coefficients": coefficient_frame,
    }
    return report


def export_card_strength_surrogate(report: dict[str, Any], out_dir: str | Path = OUTPUT_DIR) -> None:
    """Export surrogate predictions and metrics for downstream review."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    predictions = report["predictions"].copy()
    coefficients = report["coefficients"].copy()
    predictions.to_csv(out_path / "card_strength_surrogate.csv", index=False)
    coefficients.to_csv(out_path / "card_strength_coefficients.csv", index=False)

    payload = {
        key: value
        for key, value in report.items()
        if key not in {"predictions", "coefficients"}
    }
    payload["top_positive_residuals"] = _records(predictions, 10)
    payload["top_coefficients"] = _records(coefficients, 15)
    (out_path / "card_strength_model_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
