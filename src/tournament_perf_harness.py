"""
tournament_perf_harness.py — shared benchmark/gate helpers for evo
==================================================================
Defines the seeded tournament benchmark contract and the worktree
scope checks used by the tournament-throughput autoresearch pilot.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from card_parser import parse_all_cards
from game_engine_v3 import DEFAULT_TOURNAMENT_STRATEGIES, GameEngine, evaluate_strategy_tournament


DEFAULT_ALLOWED_EXPERIMENT_PATHS = (
    "07-DataScience/game_engine_v3.py",
)

FORBIDDEN_EXPERIMENT_PREFIXES = (
    "01-Vision/",
    "02-Mechanics/",
    "03-Industries/",
    "04-Cards/",
    "05-Board/",
    "06-Playtesting/",
    "07-DataScience/cards_dataset.csv",
    "07-DataScience/optimizer_outputs/",
    "07-DataScience/outputs/",
    "07-DataScience/slides/",
)

IGNORED_EXPERIMENT_PREFIXES = (
    ".evo/",
)


@dataclass(frozen=True)
class TournamentBenchmarkConfig:
    strategies: tuple[str, ...] = tuple(DEFAULT_TOURNAMENT_STRATEGIES[:6])
    n_simulations: int = 12
    base_seed: int = 20260401
    include_card_usage: bool = False
    include_matchup_results: bool = False
    verbose: bool = False
    warmup_runs: int = 1
    measured_runs: int = 5


def _matchup_digest(matchups_df: pd.DataFrame) -> str:
    ordered = matchups_df[
        ["strategy_a", "strategy_b", "win_rate_a", "win_rate_b", "draw_rate"]
    ].sort_values(["strategy_a", "strategy_b"]).reset_index(drop=True)
    payload = ordered.to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_seeded_tournament(config: TournamentBenchmarkConfig) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        cards_df = parse_all_cards()
    engine = GameEngine(cards_df)
    return evaluate_strategy_tournament(
        engine=engine,
        strategies=list(config.strategies),
        n_simulations=config.n_simulations,
        base_seed=config.base_seed,
        include_card_usage=config.include_card_usage,
        include_matchup_results=config.include_matchup_results,
        verbose=config.verbose,
    )


def collect_benchmark_contract(
    config: TournamentBenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or TournamentBenchmarkConfig()
    metrics = _run_seeded_tournament(config)
    return {
        "balance_score": metrics["balance_score"],
        "max_abs_deviation": metrics["max_abs_deviation"],
        "matchups_shape": list(metrics["matchups"].shape),
        "strategy_stats_shape": list(metrics["strategy_stats"].shape),
        "matchup_digest": _matchup_digest(metrics["matchups"]),
        "strategy_count": len(config.strategies),
        "n_simulations": config.n_simulations,
        "base_seed": config.base_seed,
    }


def run_benchmark(
    config: TournamentBenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or TournamentBenchmarkConfig()

    for _ in range(config.warmup_runs):
        _run_seeded_tournament(config)

    durations: list[float] = []
    contract: dict[str, Any] | None = None
    for _ in range(config.measured_runs):
        start = perf_counter()
        contract = collect_benchmark_contract(config)
        durations.append(perf_counter() - start)

    assert contract is not None
    mean_seconds = round(float(statistics.mean(durations)), 6)
    return {
        "score": mean_seconds,
        "mean_seconds": mean_seconds,
        "pstdev_seconds": round(float(statistics.pstdev(durations)), 6),
        "balance_score": contract["balance_score"],
        "max_abs_deviation": contract["max_abs_deviation"],
        "matchups_shape": contract["matchups_shape"],
        "strategy_stats_shape": contract["strategy_stats_shape"],
        "matchup_digest": contract["matchup_digest"],
        "strategy_count": contract["strategy_count"],
        "n_simulations": contract["n_simulations"],
        "base_seed": contract["base_seed"],
        "warmup_runs": config.warmup_runs,
        "measured_runs": config.measured_runs,
    }


def classify_changed_paths(
    changed_paths: list[str],
    allowed_paths: tuple[str, ...] = DEFAULT_ALLOWED_EXPERIMENT_PATHS,
) -> dict[str, list[str]]:
    normalized = sorted(
        {
            path.strip()
            for path in changed_paths
            if path.strip()
            and not any(path.strip().startswith(prefix) for prefix in IGNORED_EXPERIMENT_PREFIXES)
        }
    )
    allowed = [path for path in normalized if path in allowed_paths]
    forbidden = [
        path for path in normalized
        if any(path.startswith(prefix) for prefix in FORBIDDEN_EXPERIMENT_PREFIXES)
    ]
    unexpected = [
        path for path in normalized
        if path not in allowed_paths and path not in forbidden
    ]
    return {
        "allowed": allowed,
        "forbidden": forbidden,
        "unexpected": unexpected,
    }


def json_summary(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def benchmark_command(config: TournamentBenchmarkConfig | None = None) -> int:
    print(json_summary(run_benchmark(config)))
    return 0
