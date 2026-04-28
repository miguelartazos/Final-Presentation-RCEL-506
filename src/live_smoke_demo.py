"""
live_smoke_demo.py - fast tournament smoke demo
===============================================
Runs a deliberately tiny seeded benchmark. This command confirms the
simulator is live and parameterized without regenerating the full pipeline.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from game_engine_v3 import DEFAULT_TOURNAMENT_STRATEGIES
from tournament_perf_harness import TournamentBenchmarkConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast seeded Business Empire tournament smoke demo.")
    parser.add_argument("--n-simulations", type=int, default=1, help="Simulations per strategy matchup.")
    parser.add_argument("--strategy-count", type=int, default=4, help="Number of default strategies to include.")
    parser.add_argument("--base-seed", type=int, default=20260401, help="Seed for deterministic output.")
    return parser.parse_args()


def compact_summary(raw: dict[str, Any], strategies: tuple[str, ...]) -> dict[str, Any]:
    return {
        "purpose": "fast_tournament_smoke_demo",
        "config": {
            "strategies": list(strategies),
            "strategy_count": raw["strategy_count"],
            "n_simulations": raw["n_simulations"],
            "base_seed": raw["base_seed"],
            "warmup_runs": raw["warmup_runs"],
            "measured_runs": raw["measured_runs"],
        },
        "runtime": {
            "mean_seconds": raw["mean_seconds"],
            "pstdev_seconds": raw["pstdev_seconds"],
        },
        "metrics": {
            "balance_score": raw["balance_score"],
            "max_abs_deviation": raw["max_abs_deviation"],
            "matchups_shape": raw["matchups_shape"],
            "strategy_stats_shape": raw["strategy_stats_shape"],
        },
        "parameter_change_examples": [
            "increase --n-simulations to trade speed for stability",
            "increase --strategy-count to test a wider metagame",
            "change --base-seed to confirm reproducibility assumptions",
        ],
    }


def main() -> int:
    args = parse_args()
    strategy_count = max(2, min(args.strategy_count, len(DEFAULT_TOURNAMENT_STRATEGIES)))
    strategies = tuple(DEFAULT_TOURNAMENT_STRATEGIES[:strategy_count])
    n_simulations = max(1, args.n_simulations)
    config = TournamentBenchmarkConfig(
        strategies=strategies,
        n_simulations=n_simulations,
        base_seed=args.base_seed,
        include_card_usage=False,
        include_matchup_results=False,
        verbose=False,
        warmup_runs=0,
        measured_runs=1,
    )
    result = run_benchmark(config)
    print(json.dumps(compact_summary(result, strategies), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
