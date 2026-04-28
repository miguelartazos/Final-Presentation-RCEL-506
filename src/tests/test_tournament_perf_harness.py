import unittest

from tournament_perf_harness import (
    TournamentBenchmarkConfig,
    classify_changed_paths,
    collect_benchmark_contract,
    run_benchmark,
)


class TournamentPerfHarnessTest(unittest.TestCase):
    def make_config(self) -> TournamentBenchmarkConfig:
        return TournamentBenchmarkConfig(
            strategies=("Balanced_Tempo", "Bootstrap", "Random", "Greedy_VP"),
            n_simulations=4,
            warmup_runs=0,
            measured_runs=2,
        )

    def test_benchmark_returns_machine_readable_summary(self) -> None:
        summary = run_benchmark(self.make_config())
        self.assertIn("score", summary)
        self.assertIn("mean_seconds", summary)
        self.assertIn("pstdev_seconds", summary)
        self.assertIn("balance_score", summary)
        self.assertIn("max_abs_deviation", summary)
        self.assertIn("strategy_count", summary)
        self.assertIn("n_simulations", summary)
        self.assertGreater(summary["mean_seconds"], 0.0)
        self.assertEqual(summary["score"], summary["mean_seconds"])
        self.assertEqual(summary["strategy_count"], 4)
        self.assertEqual(summary["n_simulations"], 4)

    def test_contract_is_seeded_and_deterministic(self) -> None:
        first = collect_benchmark_contract(self.make_config())
        second = collect_benchmark_contract(self.make_config())
        self.assertEqual(first["balance_score"], second["balance_score"])
        self.assertEqual(first["max_abs_deviation"], second["max_abs_deviation"])
        self.assertEqual(first["matchup_digest"], second["matchup_digest"])
        self.assertEqual(first["matchups_shape"], second["matchups_shape"])
        self.assertEqual(first["strategy_stats_shape"], second["strategy_stats_shape"])

    def test_changed_path_classifier_flags_forbidden_and_unexpected(self) -> None:
        verdict = classify_changed_paths(
            [
                ".evo/run_0000/graph.json",
                "07-DataScience/game_engine_v3.py",
                "07-DataScience/slides/slide_03_pipeline.png",
                "README.md",
            ]
        )
        self.assertEqual(verdict["allowed"], ["07-DataScience/game_engine_v3.py"])
        self.assertEqual(verdict["forbidden"], ["07-DataScience/slides/slide_03_pipeline.png"])
        self.assertEqual(verdict["unexpected"], ["README.md"])


if __name__ == "__main__":
    unittest.main()
