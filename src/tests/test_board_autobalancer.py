import json
import tempfile
import unittest
from pathlib import Path

from board_autobalancer import (
    BoardSearchConfig,
    _dynamic_balance_report,
    _face_diagnostics,
    _generate_candidates,
    export_board_recommendations,
    optimize_board,
)
from board_config import build_runtime_config


class BoardAutobalancerSmokeTest(unittest.TestCase):
    def make_config(self) -> BoardSearchConfig:
        return BoardSearchConfig(
            baseline_boards=6,
            search_boards=3,
            final_verification_boards=8,
            max_accepted_changes=1,
            minimum_improvement=0.0,
        )

    def test_optimize_board_returns_metrics_and_runtime(self) -> None:
        result = optimize_board(config=self.make_config())

        self.assertIn("grand_mean", result.baseline_metrics)
        self.assertIn("grand_mean", result.final_metrics)
        self.assertIn("graph_grand_mean", result.final_metrics)
        self.assertGreater(result.baseline_metrics["grand_mean"]["mean_overall_balance"], 0.0)
        self.assertFalse(result.final_face_diagnostics.empty)

    def test_optimize_board_is_reproducible(self) -> None:
        result_a = optimize_board(config=self.make_config())
        result_b = optimize_board(config=self.make_config())

        seq_a = [(change.field_path, str(change.new_value)) for change in result_a.accepted_changes]
        seq_b = [(change.field_path, str(change.new_value)) for change in result_b.accepted_changes]
        self.assertEqual(seq_a, seq_b)
        self.assertEqual(
            result_a.final_metrics["grand_mean"]["mean_overall_balance"],
            result_b.final_metrics["grand_mean"]["mean_overall_balance"],
        )

    def test_frame_targeted_candidates_exist_without_global_fee_change(self) -> None:
        runtime = build_runtime_config()
        diagnostics = _face_diagnostics(runtime)
        self.assertIn("compatible_frame_mean_closeness", diagnostics.columns)
        self.assertIn("compatible_frame_mean_betweenness", diagnostics.columns)
        config = self.make_config()
        dynamic = _dynamic_balance_report(runtime, config, n_simulations=4, base_seed=20260409)

        candidates = _generate_candidates(
            runtime,
            diagnostics,
            current_overall=0.5,
            dynamic_signals=dynamic,
            config=config,
        )
        descriptions = [candidate.description for candidate, _ in candidates]
        self.assertTrue(any("Darsena Sur" in desc or "weak frame" in desc for desc in descriptions))
        self.assertFalse(any(candidate.field_path == "LOCATION_FEE" for candidate, _ in candidates))

    def test_global_fee_candidate_is_opt_in(self) -> None:
        runtime = build_runtime_config()
        diagnostics = _face_diagnostics(runtime)
        config = self.make_config()
        dynamic = _dynamic_balance_report(runtime, config, n_simulations=4, base_seed=20260409)

        allow_fee = BoardSearchConfig(**{**config.__dict__, "allow_global_fee_changes": True})
        candidates = _generate_candidates(
            runtime,
            diagnostics,
            current_overall=0.5,
            dynamic_signals=dynamic,
            config=allow_fee,
        )
        self.assertTrue(any(candidate.field_path == "LOCATION_FEE" for candidate, _ in candidates))

    def test_export_board_recommendations_writes_expected_artifacts(self) -> None:
        result = optimize_board(config=self.make_config())

        with tempfile.TemporaryDirectory() as tmpdir:
            export_board_recommendations(result, tmpdir)
            out = Path(tmpdir)

            self.assertTrue((out / "board_recommendations.csv").exists())
            self.assertTrue((out / "board_search_trace.csv").exists())
            self.assertTrue((out / "board_face_diagnostics_before.csv").exists())
            self.assertTrue((out / "board_face_diagnostics_after.csv").exists())
            self.assertTrue((out / "board_slot_graph_metrics.csv").exists())
            self.assertTrue((out / "board_frame_graph_metrics.csv").exists())
            self.assertTrue((out / "board_graph_report.json").exists())
            self.assertTrue((out / "board_balance_report.json").exists())

            payload = json.loads((out / "board_balance_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metric_direction"]["overall_balance"], "higher_is_better")
            self.assertIn("baseline_graph", payload)
            self.assertIn("final_graph", payload)
            self.assertIn("graph_metric_direction", payload)
            self.assertIn("optimized_runtime", payload)


if __name__ == "__main__":
    unittest.main()
