import tempfile
import unittest
from pathlib import Path

from run_balance_pipeline import run_pipeline
from board_autobalancer import BoardSearchConfig


class RunBalancePipelineSmokeTest(unittest.TestCase):
    def test_pipeline_writes_candidate_comparison_reports(self) -> None:
        import run_balance_pipeline as pipeline_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline_module.OUTPUT_DIR = root
            pipeline_module.BOARD_DIR = root / "board"
            pipeline_module.GAME_DIR = root / "game"
            pipeline_module.UNIFIED_DIR = root / "unified"

            summary = run_pipeline(
                board_search_config=BoardSearchConfig(
                    baseline_boards=4,
                    search_boards=2,
                    final_verification_boards=6,
                    dynamic_search_simulations=2,
                    dynamic_verification_simulations=4,
                    max_accepted_changes=1,
                    minimum_improvement=0.0,
                    allow_global_fee_changes=False,
                ),
                tournament_simulations=2,
                card_baseline_simulations=6,
                strategies=["Random", "Balanced_Tempo"],
            )

            self.assertIn("board", summary)
            self.assertIn("ml_surrogate", summary)
            self.assertIn("graph_metrics", summary)
            self.assertIn("graph_metrics", summary["board"])
            self.assertTrue((pipeline_module.BOARD_DIR / "board_slot_graph_metrics.csv").exists())
            self.assertTrue((pipeline_module.BOARD_DIR / "board_frame_graph_metrics.csv").exists())
            self.assertTrue((pipeline_module.BOARD_DIR / "board_graph_report.json").exists())
            self.assertTrue((pipeline_module.GAME_DIR / "candidate_zone_comparison.csv").exists())
            self.assertTrue((pipeline_module.GAME_DIR / "candidate_frame_comparison.csv").exists())
            self.assertTrue((pipeline_module.GAME_DIR / "candidate_verification_report.json").exists())
            self.assertTrue((pipeline_module.GAME_DIR / "card_strength_surrogate.csv").exists())
            self.assertTrue((pipeline_module.GAME_DIR / "card_strength_model_report.json").exists())


if __name__ == "__main__":
    unittest.main()
