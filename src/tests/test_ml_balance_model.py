import json
import tempfile
import unittest
from pathlib import Path

from balance_optimizer import BalanceConfig, _evaluate_cards
from card_parser import parse_all_cards
from ml_balance_model import (
    SurrogateConfig,
    export_card_strength_surrogate,
    fit_card_strength_surrogate,
)


class CardStrengthSurrogateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards_df = parse_all_cards()
        cls.metrics = _evaluate_cards(
            cls.cards_df,
            BalanceConfig(
                baseline_simulations=18,
                search_simulations=8,
                final_verification_simulations=24,
                candidate_pool_size=4,
                minimum_improvement=0.0,
                base_seed=20260414,
            ),
            n_simulations=18,
            comparison_seed=20260414,
            include_results=True,
            include_card_usage=True,
        )

    def test_surrogate_returns_cross_validated_predictions(self) -> None:
        report = fit_card_strength_surrogate(
            self.cards_df,
            self.metrics,
            SurrogateConfig(seed=20260414, n_folds=5),
        )

        predictions = report["predictions"]
        self.assertEqual(len(predictions), 56)
        self.assertIn("predicted_win_bias", predictions.columns)
        self.assertIn("win_bias_residual", predictions.columns)
        self.assertIn("positive_residual_norm", predictions.columns)
        self.assertIn("fold", predictions.columns)
        self.assertGreaterEqual(report["baseline_metrics"]["mae"], 0.0)
        self.assertGreaterEqual(report["oof_metrics"]["mae"], 0.0)

    def test_surrogate_export_writes_expected_files(self) -> None:
        report = fit_card_strength_surrogate(
            self.cards_df,
            self.metrics,
            SurrogateConfig(seed=20260414, n_folds=4),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_card_strength_surrogate(report, tmpdir)
            out = Path(tmpdir)
            self.assertTrue((out / "card_strength_surrogate.csv").exists())
            self.assertTrue((out / "card_strength_coefficients.csv").exists())
            self.assertTrue((out / "card_strength_model_report.json").exists())

            payload = json.loads((out / "card_strength_model_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["model_type"], "ridge_regression")
            self.assertEqual(payload["target_column"], "win_bias")
            self.assertIn("oof_metrics", payload)


if __name__ == "__main__":
    unittest.main()
