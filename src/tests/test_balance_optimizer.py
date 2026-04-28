import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from balance_optimizer import (
    BalanceConfig,
    _evaluate_cards,
    _resolve_comparison_seed,
    evaluate_current_balance,
    export_recommendations,
    generate_candidate_changes,
    optimize_balance,
)
from card_parser import parse_all_cards


class BalanceOptimizerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards_df = parse_all_cards()
        cls.business_ids = set(
            cls.cards_df.loc[
                cls.cards_df["type"].astype(str).str.strip().str.title() == "Business",
                "id",
            ].tolist()
        )

    def make_config(self, **overrides) -> BalanceConfig:
        base = {
            "baseline_simulations": 24,
            "search_simulations": 12,
            "final_verification_simulations": 36,
            "candidate_pool_size": 5,
            "max_accepted_edits": 2,
            "minimum_improvement": 0.001,
            "base_seed": 20260316,
        }
        base.update(overrides)
        return BalanceConfig(**base)

    def test_evaluate_current_balance_returns_expected_metrics(self) -> None:
        config = self.make_config()
        result = evaluate_current_balance(self.cards_df, config)

        self.assertIn("balance_score", result.baseline_metrics)
        self.assertIn("max_abs_deviation", result.baseline_metrics)
        self.assertEqual(len(result.baseline_metrics["strategies"]), 8)
        self.assertFalse(result.risky_cards.empty)
        self.assertIn("predicted_win_bias", result.risky_cards.columns)
        self.assertIn("win_bias_residual", result.risky_cards.columns)
        self.assertIn("positive_residual_norm", result.risky_cards.columns)
        self.assertGreater(len(result.candidate_changes), 0)

    def test_same_comparison_seed_gives_identical_metrics(self) -> None:
        config = self.make_config()
        seed = _resolve_comparison_seed(config, "baseline")
        metrics_a = _evaluate_cards(
            self.cards_df,
            config,
            n_simulations=config.baseline_simulations,
            comparison_seed=seed,
            include_results=False,
            include_card_usage=False,
        )
        metrics_b = _evaluate_cards(
            self.cards_df,
            config,
            n_simulations=config.baseline_simulations,
            comparison_seed=seed,
            include_results=False,
            include_card_usage=False,
        )

        cols = ["strategy_a", "strategy_b", "win_rate_a", "win_rate_b", "draw_rate"]
        pd.testing.assert_frame_equal(
            metrics_a["matchups"][cols].reset_index(drop=True),
            metrics_b["matchups"][cols].reset_index(drop=True),
        )
        self.assertEqual(metrics_a["balance_score"], metrics_b["balance_score"])

    def test_v2_pool_surface_matches_expected_card_families(self) -> None:
        business_df = self.cards_df[
            self.cards_df["type"].astype(str).str.strip().str.title() == "Business"
        ].copy()

        self.assertEqual(len(business_df), 56)
        self.assertEqual(
            set(business_df["industry"].unique()),
            {"Service", "Food", "Retail", "Professional", "Tech", "Real Estate", "Trades"},
        )
        self.assertNotIn("Franchise", set(business_df["industry"].unique()))
        self.assertNotIn("BUS-TECH-003", set(business_df["id"].tolist()))
        self.assertIn("BUS-TECH-004", set(business_df["id"].tolist()))
        self.assertIn("BUS-REALESTATE-004", set(business_df["id"].tolist()))

        type_counts = self.cards_df["type"].astype(str).str.strip().str.title().value_counts().to_dict()
        self.assertEqual(type_counts.get("Staff"), 7)
        self.assertEqual(type_counts.get("Boost"), 10)
        self.assertEqual(type_counts.get("Market Condition"), 8)
        self.assertEqual(type_counts.get("Milestone"), 12)

    def test_optimizer_is_reproducible_and_confirmed(self) -> None:
        config = self.make_config(
            baseline_simulations=30,
            search_simulations=16,
            final_verification_simulations=40,
            candidate_pool_size=6,
            minimum_improvement=0.0,
        )
        result_a = optimize_balance(self.cards_df, config)
        result_b = optimize_balance(self.cards_df, config)

        seq_a = [(c.card_id, c.field, c.new_value) for c in result_a.accepted_changes]
        seq_b = [(c.card_id, c.field, c.new_value) for c in result_b.accepted_changes]
        self.assertEqual(seq_a, seq_b)
        self.assertGreaterEqual(len(result_a.accepted_changes), 1)

        for change in result_a.accepted_changes:
            self.assertTrue(change.confirmation_passed)
            self.assertIsNotNone(change.estimated_after_balance_score)
            self.assertIsNotNone(change.confirmed_after_balance_score)
            self.assertEqual(change.after_balance_score, change.confirmed_after_balance_score)
            self.assertGreaterEqual(change.confirmed_improvement, config.minimum_improvement)

    def test_export_recommendations_writes_truthful_artifacts(self) -> None:
        config = self.make_config(
            baseline_simulations=18,
            search_simulations=10,
            final_verification_simulations=24,
            candidate_pool_size=4,
            minimum_improvement=0.0,
        )
        result = optimize_balance(self.cards_df, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_recommendations(result, tmpdir)
            out = Path(tmpdir)
            self.assertTrue((out / "balance_recommendations.csv").exists())
            self.assertTrue((out / "search_trace.csv").exists())
            self.assertTrue((out / "balance_report.json").exists())
            self.assertTrue((out / "card_strength_surrogate.csv").exists())
            self.assertTrue((out / "card_strength_model_report.json").exists())

            recs = pd.read_csv(out / "balance_recommendations.csv")
            self.assertIn("estimated_after_balance_score", recs.columns)
            self.assertIn("confirmed_after_balance_score", recs.columns)
            self.assertIn("confirmed_improvement", recs.columns)

            trace = pd.read_csv(out / "search_trace.csv")
            self.assertIn("evaluation_stage", trace.columns)
            self.assertIn("confirmation_passed", trace.columns)

            payload = json.loads((out / "balance_report.json").read_text(encoding="utf-8"))
            self.assertIn("run_summary", payload)
            self.assertIn("baseline", payload)
            self.assertIn("final_verification", payload)


if __name__ == "__main__":
    unittest.main()
