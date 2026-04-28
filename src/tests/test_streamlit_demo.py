"""
Sanity tests for the Streamlit app.

The UI is not exercised here. These tests cover the deploy-facing data
and helper functions so a stale dataset, missing thumbnails, or broken
import path is caught before the demo.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit_demo as demo  # noqa: E402


class StreamlitDemoHelpersTest(unittest.TestCase):
    def test_loads_frozen_card_dataset(self) -> None:
        cards_df = demo.load_cards_dataframe.__wrapped__()
        self.assertEqual(len(cards_df), 93)
        self.assertIn("BUS-FOOD-001", set(cards_df["id"]))
        self.assertGreaterEqual(cards_df["industry"].nunique(), 7)

    def test_gallery_uses_committed_thumbnails_and_metadata(self) -> None:
        catalog = demo.discover_card_assets.__wrapped__()
        self.assertEqual(len(catalog), 61)
        self.assertIn("BUS-FOOD-001", set(catalog["id"]))
        coffee = catalog[catalog["id"] == "BUS-FOOD-001"].iloc[0]
        self.assertEqual(coffee["name"], "Coffee Cart")
        self.assertTrue(str(coffee["image_path"]).endswith(".png"))

    def test_strategy_count_guard(self) -> None:
        with self.assertRaises(ValueError):
            demo.clamp_strategies([])
        with self.assertRaises(ValueError):
            demo.clamp_strategies(["Random"])
        self.assertEqual(demo.clamp_strategies(["Random", "Greedy_VP"]), ["Random", "Greedy_VP"])

    def test_default_tournament_matches_smoke_demo_balance_score(self) -> None:
        result = demo.run_tournament.__wrapped__(
            strategies=tuple(demo.DEFAULT_TOURNAMENT_STRATEGIES[: demo.DEFAULT_STRATEGY_COUNT]),
            n_simulations=1,
            base_seed=demo.DEFAULT_BASE_SEED,
            include_card_usage=False,
            include_matchup_results=True,
        )
        self.assertAlmostEqual(result["balance_score"], 0.4924, places=4)
        self.assertFalse(result["non_mirror"].empty)


if __name__ == "__main__":
    unittest.main()
