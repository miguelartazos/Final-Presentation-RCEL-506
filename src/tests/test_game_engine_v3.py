import unittest

from card_parser import parse_all_cards
from game_engine_v3 import (
    BusinessCard,
    DEPARTMENTS,
    DEFAULT_TOURNAMENT_STRATEGIES,
    GameConfig,
    GameEngine,
    _build_board_state,
    _placement_candidates,
    _profile_for,
    evaluate_strategy_tournament,
)


class GameEngineV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards_df = parse_all_cards()

    def test_config_matches_active_build_defaults(self) -> None:
        cfg = GameConfig()
        self.assertEqual(cfg.starting_employees, 9)
        self.assertEqual(cfg.location_fee, 2.75)
        self.assertEqual(cfg.turns_per_quarter, 4)
        self.assertEqual(cfg.giro_cost, 2.0)
        self.assertEqual(cfg.focus_cube_cost, 2.0)

    def test_single_game_is_deterministic(self) -> None:
        engine = GameEngine(self.cards_df)
        result_a = engine.simulate_game("Balanced_Tempo", "Bootstrap", seed=42)
        result_b = engine.simulate_game("Balanced_Tempo", "Bootstrap", seed=42)

        self.assertEqual(result_a["winner"], result_b["winner"])
        self.assertEqual(result_a["score_1"], result_b["score_1"])
        self.assertEqual(result_a["score_2"], result_b["score_2"])
        self.assertEqual(result_a["reserve_1"], result_b["reserve_1"])
        self.assertEqual(result_a["reserve_2"], result_b["reserve_2"])

    def test_results_expose_quarter_planning_surface(self) -> None:
        engine = GameEngine(self.cards_df)
        result = engine.simulate_game("Synergy_Builder", "Brand_Rush", seed=7)

        self.assertIn(result["reserve_1"], DEPARTMENTS)
        self.assertIn(result["reserve_2"], DEPARTMENTS)
        self.assertGreaterEqual(result["giro_uses_1"], 0)
        self.assertGreaterEqual(result["giro_uses_2"], 0)
        self.assertGreaterEqual(result["focus_spent_1"], 0)
        self.assertGreaterEqual(result["focus_spent_2"], 0)

    def test_tournament_output_still_matches_optimizer_contract(self) -> None:
        engine = GameEngine(self.cards_df)
        metrics = evaluate_strategy_tournament(
            engine=engine,
            strategies=list(DEFAULT_TOURNAMENT_STRATEGIES[:4]),
            n_simulations=6,
            base_seed=20260401,
            include_card_usage=True,
            include_matchup_results=True,
            verbose=False,
        )

        self.assertIn("matchups", metrics)
        self.assertIn("results", metrics)
        self.assertIn("card_usage", metrics)
        self.assertIn("balance_score", metrics)
        self.assertFalse(metrics["matchups"].empty)

    def test_board_enabled_surface_emits_slot_and_zone_telemetry(self) -> None:
        engine = GameEngine(self.cards_df, GameConfig(board_enabled=True))
        result = engine.simulate_game("Balanced_Tempo", "Bootstrap", seed=42)

        self.assertTrue(result["board_enabled"])
        self.assertIn("city_slot_usage", result)
        self.assertIn("starting_zone_1", result)
        self.assertIn("blocked_city_launches_1", result)
        self.assertIsInstance(result["city_slot_usage"], list)
        self.assertGreaterEqual(result["city_launches_1"], 0)
        self.assertGreaterEqual(result["barrio_launches_1"], 0)

    def test_board_enabled_tournament_exposes_board_telemetry_tables(self) -> None:
        engine = GameEngine(self.cards_df, GameConfig(board_enabled=True))
        metrics = evaluate_strategy_tournament(
            engine=engine,
            strategies=list(DEFAULT_TOURNAMENT_STRATEGIES[:2]),
            n_simulations=4,
            base_seed=20260409,
            include_card_usage=True,
            include_matchup_results=True,
            verbose=False,
        )

        self.assertIn("board_telemetry", metrics)
        self.assertIn("win_rate_by_starting_zone", metrics["board_telemetry"])
        self.assertIn("slot_usage", metrics["board_telemetry"])

    def test_two_slot_business_has_legal_board_placements(self) -> None:
        cfg = GameConfig(board_enabled=True)
        board_state = _build_board_state(("Player_1", "Player_2"), cfg, seed=42)
        player_card = BusinessCard(
            id="TEST-PREMIUM-001",
            name="Test Premium",
            industry="Service",
            tier="Premium",
            cost=8.0,
            income=3.0,
            valuation_points=5.0,
            exit_value=5.0,
            tempo="Estable",
            mode="Physical",
            staff_min=2,
            income_scaled=0.0,
            synergy_gives="",
            synergy_receives="",
        )
        from game_engine_v3 import Player

        player = Player(
            name="Player_1",
            strategy="Balanced_Tempo",
            cash=20.0,
            brand=5,
            employees_reserve=9,
            total_employees=9,
        )
        candidates = _placement_candidates(
            player=player,
            card=player_card,
            level=3,
            profile=_profile_for(player.strategy),
            cfg=cfg,
            board_state=board_state,
        )
        self.assertTrue(any(len(candidate.slot_ids) == 2 for candidate in candidates))
        self.assertTrue(any(candidate.area == "Ciudad" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
