"""
test_board_evaluator.py — Tests for the board layout evaluator
==============================================================
Run: .venv/bin/python3 -m unittest test_board_evaluator -v
"""

import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


class BoardConfigTest(unittest.TestCase):
    """Validate board_config.py definitions."""

    def test_tile_count(self):
        import board_config
        self.assertEqual(len(board_config.DISTRICT_TILES), 8)

    def test_all_tiles_have_valid_affinities(self):
        import board_config
        VALID = {"Service", "Food", "Retail", "Professional", "Tech", "Real Estate", "Trades"}
        for tile in board_config.DISTRICT_TILES:
            for face in [tile.side_a, tile.side_b]:
                self.assertEqual(
                    len(face.affinities), 2,
                    f"{face.name} must have exactly 2 affinities, got {len(face.affinities)}",
                )
                for aff in face.affinities:
                    self.assertIn(
                        aff, VALID,
                        f"{face.name} has invalid affinity: {aff}",
                    )
                self.assertNotEqual(
                    face.affinities[0], face.affinities[1],
                    f"{face.name} has duplicate affinities: {face.affinities}",
                )

    def test_bonus_slot_indices_valid(self):
        import board_config
        VALID_TYPES = {"Trafico", "Prestigio", "Descuento"}
        max_idx = board_config.LOTS_PER_FRAME
        for tile in board_config.DISTRICT_TILES:
            for face in [tile.side_a, tile.side_b]:
                seen_indices = set()
                for bonus in face.bonuses:
                    self.assertIn(
                        bonus.slot_index, range(max_idx),
                        f"{face.name} has out-of-range slot_index: {bonus.slot_index} (max {max_idx - 1})",
                    )
                    self.assertIn(
                        bonus.bonus_type, VALID_TYPES,
                        f"{face.name} has invalid bonus_type: {bonus.bonus_type}",
                    )
                    self.assertNotIn(
                        bonus.slot_index, seen_indices,
                        f"{face.name} has duplicate slot_index: {bonus.slot_index}",
                    )
                    seen_indices.add(bonus.slot_index)

    def test_plaza_variants_valid(self):
        """All plaza variants have valid affinities and bonus indices."""
        import board_config
        VALID = {"Service", "Food", "Retail", "Professional", "Tech", "Real Estate", "Trades"}
        self.assertGreaterEqual(len(board_config.PLAZA_VARIANTS), 2)
        for variant in board_config.PLAZA_VARIANTS:
            self.assertEqual(variant.name, "Plaza Central")
            self.assertEqual(len(variant.affinities), 2)
            for aff in variant.affinities:
                self.assertIn(aff, VALID)
            for bonus in variant.bonuses:
                self.assertIn(bonus.slot_index, range(board_config.LOTS_PER_FRAME))

    def test_generate_board_is_deterministic(self):
        import board_config
        board_a = board_config.generate_board(player_count=4, seed=42)
        board_b = board_config.generate_board(player_count=4, seed=42)
        self.assertEqual(len(board_a["slots"]), len(board_b["slots"]))
        for sa, sb in zip(board_a["slots"], board_b["slots"]):
            self.assertEqual(sa["tile_name"], sb["tile_name"])
            self.assertEqual(sa["bonus_type"], sb["bonus_type"])

    def test_generate_board_slot_count_v2(self):
        """v2: plaza (4 lots) + N modular frames (4 lots each)."""
        import board_config
        for pc, frames in board_config.CITY_FRAMES.items():
            n_modular = len(frames)
            expected_total = board_config.LOTS_PER_FRAME + n_modular * board_config.LOTS_PER_FRAME
            board = board_config.generate_board(pc, seed=99)
            self.assertEqual(
                len(board["slots"]), expected_total,
                f"Expected {expected_total} slots for {pc}p "
                f"(4 plaza + {n_modular}×4 modular), got {len(board['slots'])}",
            )
            self.assertEqual(len(board["barrio_slots"]), board_config.BARRIO_SLOTS)

    def test_plaza_always_present(self):
        """Plaza Central slots appear at all player counts and seeds."""
        import board_config
        for pc in board_config.CITY_FRAMES:
            for seed in [1, 42, 99, 777]:
                board = board_config.generate_board(pc, seed=seed)
                plaza_slots = [s for s in board["slots"] if s.get("is_plaza")]
                self.assertEqual(
                    len(plaza_slots), board_config.LOTS_PER_FRAME,
                    f"Expected {board_config.LOTS_PER_FRAME} plaza slots at {pc}p seed={seed}",
                )
                for ps in plaza_slots:
                    self.assertEqual(ps["tile_name"], "Plaza Central")
                # Affinities always Professional + Real Estate
                self.assertEqual(plaza_slots[0]["affinities"], ("Professional", "Real Estate"))

    def test_plaza_variant_selection(self):
        """Different seeds can produce different plaza bonus patterns."""
        import board_config
        bonus_patterns = set()
        for seed in range(50):
            board = board_config.generate_board(4, seed=seed)
            plaza_slots = [s for s in board["slots"] if s.get("is_plaza")]
            pattern = tuple(s["bonus_type"] for s in plaza_slots)
            bonus_patterns.add(pattern)
        # With 3 variants and 50 seeds, we should see at least 2 distinct patterns
        self.assertGreaterEqual(len(bonus_patterns), 2, "Plaza variants not rotating")

    def test_2p_has_darsena_sur(self):
        """2p layout always includes Darsena Sur (south bank matters)."""
        import board_config
        for seed in [1, 42, 99, 777]:
            board = board_config.generate_board(2, seed=seed)
            frame_names = {s["frame_name"] for s in board["slots"]}
            self.assertIn("Darsena Sur", frame_names, f"Darsena Sur missing at 2p seed={seed}")

    def test_different_seeds_give_different_boards(self):
        import board_config
        board_a = board_config.generate_board(4, seed=1)
        board_b = board_config.generate_board(4, seed=2)
        # Only compare modular slots (skip plaza which is always the same)
        modular_a = tuple(s["tile_name"] for s in board_a["slots"] if not s.get("is_plaza"))
        modular_b = tuple(s["tile_name"] for s in board_b["slots"] if not s.get("is_plaza"))
        self.assertNotEqual(modular_a, modular_b)

    def test_row_modifiers_present(self):
        """All slots have row_modifier values."""
        import board_config
        board = board_config.generate_board(4, seed=42)
        for slot in board["slots"]:
            self.assertIn("row_modifier", slot)
            self.assertIn(slot["row_modifier"], [0.1, -0.1])


class BoardEvaluatorTest(unittest.TestCase):
    """Validate board_evaluator.py calculations."""

    @classmethod
    def setUpClass(cls):
        from board_evaluator import load_physical_businesses, _industry_stats
        cls.physical_biz = load_physical_businesses()
        cls.industry_stats = _industry_stats(cls.physical_biz)

    def test_physical_business_count(self):
        self.assertGreater(len(self.physical_biz), 30)
        self.assertLess(len(self.physical_biz), 60)

    def test_industry_stats_cover_physical_industries(self):
        physical_industries = set(self.physical_biz["industry"].unique())
        for ind in physical_industries:
            self.assertIn(ind, self.industry_stats)
            self.assertGreater(self.industry_stats[ind]["count"], 0)

    def test_tech_has_no_physical_cards(self):
        tech_count = self.industry_stats.get("Tech", {}).get("count", -1)
        self.assertEqual(tech_count, 0)

    def test_slot_ev_is_non_negative(self):
        """All slot EVs should be non-negative (Interior row modifier may reduce but not go below 0)."""
        import board_config
        from board_evaluator import compute_slot_expected_value

        board = board_config.generate_board(4, seed=42)
        for slot in board["slots"]:
            ev = compute_slot_expected_value(slot, self.industry_stats, board_config)
            self.assertGreaterEqual(ev, 0.0, f"Negative EV at {slot['frame_name']}/{slot['tile_name']} L{slot['slot_idx']}")

    def test_fachada_higher_than_interior(self):
        """Fachada (row 0) slots should have higher EV than Interior (row 1) same-frame slots."""
        import board_config
        from board_evaluator import compute_slot_expected_value

        # Use identical slot except for row_modifier
        base = {
            "affinities": ("Food", "Retail"),
            "bonus_type": None,
            "frame_features": ("riverfront",),
        }
        fachada = {**base, "row_modifier": 0.1}
        interior = {**base, "row_modifier": -0.1}
        ev_f = compute_slot_expected_value(fachada, self.industry_stats, board_config)
        ev_i = compute_slot_expected_value(interior, self.industry_stats, board_config)
        self.assertGreater(ev_f, ev_i)

    def test_frame_compatibility_bonus(self):
        """Compatible insert gets higher EV than incompatible."""
        import board_config
        from board_evaluator import compute_slot_expected_value

        base = {"affinities": ("Food", "Retail"), "bonus_type": None,
                "frame_features": ("riverfront",), "row_modifier": 0.0}
        compatible = {**base, "frame_compatible": True}
        incompatible = {**base, "frame_compatible": False}
        ev_compat = compute_slot_expected_value(compatible, self.industry_stats, board_config)
        ev_incompat = compute_slot_expected_value(incompatible, self.industry_stats, board_config)
        self.assertGreater(ev_compat, ev_incompat)

    def test_bonus_slot_has_higher_ev_than_plain(self):
        import board_config
        from board_evaluator import compute_slot_expected_value

        base_slot = {"affinities": ("Food", "Retail"), "bonus_type": None, "frame_features": (), "row_modifier": 0.0}
        bonus_slot = {"affinities": ("Food", "Retail"), "bonus_type": "Trafico", "frame_features": (), "row_modifier": 0.0}
        ev_plain = compute_slot_expected_value(base_slot, self.industry_stats, board_config)
        ev_bonus = compute_slot_expected_value(bonus_slot, self.industry_stats, board_config)
        self.assertGreater(ev_bonus, ev_plain)

    def test_evaluate_board_returns_valid_metrics(self):
        import board_config
        from board_evaluator import evaluate_board

        board = board_config.generate_board(4, seed=42)
        metrics = evaluate_board(board, self.industry_stats, board_config)
        self.assertGreaterEqual(metrics.slot_cv, 0.0)
        self.assertGreaterEqual(metrics.zone_parity, 0.0)
        self.assertGreaterEqual(metrics.industry_coverage, 0.0)
        self.assertLessEqual(metrics.industry_coverage, 1.0)
        self.assertGreater(metrics.centro_vs_barrio, 0.0)
        self.assertGreaterEqual(metrics.overall_balance, 0.0)
        self.assertLessEqual(metrics.overall_balance, 1.0)

    def test_full_evaluation_runs(self):
        from board_evaluator import run_evaluation
        results = run_evaluation(n_boards=10, player_counts=(4,), base_seed=42)
        self.assertIn(4, results["by_player_count"])
        grand = results["grand_mean"]
        self.assertGreater(grand["mean_overall_balance"], 0.0)

    def test_evaluation_is_deterministic(self):
        from board_evaluator import run_evaluation
        r1 = run_evaluation(n_boards=5, player_counts=(4,), base_seed=123)
        r2 = run_evaluation(n_boards=5, player_counts=(4,), base_seed=123)
        self.assertAlmostEqual(
            r1["grand_mean"]["mean_overall_balance"],
            r2["grand_mean"]["mean_overall_balance"],
            places=6,
        )

    def test_barrio_ev_is_positive(self):
        import board_config
        from board_evaluator import compute_barrio_slot_ev
        ev = compute_barrio_slot_ev(board_config)
        self.assertGreater(ev, 0.0)

    def test_city_slot_graph_matches_city_slot_count(self):
        import board_config
        from board_graph import build_city_slot_graph

        for player_count in (2, 3, 4):
            board = board_config.generate_board(player_count, seed=42)
            graph = build_city_slot_graph(board)
            self.assertEqual(graph.number_of_nodes(), len(board["slots"]))

    def test_intra_frame_graph_edges_are_orthogonal_only(self):
        import board_config
        from board_graph import build_city_slot_graph

        board = board_config.generate_board(4, seed=42)
        graph = build_city_slot_graph(board)
        self.assertTrue(graph.has_edge("Plaza Central::L0", "Plaza Central::L1"))
        self.assertTrue(graph.has_edge("Plaza Central::L0", "Plaza Central::L2"))
        self.assertTrue(graph.has_edge("Plaza Central::L1", "Plaza Central::L3"))
        self.assertTrue(graph.has_edge("Plaza Central::L2", "Plaza Central::L3"))
        self.assertFalse(graph.has_edge("Plaza Central::L0", "Plaza Central::L3"))
        self.assertFalse(graph.has_edge("Plaza Central::L1", "Plaza Central::L2"))

    def test_graph_metrics_are_present_and_plaza_distance_is_finite(self):
        import board_config
        from board_graph import compute_graph_metrics

        board = board_config.generate_board(4, seed=42)
        diagnostics = compute_graph_metrics(board)
        slot_metrics = diagnostics["slot_metrics"]

        for column in (
            "graph_degree",
            "graph_betweenness",
            "graph_closeness",
            "graph_distance_to_plaza",
            "graph_eigenvector",
            "graph_component_id",
        ):
            self.assertIn(column, slot_metrics.columns)
        self.assertTrue(np.isfinite(slot_metrics["graph_distance_to_plaza"]).all())
        plaza_distances = slot_metrics[slot_metrics["frame_name"] == "Plaza Central"]["graph_distance_to_plaza"]
        self.assertTrue((plaza_distances == 0.0).all())

    def test_frame_graph_metrics_follow_expected_direction(self):
        import board_config
        from board_graph import compute_graph_metrics

        board = board_config.generate_board(4, seed=42)
        frame_metrics = compute_graph_metrics(board)["frame_metrics"].set_index("frame_name")

        self.assertEqual(frame_metrics.loc["Plaza Central", "frame_distance_to_plaza"], 0.0)
        self.assertLess(
            frame_metrics.loc["Periferia", "frame_closeness"],
            frame_metrics.loc["Plaza Central", "frame_closeness"],
        )
        self.assertGreater(
            frame_metrics["frame_betweenness"].max(),
            0.0,
        )

    def test_run_evaluation_includes_graph_summary(self):
        from board_evaluator import run_evaluation

        results = run_evaluation(n_boards=3, player_counts=(4,), base_seed=99)
        self.assertIn("graph_grand_mean", results)
        self.assertIn("frame_betweenness_gap", results["graph_grand_mean"])


if __name__ == "__main__":
    unittest.main()
