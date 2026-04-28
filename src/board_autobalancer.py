"""
board_autobalancer.py — deterministic board autoresearch for Ciudad Viva
========================================================================
Searches over an in-memory RuntimeBoardConfig while keeping the fixed
board evaluator untouched. Exports artifacts analogous to the card
optimizer and returns accepted changes that can then be applied to
board_config.py intentionally.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from board_graph import compute_graph_metrics, summarize_graph_metrics
from board_config import (
    RuntimeBoardConfig,
    SlotBonus,
    build_runtime_config,
    generate_board_from_runtime,
)
from board_evaluator import (
    _industry_stats,
    build_board_diagnostics,
    compute_slot_expected_value,
    evaluate_board,
    load_physical_businesses,
)
from card_parser import parse_all_cards
from game_engine_v3 import GameConfig, GameEngine, evaluate_strategy_tournament


OUTPUT_DIR = Path(__file__).resolve().parent / "optimizer_outputs" / "board"
VALID_INDUSTRIES = ("Service", "Food", "Retail", "Professional", "Tech", "Real Estate", "Trades")
VALID_BONUS_TYPES = ("Trafico", "Prestigio", "Descuento")


@dataclass
class BoardSearchConfig:
    player_counts: tuple[int, ...] = (2, 3, 4)
    baseline_boards: int = 120
    search_boards: int = 48
    final_verification_boards: int = 240
    dynamic_search_simulations: int = 6
    dynamic_verification_simulations: int = 12
    dynamic_player_count: int = 2
    dynamic_strategies: tuple[str, ...] = ("Random", "Greedy_VP", "Cash_Machine", "Balanced_Tempo")
    max_accepted_changes: int = 5
    minimum_improvement: float = 0.01
    zone_gap_tolerance: float = 0.01
    allow_global_fee_changes: bool = False
    base_seed: int = 20260409


@dataclass
class BoardCandidateChange:
    description: str
    field_path: str
    old_value: Any
    new_value: Any
    change_type: str
    before_overall_balance: float | None = None
    estimated_after_overall_balance: float | None = None
    confirmed_after_overall_balance: float | None = None
    estimated_improvement: float | None = None
    confirmed_improvement: float | None = None
    estimated_zone_gap: float | None = None
    confirmed_zone_gap: float | None = None
    estimated_weak_zone: str | None = None
    confirmed_weak_zone: str | None = None
    estimated_weak_zone_win_rate: float | None = None
    confirmed_weak_zone_win_rate: float | None = None
    estimated_graph_disparity: float | None = None
    confirmed_graph_disparity: float | None = None
    estimated_overcentralized_frame: str | None = None
    confirmed_overcentralized_frame: str | None = None
    confirmation_passed: bool = False
    dynamic_gate_passed: bool = False


@dataclass
class BoardOptimizationResult:
    config: BoardSearchConfig
    baseline_metrics: dict[str, Any]
    optimized_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    baseline_face_diagnostics: pd.DataFrame
    final_face_diagnostics: pd.DataFrame
    baseline_dynamic_metrics: dict[str, Any]
    optimized_dynamic_metrics: dict[str, Any]
    final_dynamic_metrics: dict[str, Any]
    accepted_changes: list[BoardCandidateChange]
    search_trace: pd.DataFrame
    optimized_runtime: RuntimeBoardConfig


def _runtime_module(runtime: RuntimeBoardConfig) -> SimpleNamespace:
    module = SimpleNamespace(
        TRAFFIC_BONUS_IR=runtime.traffic_bonus_ir,
        PRESTIGE_BONUS_MARCA=runtime.prestige_bonus_marca,
        DISCOUNT_BONUS_K=runtime.discount_bonus_k,
        SPATIAL_FEATURE_VALUES=runtime.spatial_feature_values,
        ROW_MODIFIERS=runtime.row_modifiers,
        FRAME_AFFINITY_PREFERENCE=runtime.frame_affinity_preference,
        FRAME_COMPATIBILITY_BONUS=runtime.frame_compatibility_bonus,
        LOCATION_FEE=runtime.location_fee,
        BARRIO_SLOTS=runtime.barrio_slots,
        BARRIO_OPEN_SLOTS=runtime.barrio_open_slots,
        BARRIO_UNLOCK_SLOTS=runtime.barrio_unlock_slots,
        LOTS_PER_FRAME=runtime.lots_per_frame,
        FRAME_COLS=runtime.frame_cols,
        FRAME_ROWS=runtime.frame_rows,
        SLOT_SIZE_BY_TIER=runtime.slot_size_by_tier,
        PLAZA_VARIANTS=runtime.plaza_variants,
        PLAZA_FRAME=runtime.plaza_frame,
        CITY_FRAMES=runtime.city_frames,
        FRAME_ADJACENCY_BY_PLAYER_COUNT=runtime.frame_adjacency_by_player_count,
        DISTRICT_TILES=runtime.district_tiles,
    )
    module.generate_board = lambda player_count, seed: generate_board_from_runtime(runtime, player_count, seed)
    return module


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


_CACHED_CARDS_DF: pd.DataFrame | None = None


def _cards_df() -> pd.DataFrame:
    global _CACHED_CARDS_DF
    if _CACHED_CARDS_DF is None:
        _CACHED_CARDS_DF = parse_all_cards()
    return _CACHED_CARDS_DF.copy(deep=True)


def _dynamic_balance_report(
    runtime: RuntimeBoardConfig,
    config: BoardSearchConfig,
    n_simulations: int,
    base_seed: int,
) -> dict[str, Any]:
    engine = GameEngine(
        _cards_df(),
        GameConfig(
            board_enabled=True,
            player_count=config.dynamic_player_count,
            runtime_board_config=runtime,
        ),
    )
    metrics = evaluate_strategy_tournament(
        engine=engine,
        strategies=list(config.dynamic_strategies),
        n_simulations=n_simulations,
        base_seed=base_seed,
        include_card_usage=False,
        include_matchup_results=True,
        verbose=False,
    )

    zone_df = metrics["board_telemetry"]["win_rate_by_starting_zone"].copy()
    frame_df = metrics["board_telemetry"]["frame_usage"].copy()
    slot_df = metrics["board_telemetry"]["slot_usage"].copy()
    industry_df = metrics["board_telemetry"]["industry_zone_matrix"].copy()

    if zone_df.empty:
        strongest_zone = weakest_zone = None
        strongest_zone_win_rate = weakest_zone_win_rate = 0.5
        zone_gap = 0.0
    else:
        strongest = zone_df.sort_values(["win_rate", "games"], ascending=[False, False]).iloc[0]
        weakest = zone_df.sort_values(["win_rate", "games"], ascending=[True, False]).iloc[0]
        strongest_zone = str(strongest["starting_zone"])
        weakest_zone = str(weakest["starting_zone"])
        strongest_zone_win_rate = float(strongest["win_rate"])
        weakest_zone_win_rate = float(weakest["win_rate"])
        zone_gap = round(strongest_zone_win_rate - weakest_zone_win_rate, 4)

    top_slot_share = 0.0
    if not slot_df.empty:
        total_slot_uses = float(slot_df["times_used"].sum())
        top_slot_share = round(float(slot_df["times_used"].max() / max(total_slot_uses, 1.0)), 4)

    weak_zone_industries: list[str] = []
    if weakest_zone and not industry_df.empty:
        weak_zone_rows = industry_df[industry_df["starting_zone"] == weakest_zone].copy()
        if not weak_zone_rows.empty:
            weak_zone_rows = weak_zone_rows.sort_values(["count", "industry"], ascending=[False, True])
            weak_zone_industries = weak_zone_rows["industry"].head(3).tolist()

    return {
        "balance_score": metrics["balance_score"],
        "max_abs_deviation": metrics["max_abs_deviation"],
        "zone_gap": zone_gap,
        "strongest_zone": strongest_zone,
        "strongest_zone_win_rate": strongest_zone_win_rate,
        "weakest_zone": weakest_zone,
        "weakest_zone_win_rate": weakest_zone_win_rate,
        "top_slot_share": top_slot_share,
        "weak_zone_industries": weak_zone_industries,
        "board_telemetry": metrics["board_telemetry"],
    }


def _evaluate_runtime(
    runtime: RuntimeBoardConfig,
    n_boards: int,
    player_counts: tuple[int, ...],
    base_seed: int,
) -> dict[str, Any]:
    module = _runtime_module(runtime)
    physical = load_physical_businesses()
    industry_stats = _industry_stats(physical)

    by_player_count: dict[int, dict[str, float]] = {}
    graph_by_player_count: dict[int, dict[str, Any]] = {}
    all_metrics = []
    for player_count in player_counts:
        metrics_list = []
        graph_summary = None
        for index in range(n_boards):
            board_seed = _stable_seed(base_seed, player_count, index)
            board = module.generate_board(player_count, board_seed)
            metrics_list.append(evaluate_board(board, industry_stats, module))
            if graph_summary is None:
                graph_summary = summarize_graph_metrics(compute_graph_metrics(board))
        by_player_count[player_count] = {
            "mean_slot_cv": float(sum(m.slot_cv for m in metrics_list) / len(metrics_list)),
            "mean_zone_parity": float(sum(m.zone_parity for m in metrics_list) / len(metrics_list)),
            "mean_industry_coverage": float(sum(m.industry_coverage for m in metrics_list) / len(metrics_list)),
            "mean_affinity_balance": float(sum(m.affinity_balance for m in metrics_list) / len(metrics_list)),
            "mean_centro_vs_barrio": float(sum(m.centro_vs_barrio for m in metrics_list) / len(metrics_list)),
            "mean_overall_balance": float(sum(m.overall_balance for m in metrics_list) / len(metrics_list)),
        }
        graph_by_player_count[player_count] = graph_summary or {}
        all_metrics.extend(metrics_list)

    grand_mean = {
        "mean_slot_cv": float(sum(m.slot_cv for m in all_metrics) / len(all_metrics)),
        "mean_zone_parity": float(sum(m.zone_parity for m in all_metrics) / len(all_metrics)),
        "mean_industry_coverage": float(sum(m.industry_coverage for m in all_metrics) / len(all_metrics)),
        "mean_affinity_balance": float(sum(m.affinity_balance for m in all_metrics) / len(all_metrics)),
        "mean_centro_vs_barrio": float(sum(m.centro_vs_barrio for m in all_metrics) / len(all_metrics)),
        "mean_overall_balance": float(sum(m.overall_balance for m in all_metrics) / len(all_metrics)),
    }
    numeric_keys = [
        key
        for key in next(iter(graph_by_player_count.values()), {}).keys()
        if isinstance(next(iter(graph_by_player_count.values()), {}).get(key), (int, float))
    ]
    graph_grand_mean = {
        key: float(sum(float(graph_by_player_count[pc][key]) for pc in player_counts) / len(player_counts))
        for key in numeric_keys
    } if graph_by_player_count else {}
    overcentralized_counts: dict[str, int] = {}
    for summary in graph_by_player_count.values():
        frame_name = summary.get("overcentralized_frame")
        if not frame_name:
            continue
        overcentralized_counts[str(frame_name)] = overcentralized_counts.get(str(frame_name), 0) + 1
    graph_grand_mean["most_common_overcentralized_frame"] = (
        sorted(overcentralized_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if overcentralized_counts
        else None
    )
    return {
        "by_player_count": by_player_count,
        "grand_mean": grand_mean,
        "graph_by_player_count": graph_by_player_count,
        "graph_grand_mean": graph_grand_mean,
    }


def _face_diagnostics(runtime: RuntimeBoardConfig) -> pd.DataFrame:
    module = _runtime_module(runtime)
    physical = load_physical_businesses()
    industry_stats = _industry_stats(physical)
    frame_metrics_by_player_count: dict[int, pd.DataFrame] = {}
    for player_count in sorted(runtime.city_frames):
        board = generate_board_from_runtime(runtime, player_count, seed=0)
        frame_metrics_by_player_count[player_count] = build_board_diagnostics(
            board,
            industry_stats,
            module,
        )["frame_metrics"].copy()

    rows = []
    industry_counts = {industry: 0 for industry in VALID_INDUSTRIES}
    for tile in runtime.district_tiles:
        for side_name, face in (("a", tile.side_a), ("b", tile.side_b)):
            bonus_map = {bonus.slot_index: bonus.bonus_type for bonus in face.bonuses}
            evs = []
            for slot_idx in range(runtime.lots_per_frame):
                slot = {
                    "affinities": face.affinities,
                    "bonus_type": bonus_map.get(slot_idx),
                    "frame_features": (),
                    "row_modifier": 0.0,
                    "frame_compatible": False,
                }
                evs.append(compute_slot_expected_value(slot, industry_stats, module))
            for industry in face.affinities:
                industry_counts[industry] += 1
            compatible_rows = []
            compatible_names: list[str] = []
            for player_count, frame_metrics in frame_metrics_by_player_count.items():
                active_names = {frame.name for frame in runtime.city_frames[player_count]}
                for _, frame_row in frame_metrics.iterrows():
                    frame_name = str(frame_row["frame_name"])
                    if frame_name == "Plaza Central" or frame_name not in active_names:
                        continue
                    prefs = set(runtime.frame_affinity_preference.get(frame_name, ()))
                    if prefs and not (prefs & set(face.affinities)):
                        continue
                    compatible_rows.append(frame_row.to_dict())
                    compatible_names.append(frame_name)

            compatible_frame_mean_closeness = 0.0
            compatible_frame_mean_betweenness = 0.0
            compatible_frame_min_distance = None
            if compatible_rows:
                compatible_df = pd.DataFrame(compatible_rows)
                compatible_frame_mean_closeness = round(float(compatible_df["frame_closeness"].mean()), 4)
                compatible_frame_mean_betweenness = round(float(compatible_df["frame_betweenness"].mean()), 4)
                compatible_frame_min_distance = round(float(compatible_df["frame_distance_to_plaza"].min()), 4)
            rows.append(
                {
                    "tile_id": tile.tile_id,
                    "side": side_name,
                    "face_name": face.name,
                    "affinity_a": face.affinities[0],
                    "affinity_b": face.affinities[1],
                    "base_ev": round(sum(evs) / len(evs), 4),
                    "bonus_count": len(face.bonuses),
                    "has_tech": "Tech" in face.affinities,
                    "has_real_estate": "Real Estate" in face.affinities,
                    "compatible_frame_count": len(compatible_names),
                    "compatible_frame_mean_closeness": compatible_frame_mean_closeness,
                    "compatible_frame_mean_betweenness": compatible_frame_mean_betweenness,
                    "compatible_frame_min_distance_to_plaza": compatible_frame_min_distance,
                    "compatible_frame_names": ", ".join(sorted(set(compatible_names))),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["industry_count_min"] = df.apply(
            lambda row: min(industry_counts[row["affinity_a"]], industry_counts[row["affinity_b"]]),
            axis=1,
        )
    return df.sort_values(["base_ev", "tile_id", "side"]).reset_index(drop=True)


def _first_empty_slot(face) -> int | None:
    occupied = {bonus.slot_index for bonus in face.bonuses}
    for slot_idx in range(4):
        if slot_idx not in occupied:
            return slot_idx
    return None


def _industry_replacement(face_affinities: tuple[str, str], industry_counts: dict[str, int]) -> str | None:
    options = [
        industry
        for industry in VALID_INDUSTRIES
        if industry not in face_affinities
    ]
    if not options:
        return None
    options.sort(key=lambda industry: (industry_counts.get(industry, 0), industry))
    return options[0]


def _generate_candidates(
    runtime: RuntimeBoardConfig,
    diagnostics: pd.DataFrame,
    current_overall: float,
    dynamic_signals: dict[str, Any],
    config: BoardSearchConfig,
) -> list[tuple[BoardCandidateChange, RuntimeBoardConfig]]:
    candidates: list[tuple[BoardCandidateChange, RuntimeBoardConfig]] = []
    weak_frame = dynamic_signals.get("weakest_zone")
    weak_zone_industries = [
        industry
        for industry in dynamic_signals.get("weak_zone_industries", [])
        if industry in VALID_INDUSTRIES
    ]

    for feature, delta in [("riverfront", -0.05), ("plaza_adjacent", -0.05), ("bridge_adjacent", -0.05), ("market_street", -0.05), ("dock_access", -0.05), ("outer_ring", 0.05)]:
        if feature not in runtime.spatial_feature_values:
            continue
        updated = copy.deepcopy(runtime)
        old_value = updated.spatial_feature_values[feature]
        new_value = round(old_value + delta, 2)
        updated.spatial_feature_values[feature] = new_value
        candidates.append(
            (
                BoardCandidateChange(
                    description=f"Adjust spatial feature {feature} {old_value:.2f} -> {new_value:.2f}",
                    field_path=f"SPATIAL_FEATURE_VALUES.{feature}",
                    old_value=old_value,
                    new_value=new_value,
                    change_type="spatial",
                    before_overall_balance=current_overall,
                ),
                updated,
            )
        )

    for delta in (-0.05, 0.05):
        updated = copy.deepcopy(runtime)
        old_value = updated.frame_compatibility_bonus
        new_value = round(old_value + delta, 2)
        updated.frame_compatibility_bonus = new_value
        candidates.append(
            (
                BoardCandidateChange(
                    description=f"Adjust frame compatibility bonus {old_value:.2f} -> {new_value:.2f}",
                    field_path="FRAME_COMPATIBILITY_BONUS",
                    old_value=old_value,
                    new_value=new_value,
                    change_type="compatibility",
                    before_overall_balance=current_overall,
                ),
                updated,
            )
        )

    if config.allow_global_fee_changes:
        updated = copy.deepcopy(runtime)
        old_value = updated.location_fee
        updated.location_fee = round(old_value + 0.5, 2)
        candidates.append(
            (
                BoardCandidateChange(
                    description=f"Raise city location fee {old_value:.2f} -> {updated.location_fee:.2f}",
                    field_path="LOCATION_FEE",
                    old_value=old_value,
                    new_value=updated.location_fee,
                    change_type="location_fee",
                    before_overall_balance=current_overall,
                ),
                updated,
            )
        )

    industry_counts = {industry: 0 for industry in VALID_INDUSTRIES}
    for _, row in diagnostics.iterrows():
        industry_counts[row["affinity_a"]] += 1
        industry_counts[row["affinity_b"]] += 1

    low_faces = diagnostics.nsmallest(3, "base_ev")
    high_faces = diagnostics.nlargest(2, "base_ev")

    for _, row in low_faces.iterrows():
        tile = copy.deepcopy(runtime.district_tiles[int(row["tile_id"]) - 1])
        face = tile.side_a if row["side"] == "a" else tile.side_b
        empty_slot = _first_empty_slot(face)
        if empty_slot is not None:
            updated = copy.deepcopy(runtime)
            target_tile = updated.district_tiles[int(row["tile_id"]) - 1]
            target_face = target_tile.side_a if row["side"] == "a" else target_tile.side_b
            target_face.bonuses.append(SlotBonus(empty_slot, "Trafico"))
            target_face.bonuses.sort(key=lambda bonus: bonus.slot_index)
            candidates.append(
                (
                    BoardCandidateChange(
                        description=f"Add Trafico to weak face {row['face_name']} slot {empty_slot}",
                        field_path=f"DISTRICT_TILES[{int(row['tile_id']) - 1}].{row['side']}.bonuses",
                        old_value=int(row["bonus_count"]),
                        new_value=int(row["bonus_count"]) + 1,
                        change_type="face_bonus",
                        before_overall_balance=current_overall,
                    ),
                    updated,
                )
            )

        if row["has_tech"]:
            replacement = _industry_replacement((row["affinity_a"], row["affinity_b"]), industry_counts)
            if replacement:
                updated = copy.deepcopy(runtime)
                target_face = updated.district_tiles[int(row["tile_id"]) - 1].side_a if row["side"] == "a" else updated.district_tiles[int(row["tile_id"]) - 1].side_b
                new_affinities = tuple(replacement if aff == "Tech" else aff for aff in target_face.affinities)
                if len(set(new_affinities)) == 2:
                    old_affinities = target_face.affinities
                    target_face.affinities = new_affinities
                    candidates.append(
                        (
                            BoardCandidateChange(
                                description=f"Replace Tech on {row['face_name']} with {replacement}",
                                field_path=f"DISTRICT_TILES[{int(row['tile_id']) - 1}].{row['side']}.affinities",
                                old_value=old_affinities,
                                new_value=new_affinities,
                                change_type="face_affinity",
                                before_overall_balance=current_overall,
                            ),
                            updated,
                        )
                    )

    for _, row in high_faces.iterrows():
        if not row["has_real_estate"]:
            continue
        replacement = _industry_replacement((row["affinity_a"], row["affinity_b"]), industry_counts)
        if not replacement or replacement == "Tech":
            continue
        updated = copy.deepcopy(runtime)
        target_face = updated.district_tiles[int(row["tile_id"]) - 1].side_a if row["side"] == "a" else updated.district_tiles[int(row["tile_id"]) - 1].side_b
        new_affinities = tuple(replacement if aff == "Real Estate" else aff for aff in target_face.affinities)
        if len(set(new_affinities)) == 2:
            old_affinities = target_face.affinities
            target_face.affinities = new_affinities
            candidates.append(
                (
                    BoardCandidateChange(
                        description=f"Replace Real Estate on {row['face_name']} with {replacement}",
                        field_path=f"DISTRICT_TILES[{int(row['tile_id']) - 1}].{row['side']}.affinities",
                        old_value=old_affinities,
                        new_value=new_affinities,
                        change_type="face_affinity",
                        before_overall_balance=current_overall,
                    ),
                    updated,
                    )
            )

    if weak_frame:
        current_pref = tuple(runtime.frame_affinity_preference.get(weak_frame, ()))
        if weak_zone_industries:
            proposed_pref = tuple(dict.fromkeys((*weak_zone_industries, *current_pref)))[:2]
            if len(proposed_pref) == 2 and proposed_pref != current_pref:
                updated = copy.deepcopy(runtime)
                old_pref = tuple(updated.frame_affinity_preference.get(weak_frame, ()))
                updated.frame_affinity_preference[weak_frame] = proposed_pref
                candidates.append(
                    (
                        BoardCandidateChange(
                            description=f"Retarget frame preference for {weak_frame} to {proposed_pref[0]} + {proposed_pref[1]}",
                            field_path=f"FRAME_AFFINITY_PREFERENCE.{weak_frame}",
                            old_value=old_pref,
                            new_value=proposed_pref,
                            change_type="frame_affinity_preference",
                            before_overall_balance=current_overall,
                        ),
                        updated,
                    )
                )

        for added_feature in ("avenue_access", "market_street"):
            frame_has_feature = any(
                added_feature in frame.features
                for frames in runtime.city_frames.values()
                for frame in frames
                if frame.name == weak_frame
            )
            if frame_has_feature:
                continue
            updated = copy.deepcopy(runtime)
            old_features = None
            for frames in updated.city_frames.values():
                for idx, frame in enumerate(frames):
                    if frame.name != weak_frame:
                        continue
                    if old_features is None:
                        old_features = tuple(frame.features)
                    frames[idx] = type(frame)(frame.name, tuple((*frame.features, added_feature)))
            if old_features is not None:
                candidates.append(
                    (
                        BoardCandidateChange(
                            description=f"Add {added_feature} to weak frame {weak_frame}",
                            field_path=f"CITY_FRAMES.*.{weak_frame}.features",
                            old_value=old_features,
                            new_value=tuple((*old_features, added_feature)),
                            change_type="frame_feature",
                            before_overall_balance=current_overall,
                        ),
                        updated,
                    )
                )

        if weak_frame in runtime.frame_affinity_preference:
            updated = copy.deepcopy(runtime)
            old_value = updated.frame_compatibility_bonus
            updated.frame_compatibility_bonus = round(old_value + 0.05, 2)
            candidates.append(
                (
                    BoardCandidateChange(
                        description=f"Boost compatibility pressure for weak frame targeting ({weak_frame})",
                        field_path="FRAME_COMPATIBILITY_BONUS",
                        old_value=old_value,
                        new_value=updated.frame_compatibility_bonus,
                        change_type="compatibility",
                        before_overall_balance=current_overall,
                    ),
                    updated,
                )
            )

        weak_pref_set = set(runtime.frame_affinity_preference.get(weak_frame, ())) | set(weak_zone_industries)
        for _, row in low_faces.iterrows():
            face_affinities = {row["affinity_a"], row["affinity_b"]}
            if weak_pref_set and not (face_affinities & weak_pref_set):
                continue
            tile_index = int(row["tile_id"]) - 1
            updated = copy.deepcopy(runtime)
            target_face = updated.district_tiles[tile_index].side_a if row["side"] == "a" else updated.district_tiles[tile_index].side_b
            empty_slot = _first_empty_slot(target_face)
            if empty_slot is None:
                continue
            target_face.bonuses.append(SlotBonus(empty_slot, "Descuento"))
            target_face.bonuses.sort(key=lambda bonus: bonus.slot_index)
            candidates.append(
                (
                    BoardCandidateChange(
                        description=f"Add Descuento to weak-frame-compatible face {row['face_name']} slot {empty_slot}",
                        field_path=f"DISTRICT_TILES[{tile_index}].{row['side']}.bonuses",
                        old_value=int(row["bonus_count"]),
                        new_value=int(row["bonus_count"]) + 1,
                        change_type="frame_targeted_face_bonus",
                        before_overall_balance=current_overall,
                    ),
                    updated,
                )
            )

    unique: dict[tuple[str, str], tuple[BoardCandidateChange, RuntimeBoardConfig]] = {}
    for candidate, candidate_runtime in candidates:
        unique[(candidate.field_path, str(candidate.new_value))] = (candidate, candidate_runtime)
    return list(unique.values())


def optimize_board(runtime: RuntimeBoardConfig | None = None, config: BoardSearchConfig | None = None) -> BoardOptimizationResult:
    runtime = copy.deepcopy(runtime or build_runtime_config())
    config = config or BoardSearchConfig()

    baseline_metrics = _evaluate_runtime(runtime, config.baseline_boards, config.player_counts, config.base_seed)
    baseline_dynamic_metrics = _dynamic_balance_report(runtime, config, config.dynamic_verification_simulations, _stable_seed(config.base_seed, "dynamic", "baseline"))
    current_runtime = copy.deepcopy(runtime)
    current_metrics = baseline_metrics
    current_dynamic_metrics = baseline_dynamic_metrics
    accepted_changes: list[BoardCandidateChange] = []
    search_rows: list[dict[str, Any]] = []

    while len(accepted_changes) < config.max_accepted_changes:
        diagnostics = _face_diagnostics(current_runtime)
        candidates = _generate_candidates(
            current_runtime,
            diagnostics,
            current_metrics["grand_mean"]["mean_overall_balance"],
            current_dynamic_metrics,
            config,
        )
        if not candidates:
            break

        ranked: list[tuple[BoardCandidateChange, RuntimeBoardConfig, dict[str, Any]]] = []
        for candidate, candidate_runtime in candidates:
            estimated = _evaluate_runtime(candidate_runtime, config.search_boards, config.player_counts, config.base_seed + len(search_rows) + 1)
            estimated_dynamic = _dynamic_balance_report(
                candidate_runtime,
                config,
                config.dynamic_search_simulations,
                _stable_seed(config.base_seed, "dynamic-estimate", len(search_rows), candidate.field_path, candidate.new_value),
            )
            candidate.estimated_after_overall_balance = estimated["grand_mean"]["mean_overall_balance"]
            candidate.estimated_improvement = round(
                candidate.estimated_after_overall_balance - current_metrics["grand_mean"]["mean_overall_balance"],
                4,
            )
            candidate.estimated_zone_gap = estimated_dynamic["zone_gap"]
            candidate.estimated_weak_zone = estimated_dynamic["weakest_zone"]
            candidate.estimated_weak_zone_win_rate = estimated_dynamic["weakest_zone_win_rate"]
            candidate.estimated_graph_disparity = float(
                estimated.get("graph_grand_mean", {}).get("frame_betweenness_gap", 0.0)
            )
            candidate.estimated_overcentralized_frame = estimated.get("graph_grand_mean", {}).get(
                "most_common_overcentralized_frame"
            )
            ranked.append((candidate, candidate_runtime, estimated))
            search_rows.append(
                {
                    "stage": "estimate",
                    "description": candidate.description,
                    "field_path": candidate.field_path,
                    "old_value": candidate.old_value,
                    "new_value": candidate.new_value,
                    "before_overall_balance": current_metrics["grand_mean"]["mean_overall_balance"],
                    "estimated_after_overall_balance": candidate.estimated_after_overall_balance,
                    "estimated_improvement": candidate.estimated_improvement,
                    "before_zone_gap": current_dynamic_metrics["zone_gap"],
                    "estimated_zone_gap": candidate.estimated_zone_gap,
                    "estimated_weak_zone": candidate.estimated_weak_zone,
                    "estimated_weak_zone_win_rate": candidate.estimated_weak_zone_win_rate,
                    "estimated_graph_disparity": candidate.estimated_graph_disparity,
                    "estimated_overcentralized_frame": candidate.estimated_overcentralized_frame,
                    "confirmed_after_overall_balance": None,
                    "confirmed_improvement": None,
                    "confirmed_zone_gap": None,
                    "confirmed_weak_zone": None,
                    "confirmed_weak_zone_win_rate": None,
                    "confirmed_graph_disparity": None,
                    "confirmed_overcentralized_frame": None,
                    "dynamic_gate_passed": None,
                    "accepted": False,
                }
            )

        ranked = [row for row in ranked if row[0].estimated_improvement and row[0].estimated_improvement > 0]
        if not ranked:
            break
        ranked.sort(
            key=lambda item: (
                -item[0].estimated_after_overall_balance,
                abs(item[2]["grand_mean"]["mean_centro_vs_barrio"] - 2.0),
                item[0].estimated_graph_disparity if item[0].estimated_graph_disparity is not None else float("inf"),
                item[0].estimated_zone_gap if item[0].estimated_zone_gap is not None else float("inf"),
                -(item[0].estimated_weak_zone_win_rate or 0.0),
            )
        )

        accepted_this_round = False
        for candidate, candidate_runtime, _ in ranked[:6]:
            confirmed = _evaluate_runtime(candidate_runtime, config.baseline_boards, config.player_counts, config.base_seed + 10_000 + len(accepted_changes))
            confirmed_dynamic = _dynamic_balance_report(
                candidate_runtime,
                config,
                config.dynamic_verification_simulations,
                _stable_seed(config.base_seed, "dynamic-confirmation", len(accepted_changes), candidate.field_path, candidate.new_value),
            )
            candidate.confirmed_after_overall_balance = confirmed["grand_mean"]["mean_overall_balance"]
            candidate.confirmed_improvement = round(
                candidate.confirmed_after_overall_balance - current_metrics["grand_mean"]["mean_overall_balance"],
                4,
            )
            candidate.confirmed_zone_gap = confirmed_dynamic["zone_gap"]
            candidate.confirmed_weak_zone = confirmed_dynamic["weakest_zone"]
            candidate.confirmed_weak_zone_win_rate = confirmed_dynamic["weakest_zone_win_rate"]
            candidate.confirmed_graph_disparity = float(
                confirmed.get("graph_grand_mean", {}).get("frame_betweenness_gap", 0.0)
            )
            candidate.confirmed_overcentralized_frame = confirmed.get("graph_grand_mean", {}).get(
                "most_common_overcentralized_frame"
            )
            candidate.dynamic_gate_passed = candidate.confirmed_zone_gap <= (
                current_dynamic_metrics["zone_gap"] + config.zone_gap_tolerance
            )
            candidate.confirmation_passed = (
                candidate.confirmed_improvement >= config.minimum_improvement
                and candidate.dynamic_gate_passed
            )
            search_rows.append(
                {
                    "stage": "confirmation",
                    "description": candidate.description,
                    "field_path": candidate.field_path,
                    "old_value": candidate.old_value,
                    "new_value": candidate.new_value,
                    "before_overall_balance": current_metrics["grand_mean"]["mean_overall_balance"],
                    "estimated_after_overall_balance": candidate.estimated_after_overall_balance,
                    "estimated_improvement": candidate.estimated_improvement,
                    "before_zone_gap": current_dynamic_metrics["zone_gap"],
                    "estimated_zone_gap": candidate.estimated_zone_gap,
                    "estimated_weak_zone": candidate.estimated_weak_zone,
                    "estimated_weak_zone_win_rate": candidate.estimated_weak_zone_win_rate,
                    "estimated_graph_disparity": candidate.estimated_graph_disparity,
                    "estimated_overcentralized_frame": candidate.estimated_overcentralized_frame,
                    "confirmed_after_overall_balance": candidate.confirmed_after_overall_balance,
                    "confirmed_improvement": candidate.confirmed_improvement,
                    "confirmed_zone_gap": candidate.confirmed_zone_gap,
                    "confirmed_weak_zone": candidate.confirmed_weak_zone,
                    "confirmed_weak_zone_win_rate": candidate.confirmed_weak_zone_win_rate,
                    "confirmed_graph_disparity": candidate.confirmed_graph_disparity,
                    "confirmed_overcentralized_frame": candidate.confirmed_overcentralized_frame,
                    "dynamic_gate_passed": candidate.dynamic_gate_passed,
                    "accepted": candidate.confirmation_passed,
                }
            )
            if not candidate.confirmation_passed:
                continue
            current_runtime = candidate_runtime
            current_metrics = confirmed
            current_dynamic_metrics = confirmed_dynamic
            accepted_changes.append(candidate)
            accepted_this_round = True
            break

        if not accepted_this_round:
            break

    final_metrics = _evaluate_runtime(
        current_runtime,
        config.final_verification_boards,
        config.player_counts,
        config.base_seed + 99_999,
    )
    final_dynamic_metrics = _dynamic_balance_report(
        current_runtime,
        config,
        config.dynamic_verification_simulations,
        _stable_seed(config.base_seed, "dynamic", "final"),
    )
    return BoardOptimizationResult(
        config=config,
        baseline_metrics=baseline_metrics,
        optimized_metrics=current_metrics,
        final_metrics=final_metrics,
        baseline_face_diagnostics=_face_diagnostics(runtime),
        final_face_diagnostics=_face_diagnostics(current_runtime),
        baseline_dynamic_metrics=baseline_dynamic_metrics,
        optimized_dynamic_metrics=current_dynamic_metrics,
        final_dynamic_metrics=final_dynamic_metrics,
        accepted_changes=accepted_changes,
        search_trace=pd.DataFrame(search_rows),
        optimized_runtime=current_runtime,
    )


def _runtime_snapshot(runtime: RuntimeBoardConfig) -> dict[str, Any]:
    payload = asdict(runtime)
    payload["plaza_variants"] = [
        {"name": face.name, "affinities": list(face.affinities), "bonuses": [asdict(bonus) for bonus in face.bonuses]}
        for face in runtime.plaza_variants
    ]
    payload["district_tiles"] = [
        {
            "tile_id": tile.tile_id,
            "side_a": {
                "name": tile.side_a.name,
                "affinities": list(tile.side_a.affinities),
                "bonuses": [asdict(bonus) for bonus in tile.side_a.bonuses],
            },
            "side_b": {
                "name": tile.side_b.name,
                "affinities": list(tile.side_b.affinities),
                "bonuses": [asdict(bonus) for bonus in tile.side_b.bonuses],
            },
        }
        for tile in runtime.district_tiles
    ]
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def export_board_recommendations(result: BoardOptimizationResult, out_dir: str | Path = OUTPUT_DIR) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    accepted_rows = pd.DataFrame([asdict(change) for change in result.accepted_changes])
    if accepted_rows.empty:
        accepted_rows = pd.DataFrame(
            columns=[
                "description",
                "field_path",
                "old_value",
                "new_value",
                "change_type",
                "before_overall_balance",
                "estimated_after_overall_balance",
                "confirmed_after_overall_balance",
                "estimated_improvement",
                "confirmed_improvement",
                "estimated_graph_disparity",
                "confirmed_graph_disparity",
                "estimated_overcentralized_frame",
                "confirmed_overcentralized_frame",
                "confirmation_passed",
            ]
        )

    accepted_rows.to_csv(out_path / "board_recommendations.csv", index=False)
    result.search_trace.to_csv(out_path / "board_search_trace.csv", index=False)
    result.baseline_face_diagnostics.to_csv(out_path / "board_face_diagnostics_before.csv", index=False)
    result.final_face_diagnostics.to_csv(out_path / "board_face_diagnostics_after.csv", index=False)

    representative_player_count = max(result.config.player_counts)
    representative_board = generate_board_from_runtime(
        result.optimized_runtime,
        representative_player_count,
        _stable_seed(result.config.base_seed, "graph-export", representative_player_count),
    )
    representative_module = _runtime_module(result.optimized_runtime)
    physical = load_physical_businesses()
    industry_stats = _industry_stats(physical)
    representative_diagnostics = build_board_diagnostics(
        representative_board,
        industry_stats,
        representative_module,
    )
    representative_diagnostics["slot_metrics"].to_csv(out_path / "board_slot_graph_metrics.csv", index=False)
    representative_diagnostics["frame_metrics"].to_csv(out_path / "board_frame_graph_metrics.csv", index=False)
    (out_path / "board_graph_report.json").write_text(
        json.dumps(
            {
                "graph_report": representative_diagnostics["graph_report"],
                "graph_summary": representative_diagnostics["graph_summary"],
                "representative_player_count": representative_player_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "metric_direction": {
            "overall_balance": "higher_is_better",
            "slot_cv": "lower_is_better",
            "zone_parity": "lower_is_better",
            "industry_coverage": "higher_is_better",
            "affinity_balance": "lower_is_better",
            "centro_vs_barrio": "target_is_about_2.0",
            "zone_gap": "lower_is_better",
        },
        "graph_metric_direction": {
            "frame_betweenness_gap": "lower_is_better",
            "frame_closeness_std": "lower_is_better",
            "slot_distance_to_plaza_mean": "descriptive_only",
            "slot_betweenness_top_share": "lower_is_better",
        },
        "config": asdict(result.config),
        "baseline": result.baseline_metrics,
        "optimized": result.optimized_metrics,
        "final_verification": result.final_metrics,
        "baseline_graph": result.baseline_metrics.get("graph_grand_mean", {}),
        "optimized_graph": result.optimized_metrics.get("graph_grand_mean", {}),
        "final_graph": result.final_metrics.get("graph_grand_mean", {}),
        "notes": [
            "Graph metrics are structural diagnostics for the board topology.",
            "Monte Carlo and the fixed evaluator remain the balance judge.",
            "Graph disparities can inform ranking and investigation but do not directly encode win rate.",
        ],
        "baseline_dynamic": _json_ready(result.baseline_dynamic_metrics),
        "optimized_dynamic": _json_ready(result.optimized_dynamic_metrics),
        "final_dynamic": _json_ready(result.final_dynamic_metrics),
        "accepted_changes": accepted_rows.to_dict(orient="records"),
        "optimized_runtime": _runtime_snapshot(result.optimized_runtime),
    }
    (out_path / "board_balance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    result = optimize_board()
    export_board_recommendations(result)
    print("Board autobalance complete.")
    print(f"Baseline overall: {result.baseline_metrics['grand_mean']['mean_overall_balance']:.4f}")
    print(f"Final overall:    {result.final_metrics['grand_mean']['mean_overall_balance']:.4f}")
    print(f"Accepted changes: {len(result.accepted_changes)}")
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
