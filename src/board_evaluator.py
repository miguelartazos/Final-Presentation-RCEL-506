"""
board_evaluator.py — Fixed evaluator for board layout autoresearch
===================================================================
Computes board balance metrics from the card pool and the current
board_config settings.

Usage:
    .venv/bin/python3 board_evaluator.py
    .venv/bin/python3 board_evaluator.py --boards 200 --players 4
    .venv/bin/python3 board_evaluator.py --heatmap
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from board_graph import compute_graph_metrics, summarize_graph_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "data" / "cards_dataset.csv"
LEGACY_CSV_PATH = SCRIPT_DIR / "cards_dataset.csv"

ALL_INDUSTRIES = frozenset(
    ["Service", "Food", "Retail", "Professional", "Tech", "Real Estate", "Trades"]
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _stable_seed(*parts: object) -> int:
    """Create a deterministic integer seed from a sequence of values."""
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Card pool loading
# ---------------------------------------------------------------------------

def load_physical_businesses(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load physical business cards from the dataset CSV."""
    if not csv_path.exists() and csv_path == CSV_PATH and LEGACY_CSV_PATH.exists():
        csv_path = LEGACY_CSV_PATH
    df = pd.read_csv(csv_path)
    biz = df[df["type"].str.strip().str.title() == "Business"].copy()
    physical = biz[biz["mode"].str.strip() == "Physical"].copy()
    return physical


def _industry_stats(physical_biz: pd.DataFrame) -> dict[str, dict]:
    """Precompute per-industry statistics for the physical card pool."""
    total = len(physical_biz)
    stats: dict[str, dict] = {}
    for industry in ALL_INDUSTRIES:
        matching = physical_biz[physical_biz["industry"] == industry]
        count = len(matching)
        avg_income = float(matching["avg_income"].mean()) if count > 0 else 0.0
        stats[industry] = {
            "count": count,
            "weight": count / total if total > 0 else 0.0,
            "avg_income": avg_income,
        }
    return stats


# ---------------------------------------------------------------------------
# Slot expected value
# ---------------------------------------------------------------------------

def compute_slot_expected_value(
    slot: dict,
    industry_stats: dict[str, dict],
    config_module,
) -> float:
    """
    Compute the expected value of a board slot.

    EV(slot) = printed_bonus + affinity_value + spatial_value + row_modifier

    Printed bonus:  value depends on bonus type and config constants.
    Affinity value:  for each of the slot's 2 industry affinities,
        (count_matching / total_physical) * (avg_income_of_industry + 1.0)
        The +1.0 represents the affinity IR bonus itself.
    Spatial value:  sum of SPATIAL_FEATURE_VALUES for each feature
        inherited from the slot's city frame position (riverfront,
        plaza_adjacent, etc.).
    Row modifier:   Fachada (row 0) gets a slight premium, Interior
        (row 1) gets a slight discount.
    """
    # Printed bonus value
    bt = slot.get("bonus_type")
    if bt == "Trafico":
        bonus_val = config_module.TRAFFIC_BONUS_IR
    elif bt == "Prestigio":
        bonus_val = config_module.PRESTIGE_BONUS_MARCA
    elif bt == "Descuento":
        bonus_val = config_module.DISCOUNT_BONUS_K
    else:
        bonus_val = 0.0

    # Industry affinity value
    affinity_val = 0.0
    for industry in slot.get("affinities", ()):
        ist = industry_stats.get(industry)
        if ist and ist["count"] > 0:
            affinity_val += ist["weight"] * (ist["avg_income"] + 1.0)

    # Spatial feature value (from the fixed city skeleton)
    spatial_val = 0.0
    feature_values = getattr(config_module, "SPATIAL_FEATURE_VALUES", {})
    for feature in slot.get("frame_features", ()):
        spatial_val += feature_values.get(feature, 0.0)

    # Row modifier (Fachada vs Interior)
    row_mod = slot.get("row_modifier", 0.0)

    # Frame-to-insert compatibility bonus
    compat_bonus = 0.0
    if slot.get("frame_compatible", False):
        compat_bonus = getattr(config_module, "FRAME_COMPATIBILITY_BONUS", 0.0)

    return bonus_val + affinity_val + spatial_val + row_mod + compat_bonus


def compute_barrio_slot_ev(config_module) -> float:
    """
    EV for a Barrio slot.

    Barrio saves LOCATION_FEE on placement (one-time). Amortized over
    the game, this is worth roughly half the fee.
    """
    return config_module.LOCATION_FEE * 0.5


def build_board_diagnostics(
    board: dict,
    industry_stats: dict[str, dict],
    config_module,
) -> dict[str, pd.DataFrame | dict]:
    """Build slot/frame diagnostics with EV and graph metrics for one board."""
    graph_metrics = compute_graph_metrics(board)
    slot_metrics = graph_metrics["slot_metrics"].copy()
    slot_lookup = {
        f"{slot['frame_name']}::L{slot['slot_idx']}": slot
        for slot in board["slots"]
    }
    slot_metrics["slot_ev"] = slot_metrics["slot_id"].map(
        lambda slot_id: compute_slot_expected_value(
            slot_lookup[slot_id],
            industry_stats,
            config_module,
        )
    )

    frame_ev = (
        slot_metrics.groupby("frame_name", as_index=False)["slot_ev"]
        .mean()
        .rename(columns={"slot_ev": "frame_mean_slot_ev"})
    )
    frame_metrics = graph_metrics["frame_metrics"].copy().merge(frame_ev, on="frame_name", how="left")

    return {
        "slot_metrics": slot_metrics,
        "frame_metrics": frame_metrics,
        "graph_report": graph_metrics["graph_report"],
        "graph_summary": summarize_graph_metrics(graph_metrics),
    }


def _aggregate_graph_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {}

    numeric_keys = [
        key
        for key in summaries[0].keys()
        if isinstance(summaries[0][key], (int, float))
    ]
    aggregated = {
        key: float(np.mean([float(summary[key]) for summary in summaries]))
        for key in numeric_keys
    }

    frame_counts: dict[str, int] = {}
    for summary in summaries:
        frame_name = summary.get("overcentralized_frame")
        if not frame_name:
            continue
        frame_counts[str(frame_name)] = frame_counts.get(str(frame_name), 0) + 1
    aggregated["most_common_overcentralized_frame"] = (
        sorted(frame_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if frame_counts
        else None
    )
    return aggregated


# ---------------------------------------------------------------------------
# Balance metrics
# ---------------------------------------------------------------------------

@dataclass
class BoardMetrics:
    """Balance metrics for a single board configuration."""
    slot_cv: float              # CV of slot EVs (lower = more balanced)
    zone_parity: float          # std of mean EV across tiles
    industry_coverage: float    # fraction of 7 industries with >= 1 affinity tile
    affinity_balance: float     # CV of industry representation across tiles
    centro_vs_barrio: float     # ratio of avg ciudad EV to barrio EV
    overall_balance: float      # weighted composite (higher = better)


def evaluate_board(
    board: dict,
    industry_stats: dict[str, dict],
    config_module,
) -> BoardMetrics:
    """Evaluate a single board configuration."""

    # 1. Compute EV for every ciudad slot
    slot_evs = np.array([
        compute_slot_expected_value(s, industry_stats, config_module)
        for s in board["slots"]
    ])

    mean_ev = float(np.mean(slot_evs)) if len(slot_evs) > 0 else 0.0

    # 2. slot_cv: coefficient of variation
    slot_cv = float(np.std(slot_evs) / mean_ev) if mean_ev > 0 else 0.0

    # 3. zone_parity: std of mean EV per frame/zone
    # Group slots by frame_name and compute mean EV per group
    frame_groups: dict[str, list[float]] = {}
    for i, s in enumerate(board["slots"]):
        fname = s.get("frame_name", f"unknown_{i}")
        frame_groups.setdefault(fname, []).append(float(slot_evs[i]))
    tile_means = [float(np.mean(evs)) for evs in frame_groups.values()]
    zone_parity = float(np.std(tile_means)) if len(tile_means) > 1 else 0.0

    # 4. industry_coverage: fraction of 7 industries with >= 1 affinity tile
    covered: set[str] = set()
    for face in board["tiles"]:
        for aff in face.affinities:
            covered.add(aff)
    industry_coverage = len(covered & ALL_INDUSTRIES) / len(ALL_INDUSTRIES)

    # 5. affinity_balance: CV of industry appearance counts
    industry_counts: dict[str, int] = {}
    for face in board["tiles"]:
        for aff in face.affinities:
            industry_counts[aff] = industry_counts.get(aff, 0) + 1
    if industry_counts:
        counts = np.array(list(industry_counts.values()), dtype=float)
        affinity_balance = (
            float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0
        )
    else:
        affinity_balance = 1.0

    # 6. centro_vs_barrio: ratio of avg ciudad EV to barrio EV
    barrio_ev = compute_barrio_slot_ev(config_module)
    centro_vs_barrio = mean_ev / barrio_ev if barrio_ev > 0 else float("inf")

    # 7. overall_balance: weighted composite (higher = better)
    s_slot = max(0.0, 1.0 - slot_cv / 0.30)
    s_zone = max(0.0, 1.0 - zone_parity / 1.0)
    s_coverage = industry_coverage
    s_affinity = max(0.0, 1.0 - affinity_balance / 0.60)
    s_ratio = max(0.0, 1.0 - abs(centro_vs_barrio - 2.0) / 1.5)

    overall_balance = (
        0.30 * s_slot
        + 0.15 * s_zone
        + 0.20 * s_coverage
        + 0.15 * s_affinity
        + 0.20 * s_ratio
    )

    return BoardMetrics(
        slot_cv=round(slot_cv, 4),
        zone_parity=round(zone_parity, 4),
        industry_coverage=round(industry_coverage, 4),
        affinity_balance=round(affinity_balance, 4),
        centro_vs_barrio=round(centro_vs_barrio, 4),
        overall_balance=round(overall_balance, 4),
    )


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    n_boards: int = 200,
    player_counts: tuple[int, ...] = (2, 3, 4),
    base_seed: int = 20260322,
) -> dict:
    """
    Evaluate board balance over many random configurations.

    For each player count, generate n_boards random boards (different tile
    selections and orientations), compute metrics for each, then aggregate.
    """
    import board_config as config_module

    physical_biz = load_physical_businesses()
    i_stats = _industry_stats(physical_biz)

    all_results: dict[int, dict] = {}
    for pc in player_counts:
        metrics_list: list[BoardMetrics] = []
        graph_summaries: list[dict[str, object]] = []
        for i in range(n_boards):
            seed = _stable_seed(base_seed, pc, i)
            board = config_module.generate_board(pc, seed)
            metrics = evaluate_board(board, i_stats, config_module)
            metrics_list.append(metrics)
            diagnostics = build_board_diagnostics(board, i_stats, config_module)
            graph_summaries.append(diagnostics["graph_summary"])

        all_results[pc] = {
            "mean_slot_cv": float(np.mean([m.slot_cv for m in metrics_list])),
            "mean_zone_parity": float(np.mean([m.zone_parity for m in metrics_list])),
            "mean_industry_coverage": float(
                np.mean([m.industry_coverage for m in metrics_list])
            ),
            "mean_affinity_balance": float(
                np.mean([m.affinity_balance for m in metrics_list])
            ),
            "mean_centro_vs_barrio": float(
                np.mean([m.centro_vs_barrio for m in metrics_list])
            ),
            "mean_overall_balance": float(
                np.mean([m.overall_balance for m in metrics_list])
            ),
            "worst_overall_balance": float(
                min(m.overall_balance for m in metrics_list)
            ),
            "best_overall_balance": float(
                max(m.overall_balance for m in metrics_list)
            ),
            "n_boards": n_boards,
            "metrics_list": metrics_list,
            "graph_summary": _aggregate_graph_summaries(graph_summaries),
        }

    # Grand mean across player counts
    metric_keys = [
        "mean_slot_cv",
        "mean_zone_parity",
        "mean_industry_coverage",
        "mean_affinity_balance",
        "mean_centro_vs_barrio",
        "mean_overall_balance",
    ]
    grand: dict[str, float] = {}
    for key in metric_keys:
        grand[key] = float(np.mean([all_results[pc][key] for pc in player_counts]))

    return {
        "by_player_count": all_results,
        "grand_mean": grand,
        "graph_grand_mean": _aggregate_graph_summaries(
            [all_results[pc]["graph_summary"] for pc in player_counts]
        ),
    }


# ---------------------------------------------------------------------------
# Heatmap visualization
# ---------------------------------------------------------------------------

def generate_heatmap(
    config_module,
    physical_biz: pd.DataFrame,
    industry_stats: dict[str, dict],
    player_count: int = 4,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Path:
    """Generate a heatmap of slot EVs for a single board configuration."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if output_dir is None:
        output_dir = SCRIPT_DIR / "slides"
    output_dir.mkdir(exist_ok=True)

    board = config_module.generate_board(player_count, seed)
    diagnostics = build_board_diagnostics(board, industry_stats, config_module)
    frame_graph = diagnostics["frame_metrics"].set_index("frame_name")
    slot_evs = [
        compute_slot_expected_value(s, industry_stats, config_module)
        for s in board["slots"]
    ]

    # Group slots by frame_name for heatmap panels
    frame_groups: list[tuple[str, list[dict], list[float]]] = []
    current_frame = None
    current_slots: list[dict] = []
    current_evs: list[float] = []
    for i, s in enumerate(board["slots"]):
        fname = s.get("frame_name", "")
        if fname != current_frame:
            if current_frame is not None:
                frame_groups.append((current_frame, current_slots, current_evs))
            current_frame = fname
            current_slots = []
            current_evs = []
        current_slots.append(s)
        current_evs.append(slot_evs[i])
    if current_frame is not None:
        frame_groups.append((current_frame, current_slots, current_evs))

    n_panels = len(frame_groups)
    lots_per_frame = getattr(config_module, "LOTS_PER_FRAME", 4)
    frame_cols = getattr(config_module, "FRAME_COLS", 2)
    frame_rows = getattr(config_module, "FRAME_ROWS", 2)

    fig_width = max(8, n_panels * 3.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 3.5))
    if n_panels == 1:
        axes = [axes]

    vmin = min(slot_evs)
    vmax = max(slot_evs)
    cmap = plt.cm.YlOrRd

    for panel_idx, (fname, slots_in_frame, evs_in_frame) in enumerate(frame_groups):
        ax = axes[panel_idx]

        grid = np.zeros((frame_rows, frame_cols))
        for s_idx, ev in enumerate(evs_in_frame):
            r = s_idx // frame_cols
            c = s_idx % frame_cols
            if r < frame_rows and c < frame_cols:
                grid[r, c] = ev

        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        for s_idx, ev in enumerate(evs_in_frame):
            r = s_idx // frame_cols
            c = s_idx % frame_cols
            if r < frame_rows and c < frame_cols:
                slot = slots_in_frame[s_idx]
                label = f"{ev:.2f}"
                if slot.get("bonus_type"):
                    abbrev = {"Trafico": "T", "Prestigio": "P", "Descuento": "D"}
                    label += f"\n({abbrev.get(slot['bonus_type'], '?')})"
                ax.text(c, r, label, ha="center", va="center", fontsize=8, fontweight="bold")

        tile_name = slots_in_frame[0].get("tile_name", "")
        affinities = slots_in_frame[0].get("affinities", ())
        aff_str = ", ".join(affinities) if affinities else ""
        is_plaza = slots_in_frame[0].get("is_plaza", False)
        graph_bits = ""
        if fname in frame_graph.index:
            frame_row = frame_graph.loc[fname]
            graph_bits = (
                f"\nclose={frame_row['frame_closeness']:.2f} "
                f"| bridge={frame_row['frame_betweenness']:.2f} "
                f"| d={frame_row['frame_distance_to_plaza']:.0f}"
            )
        prefix = "FIXED: " if is_plaza else ""
        title = (
            f"{prefix}{fname}\n{tile_name} | {aff_str}{graph_bits}"
            if tile_name != fname
            else f"{prefix}{fname}\n{aff_str}{graph_bits}"
        )
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Board Slot EV Heatmap — {player_count}p, seed={seed}",
        fontsize=11,
        fontweight="bold",
    )
    fig.colorbar(im, ax=axes, shrink=0.8, label="Expected Value")
    plt.tight_layout()

    out_path = output_dir / "slide_board_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Tile-level analysis
# ---------------------------------------------------------------------------

def print_tile_analysis(config_module, industry_stats: dict[str, dict]) -> None:
    """Print per-tile-face analysis for all 16 faces (base EV without spatial features)."""
    lots_per_frame = getattr(config_module, "LOTS_PER_FRAME", 4)
    print("\n  Tile Face Analysis (all 16 faces, base EV without spatial features):")
    print(f"  {'Face':<25s} {'Affinities':<30s} {'Base EV':>8s} {'Bonuses':>8s}")
    print("  " + "-" * 73)

    for tile in config_module.DISTRICT_TILES:
        for face in [tile.side_a, tile.side_b]:
            evs = []
            bonus_map = {b.slot_index: b.bonus_type for b in face.bonuses}
            for slot_idx in range(lots_per_frame):
                slot = {
                    "affinities": face.affinities,
                    "bonus_type": bonus_map.get(slot_idx),
                    "frame_features": (),  # base EV without spatial features
                    "row_modifier": 0.0,
                    "frame_compatible": False,
                }
                ev = compute_slot_expected_value(slot, industry_stats, config_module)
                evs.append(ev)
            mean_ev = np.mean(evs)
            n_bonuses = len(face.bonuses)
            aff_str = ", ".join(face.affinities)
            print(f"  {face.name:<25s} {aff_str:<30s} {mean_ev:8.3f} {n_bonuses:8d}")

    # Show spatial feature values
    feature_values = getattr(config_module, "SPATIAL_FEATURE_VALUES", {})
    if feature_values:
        print("\n  City Skeleton — Spatial Feature Values:")
        print(f"  {'Feature':<25s} {'Value':>8s}")
        print("  " + "-" * 35)
        for feat, val in sorted(feature_values.items(), key=lambda x: -x[1]):
            print(f"  {feat:<25s} {val:+8.2f}")

    # Show frame positions per player count
    frames_by_pc = getattr(config_module, "CITY_FRAMES", {})
    if frames_by_pc:
        print("\n  City Frames by Player Count:")
        for pc in sorted(frames_by_pc):
            frames = frames_by_pc[pc]
            total_spatial = sum(
                sum(feature_values.get(f, 0.0) for f in frame.features)
                for frame in frames
            )
            avg_spatial = total_spatial / len(frames) if frames else 0.0
            print(f"  {pc}p: {len(frames)} frames, avg spatial bonus: {avg_spatial:+.2f}")
            for frame in frames:
                fv = sum(feature_values.get(f, 0.0) for f in frame.features)
                feats = ", ".join(frame.features)
                print(f"      {frame.name:<22s} [{feats}] → {fv:+.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Board layout balance evaluator for Business Empire"
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=200,
        help="Number of random boards per player count (default: 200)",
    )
    parser.add_argument(
        "--players",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Player counts to evaluate (default: 2 3 4)",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate a heatmap PNG for a sample board",
    )
    parser.add_argument(
        "--heatmap-seed",
        type=int,
        default=42,
        help="Seed for the heatmap board (default: 42)",
    )
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="Print per-tile-face analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import board_config as config_module

    # Run evaluation
    results = run_evaluation(
        n_boards=args.boards,
        player_counts=tuple(args.players),
    )
    grand = results["grand_mean"]

    # Autoresearch output format
    print("---")
    print(f"slot_cv:            {grand['mean_slot_cv']:.4f}")
    print(f"zone_parity:        {grand['mean_zone_parity']:.4f}")
    print(f"industry_coverage:  {grand['mean_industry_coverage']:.4f}")
    print(f"affinity_balance:   {grand['mean_affinity_balance']:.4f}")
    print(f"centro_vs_barrio:   {grand['mean_centro_vs_barrio']:.4f}")
    print(f"overall_balance:    {grand['mean_overall_balance']:.4f}")
    if results.get("graph_grand_mean"):
        graph = results["graph_grand_mean"]
        print(f"graph_betw_gap:    {graph['frame_betweenness_gap']:.4f}")
        print(f"graph_close_std:   {graph['frame_closeness_std']:.4f}")
        print(f"graph_plaza_dist:  {graph['slot_distance_to_plaza_mean']:.4f}")
    print("---")

    # Per-player-count breakdown
    for pc in sorted(results["by_player_count"]):
        data = results["by_player_count"][pc]
        print(
            f"\n  {pc}p: overall={data['mean_overall_balance']:.4f} "
            f"[{data['worst_overall_balance']:.4f} - "
            f"{data['best_overall_balance']:.4f}] "
            f"slot_cv={data['mean_slot_cv']:.4f} "
            f"coverage={data['mean_industry_coverage']:.4f}"
        )
        graph = data.get("graph_summary", {})
        if graph:
            print(
                "      "
                f"graph_gap={graph['frame_betweenness_gap']:.4f} "
                f"graph_close_std={graph['frame_closeness_std']:.4f} "
                f"overcentralized={graph['most_common_overcentralized_frame']}"
            )

    # Tile analysis
    if args.tiles:
        physical_biz = load_physical_businesses()
        i_stats = _industry_stats(physical_biz)
        print_tile_analysis(config_module, i_stats)

    # Heatmap
    if args.heatmap:
        physical_biz = load_physical_businesses()
        i_stats = _industry_stats(physical_biz)
        heatmap_pc = max(args.players)
        out_path = generate_heatmap(
            config_module, physical_biz, i_stats,
            player_count=heatmap_pc, seed=args.heatmap_seed,
        )
        print(f"\n  Heatmap saved to: {out_path}")


if __name__ == "__main__":
    main()
