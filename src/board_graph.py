"""
board_graph.py — structural graph analytics for Ciudad Viva
===========================================================
Transforms a generated board layout into slot- and frame-level graphs,
then computes graph diagnostics used by reports and balance tooling.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


_INTRA_FRAME_SLOT_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))
_HORIZONTAL_BORDER_SLOT_EDGES = ((1, 0), (3, 2))
_VERTICAL_BORDER_SLOT_EDGES = ((2, 0), (3, 1))

# Ordered connector definitions used to map frame links onto slot borders.
# The first frame is interpreted as the left frame for horizontal links and
# the north frame for vertical links.
_FRAME_SLOT_CONNECTORS_BY_PLAYER_COUNT = {
    4: (
        ("Ribera Noroeste", "Plaza Central", "horizontal"),
        ("Plaza Central", "Ribera Noreste", "horizontal"),
        ("Plaza Central", "Barrio Comercial", "vertical"),
        ("Plaza Central", "Darsena Sur", "vertical"),
        ("Ribera Noroeste", "Barrio Comercial", "vertical"),
        ("Ribera Noreste", "Darsena Sur", "vertical"),
        ("Barrio Comercial", "Periferia", "horizontal"),
        ("Periferia", "Darsena Sur", "horizontal"),
    ),
    3: (
        ("Ribera Noroeste", "Plaza Central", "horizontal"),
        ("Plaza Central", "Ribera Noreste", "horizontal"),
        ("Plaza Central", "Barrio Comercial", "vertical"),
        ("Plaza Central", "Darsena Sur", "vertical"),
        ("Ribera Noroeste", "Barrio Comercial", "vertical"),
        ("Ribera Noreste", "Darsena Sur", "vertical"),
    ),
    2: (
        ("Ribera Noroeste", "Plaza Central", "horizontal"),
        ("Plaza Central", "Barrio Comercial", "vertical"),
        ("Plaza Central", "Darsena Sur", "vertical"),
        ("Ribera Noroeste", "Barrio Comercial", "vertical"),
        ("Barrio Comercial", "Darsena Sur", "horizontal"),
    ),
}


def _slot_id(slot: dict[str, Any]) -> str:
    return f"{slot['frame_name']}::L{slot['slot_idx']}"


def _city_slots(layout: dict[str, Any]) -> list[dict[str, Any]]:
    return [slot for slot in layout.get("slots", []) if slot.get("frame_name")]


def _slot_lookup(layout: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for slot in _city_slots(layout):
        lookup.setdefault(str(slot["frame_name"]), {})[int(slot["slot_idx"])] = slot
    return lookup


def _active_frame_order(layout: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for slot in _city_slots(layout):
        frame_name = str(slot["frame_name"])
        if frame_name in seen:
            continue
        ordered.append(frame_name)
        seen.add(frame_name)
    return ordered


def _active_frame_set(layout: dict[str, Any]) -> set[str]:
    return set(_active_frame_order(layout))


def _safe_eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    try:
        return nx.eigenvector_centrality(graph, max_iter=1000)
    except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
        values = {str(node): 0.0 for node in graph.nodes}
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component).copy()
            if len(subgraph) <= 1:
                continue
            try:
                partial = nx.eigenvector_centrality(subgraph, max_iter=1000)
            except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
                continue
            for node, score in partial.items():
                values[str(node)] = float(score)
        return values


def _component_ids(graph: nx.Graph) -> dict[str, int]:
    components = sorted(
        (sorted(str(node) for node in component) for component in nx.connected_components(graph)),
        key=lambda nodes: nodes[0],
    )
    mapping: dict[str, int] = {}
    for component_id, component_nodes in enumerate(components):
        for node in component_nodes:
            mapping[node] = component_id
    return mapping


def _distance_map(graph: nx.Graph, sources: list[str]) -> dict[str, float]:
    if not sources:
        return {str(node): float("inf") for node in graph.nodes}
    lengths = nx.multi_source_dijkstra_path_length(graph, sources)
    return {str(node): float(lengths.get(node, float("inf"))) for node in graph.nodes}


def _canonical_frame_edges(layout: dict[str, Any]) -> list[tuple[str, str]]:
    edges = layout.get("frame_adjacency")
    if edges:
        return [(str(left), str(right)) for left, right in edges]

    player_count = int(layout["player_count"])
    return [
        (left, right)
        for left, right, _ in _FRAME_SLOT_CONNECTORS_BY_PLAYER_COUNT[player_count]
    ]


def _frame_slot_connectors(layout: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return _FRAME_SLOT_CONNECTORS_BY_PLAYER_COUNT[int(layout["player_count"])]


def build_city_slot_graph(layout: dict[str, Any]) -> nx.Graph:
    """Build an undirected slot-level graph for all city slots."""
    graph = nx.Graph()
    slots = _city_slots(layout)
    slots_by_frame = _slot_lookup(layout)
    active_frames = _active_frame_set(layout)

    for slot in slots:
        node_id = _slot_id(slot)
        graph.add_node(
            node_id,
            slot_id=node_id,
            frame_name=str(slot["frame_name"]),
            tile_name=str(slot["tile_name"]),
            slot_idx=int(slot["slot_idx"]),
            row=int(slot["row"]),
            col=int(slot["col"]),
            is_plaza=bool(slot.get("is_plaza", False)),
        )

    for frame_name, frame_slots in slots_by_frame.items():
        for source_idx, target_idx in _INTRA_FRAME_SLOT_EDGES:
            source = frame_slots.get(source_idx)
            target = frame_slots.get(target_idx)
            if source is None or target is None:
                continue
            graph.add_edge(_slot_id(source), _slot_id(target), edge_type="intra_frame", frame_name=frame_name)

    for left_frame, right_frame, orientation in _frame_slot_connectors(layout):
        if left_frame not in active_frames or right_frame not in active_frames:
            continue
        left_slots = slots_by_frame.get(left_frame, {})
        right_slots = slots_by_frame.get(right_frame, {})
        border_pairs = (
            _HORIZONTAL_BORDER_SLOT_EDGES
            if orientation == "horizontal"
            else _VERTICAL_BORDER_SLOT_EDGES
        )
        for left_idx, right_idx in border_pairs:
            left_slot = left_slots.get(left_idx)
            right_slot = right_slots.get(right_idx)
            if left_slot is None or right_slot is None:
                continue
            graph.add_edge(
                _slot_id(left_slot),
                _slot_id(right_slot),
                edge_type="inter_frame",
                orientation=orientation,
                frames=(left_frame, right_frame),
            )

    return graph


def _build_frame_graph(layout: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    active_frames = _active_frame_order(layout)
    active_set = set(active_frames)
    for frame_name in active_frames:
        graph.add_node(frame_name, frame_name=frame_name, is_plaza=(frame_name == "Plaza Central"))

    for left, right in _canonical_frame_edges(layout):
        if left not in active_set or right not in active_set:
            continue
        graph.add_edge(left, right)
    return graph


def compute_graph_metrics(layout: dict[str, Any]) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Compute slot- and frame-level graph metrics for a generated board."""
    slot_graph = build_city_slot_graph(layout)
    frame_graph = _build_frame_graph(layout)
    slot_ids = list(slot_graph.nodes)
    frame_names = list(frame_graph.nodes)

    slot_degree = dict(slot_graph.degree())
    slot_betweenness = nx.betweenness_centrality(slot_graph)
    slot_closeness = nx.closeness_centrality(slot_graph)
    slot_eigenvector = _safe_eigenvector_centrality(slot_graph)
    slot_component = _component_ids(slot_graph)
    plaza_slots = [
        str(node_id)
        for node_id, attrs in slot_graph.nodes(data=True)
        if bool(attrs.get("is_plaza", False))
    ]
    slot_distance = _distance_map(slot_graph, plaza_slots)

    slot_rows = []
    for node_id in slot_ids:
        attrs = slot_graph.nodes[node_id]
        slot_rows.append(
            {
                "slot_id": node_id,
                "frame_name": attrs["frame_name"],
                "tile_name": attrs["tile_name"],
                "slot_idx": attrs["slot_idx"],
                "row": attrs["row"],
                "col": attrs["col"],
                "is_plaza": attrs["is_plaza"],
                "graph_degree": float(slot_degree.get(node_id, 0)),
                "graph_betweenness": float(slot_betweenness.get(node_id, 0.0)),
                "graph_closeness": float(slot_closeness.get(node_id, 0.0)),
                "graph_distance_to_plaza": float(slot_distance.get(node_id, float("inf"))),
                "graph_eigenvector": float(slot_eigenvector.get(node_id, 0.0)),
                "graph_component_id": int(slot_component.get(node_id, -1)),
            }
        )
    slot_metrics = pd.DataFrame(slot_rows).sort_values(["is_plaza", "frame_name", "slot_idx"], ascending=[False, True, True]).reset_index(drop=True)

    frame_degree = dict(frame_graph.degree())
    frame_betweenness = nx.betweenness_centrality(frame_graph)
    frame_closeness = nx.closeness_centrality(frame_graph)
    frame_distance = _distance_map(frame_graph, ["Plaza Central"])

    slot_grouped = (
        slot_metrics.groupby("frame_name", as_index=False)[["graph_betweenness", "graph_closeness"]]
        .mean()
        .rename(
            columns={
                "graph_betweenness": "frame_mean_slot_betweenness",
                "graph_closeness": "frame_mean_slot_closeness",
            }
        )
    )

    frame_rows = []
    for frame_name in frame_names:
        frame_rows.append(
            {
                "frame_name": frame_name,
                "is_plaza": frame_name == "Plaza Central",
                "frame_degree": float(frame_degree.get(frame_name, 0)),
                "frame_betweenness": float(frame_betweenness.get(frame_name, 0.0)),
                "frame_closeness": float(frame_closeness.get(frame_name, 0.0)),
                "frame_distance_to_plaza": float(frame_distance.get(frame_name, float("inf"))),
            }
        )
    frame_metrics = pd.DataFrame(frame_rows).merge(slot_grouped, on="frame_name", how="left")
    frame_metrics["frame_mean_slot_betweenness"] = frame_metrics["frame_mean_slot_betweenness"].fillna(0.0)
    frame_metrics["frame_mean_slot_closeness"] = frame_metrics["frame_mean_slot_closeness"].fillna(0.0)
    frame_metrics = frame_metrics.sort_values(["is_plaza", "frame_distance_to_plaza", "frame_name"], ascending=[False, True, True]).reset_index(drop=True)

    top_slots = (
        slot_metrics.sort_values(["graph_betweenness", "graph_closeness", "slot_id"], ascending=[False, False, True])
        [["slot_id", "frame_name", "graph_betweenness", "graph_closeness", "graph_distance_to_plaza"]]
        .head(5)
    )
    top_frames = (
        frame_metrics.sort_values(["frame_betweenness", "frame_closeness", "frame_name"], ascending=[False, False, True])
        [["frame_name", "frame_betweenness", "frame_closeness", "frame_distance_to_plaza"]]
        .head(5)
    )

    graph_report = {
        "player_count": int(layout["player_count"]),
        "city_slot_count": int(slot_graph.number_of_nodes()),
        "city_slot_edge_count": int(slot_graph.number_of_edges()),
        "frame_count": int(frame_graph.number_of_nodes()),
        "frame_edge_count": int(frame_graph.number_of_edges()),
        "component_count": int(nx.number_connected_components(slot_graph)),
        "plaza_slot_ids": plaza_slots,
        "top_bridge_slots": top_slots.to_dict(orient="records"),
        "top_bridge_frames": top_frames.to_dict(orient="records"),
    }

    return {
        "slot_metrics": slot_metrics,
        "frame_metrics": frame_metrics,
        "graph_report": graph_report,
    }


def annotate_slots_with_graph_metrics(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Return city slots enriched with graph metrics."""
    slot_metrics = compute_graph_metrics(layout)["slot_metrics"].set_index("slot_id")
    annotated: list[dict[str, Any]] = []
    for slot in _city_slots(layout):
        node_id = _slot_id(slot)
        metrics = slot_metrics.loc[node_id].to_dict()
        metrics.pop("frame_name", None)
        metrics.pop("tile_name", None)
        metrics.pop("slot_idx", None)
        metrics.pop("row", None)
        metrics.pop("col", None)
        metrics.pop("is_plaza", None)
        annotated.append({**slot, "slot_id": node_id, **metrics})
    return annotated


def summarize_graph_metrics(graph_metrics: dict[str, pd.DataFrame | dict[str, Any]]) -> dict[str, Any]:
    """Reduce graph metrics to a compact structural summary."""
    slot_metrics = graph_metrics["slot_metrics"]
    frame_metrics = graph_metrics["frame_metrics"]

    overcentralized_frame = None
    if not frame_metrics.empty:
        ranked = frame_metrics.sort_values(
            ["frame_betweenness", "frame_closeness", "frame_name"],
            ascending=[False, False, True],
        )
        overcentralized_frame = str(ranked.iloc[0]["frame_name"])

    top_slot_share = 0.0
    if not slot_metrics.empty:
        total_betweenness = float(slot_metrics["graph_betweenness"].sum())
        if total_betweenness > 0:
            top_slot_share = float(slot_metrics["graph_betweenness"].max() / total_betweenness)

    return {
        "slot_degree_mean": round(float(slot_metrics["graph_degree"].mean()), 6) if not slot_metrics.empty else 0.0,
        "slot_distance_to_plaza_mean": round(float(slot_metrics["graph_distance_to_plaza"].mean()), 6) if not slot_metrics.empty else 0.0,
        "slot_betweenness_top_share": round(top_slot_share, 6),
        "frame_betweenness_gap": round(
            float(frame_metrics["frame_betweenness"].max() - frame_metrics["frame_betweenness"].min()),
            6,
        ) if not frame_metrics.empty else 0.0,
        "frame_closeness_std": round(float(frame_metrics["frame_closeness"].std(ddof=0)), 6) if not frame_metrics.empty else 0.0,
        "overcentralized_frame": overcentralized_frame,
        "component_count": int(graph_metrics["graph_report"]["component_count"]),
    }
