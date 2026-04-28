"""
board_config.py — Board layout configuration for autoresearch
==============================================================
Board configuration data used by the layout search and evaluator.
The evaluator imports this file and runs analysis across candidate
board configurations.

Keep this module to data definitions and the generate_board assembler.

Ciudad Viva v3 topology:
  - Fixed scenic base board: river, 2 bridges, plaza, avenues, skyline
  - Plaza Central is FIXED in position. 3 bonus variants rotate per game.
  - 5 modular district frames (4 lots each, 2x2 grid), use 4-5 per game
  - Frame-to-insert compatibility scoring (soft thematic match)
  - Lots have row modifiers: Fachada (row 0) vs Interior (row 1)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Bonus Values (how much each printed bonus is worth)
# ---------------------------------------------------------------------------

TRAFFIC_BONUS_IR = 1        # +IR per Trafico slot bonus
PRESTIGE_BONUS_MARCA = 1    # +Marca per Prestigio slot bonus
DISCOUNT_BONUS_K = 1        # -k placement cost per Descuento slot bonus

# ---------------------------------------------------------------------------
# Spatial Feature Values (from the fixed city skeleton)
# ---------------------------------------------------------------------------

SPATIAL_FEATURE_VALUES = {
    "riverfront": 0.35,         # premium scenic location
    "bridge_adjacent": 0.20,    # high visibility and access
    "plaza_adjacent": 0.35,     # prestige, central location
    "avenue_access": 0.15,      # good transport connections
    "market_street": 0.25,      # foot traffic and retail flow
    "outer_ring": -0.15,        # cheaper, lower desirability
    "dock_access": 0.15,        # logistics and service support
}

# ---------------------------------------------------------------------------
# Row Modifiers (Fachada vs Interior within each district frame)
# ---------------------------------------------------------------------------

ROW_MODIFIERS = {
    0: 0.1,     # Fachada: street/river-facing, higher visibility
    1: -0.1,    # Interior: behind, less exposed
}

# ---------------------------------------------------------------------------
# Frame-to-Insert Compatibility (soft thematic match)
# ---------------------------------------------------------------------------
# When a district insert's affinities overlap with its frame's preferred
# industries, all slots in that frame get a compatibility bonus.

FRAME_AFFINITY_PREFERENCE = {
    "Ribera Noroeste":  ("Food", "Real Estate"),     # waterfront dining, hotels
    "Ribera Noreste":   ("Service", "Real Estate"),   # marina, leisure
    "Barrio Comercial": ("Retail", "Food"),           # retail, food court
    "Darsena Sur":      ("Trades", "Service"),        # docks, workshops
    "Periferia":        ("Service", "Trades"),         # cheap services, trades
}

FRAME_COMPATIBILITY_BONUS = 0.15    # +EV when insert affinity matches frame

# ---------------------------------------------------------------------------
# Location Fee
# ---------------------------------------------------------------------------

LOCATION_FEE = 2.75         # k paid by Physical businesses in La Ciudad

# ---------------------------------------------------------------------------
# Barrio (personal player board)
# ---------------------------------------------------------------------------

BARRIO_SLOTS = 6            # total slots on each player's personal board
BARRIO_OPEN_SLOTS = 4       # available from the start
BARRIO_UNLOCK_SLOTS = 2     # unlock via progression (e.g., Net Profit 10+ or 5+ businesses)

# ---------------------------------------------------------------------------
# Lots per district frame (v2+: 2x2 grid = 4 lots per insert)
# ---------------------------------------------------------------------------

LOTS_PER_FRAME = 4
FRAME_COLS = 2
FRAME_ROWS = 2

# ---------------------------------------------------------------------------
# Slot sizes by business tier
# ---------------------------------------------------------------------------

SLOT_SIZE_BY_TIER = {
    "Starter": 1,
    "Established": 1,
    "Premium": 2,
    "Empire": 2,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlotBonus:
    """A bonus printed on a specific slot within a tile."""
    slot_index: int     # 0-3 within the 2x2 grid (row = idx // 2, col = idx % 2)
    bonus_type: str     # "Trafico" | "Prestigio" | "Descuento"


@dataclass
class DistrictTileFace:
    """One face (side) of a district tile."""
    name: str
    affinities: tuple[str, str]     # two industry affinities
    bonuses: list[SlotBonus] = field(default_factory=list)


@dataclass
class DistrictTile:
    """A double-sided district tile."""
    tile_id: int
    side_a: DistrictTileFace
    side_b: DistrictTileFace


@dataclass
class CityFrame:
    """A fixed position on the city skeleton where a district insert plugs in."""
    name: str
    features: tuple[str, ...]       # spatial features from the fixed skeleton


@dataclass
class RuntimeBoardConfig:
    """Serializable runtime config for in-memory board search."""

    traffic_bonus_ir: float
    prestige_bonus_marca: float
    discount_bonus_k: float
    spatial_feature_values: dict[str, float]
    row_modifiers: dict[int, float]
    frame_affinity_preference: dict[str, tuple[str, ...]]
    frame_compatibility_bonus: float
    location_fee: float
    barrio_slots: int
    barrio_open_slots: int
    barrio_unlock_slots: int
    lots_per_frame: int
    frame_cols: int
    frame_rows: int
    slot_size_by_tier: dict[str, int]
    plaza_variants: list[DistrictTileFace]
    plaza_frame: CityFrame
    city_frames: dict[int, list[CityFrame]]
    frame_adjacency_by_player_count: dict[int, tuple[tuple[str, str], ...]]
    district_tiles: list[DistrictTile]


# ---------------------------------------------------------------------------
# Plaza Central — FIXED POSITION, 3 bonus variants
# ---------------------------------------------------------------------------
# The civic anchor. Position, affinities, and features are always the same.
# Only the bonus pattern rotates randomly each game.

PLAZA_VARIANTS = [
    # Variant A — Prestige-heavy
    DistrictTileFace(
        name="Plaza Central",
        affinities=("Professional", "Real Estate"),
        bonuses=[
            SlotBonus(0, "Prestigio"),
            SlotBonus(1, "Prestigio"),
            SlotBonus(2, "Trafico"),
            SlotBonus(3, "Trafico"),
        ],
    ),
    # Variant B — Commerce-heavy
    DistrictTileFace(
        name="Plaza Central",
        affinities=("Professional", "Real Estate"),
        bonuses=[
            SlotBonus(0, "Trafico"),
            SlotBonus(1, "Trafico"),
            SlotBonus(2, "Descuento"),
            SlotBonus(3, "Trafico"),
        ],
    ),
    # Variant C — Balanced
    DistrictTileFace(
        name="Plaza Central",
        affinities=("Professional", "Real Estate"),
        bonuses=[
            SlotBonus(0, "Prestigio"),
            SlotBonus(1, "Trafico"),
            SlotBonus(2, "Trafico"),
            SlotBonus(3, "Descuento"),
        ],
    ),
]

PLAZA_FRAME = CityFrame("Plaza Central", features=("plaza_adjacent", "avenue_access"))


# ---------------------------------------------------------------------------
# City Skeleton — Modular Frame Positions
# ---------------------------------------------------------------------------
# Plaza Central is NOT in this list — it is fixed and always present.
# Frames at lower player counts become scenic dead space (park, harbor).

CITY_FRAMES = {
    # 4p: all 5 modular frames active
    4: [
        CityFrame("Ribera Noroeste",    features=("riverfront", "bridge_adjacent")),
        CityFrame("Ribera Noreste",     features=("riverfront", "bridge_adjacent")),
        CityFrame("Barrio Comercial",   features=("market_street", "avenue_access")),
        CityFrame("Darsena Sur",        features=("dock_access", "riverfront")),
        CityFrame("Periferia",          features=("outer_ring",)),
    ],
    # 3p: Periferia closes → Parque Municipal (scenic art)
    3: [
        CityFrame("Ribera Noroeste",    features=("riverfront", "bridge_adjacent")),
        CityFrame("Ribera Noreste",     features=("riverfront", "bridge_adjacent")),
        CityFrame("Barrio Comercial",   features=("market_street", "avenue_access")),
        CityFrame("Darsena Sur",        features=("dock_access", "riverfront")),
    ],
    # 2p: Ribera Noreste + Periferia close. South bank always matters.
    2: [
        CityFrame("Ribera Noroeste",    features=("riverfront", "bridge_adjacent")),
        CityFrame("Barrio Comercial",   features=("market_street", "avenue_access")),
        CityFrame("Darsena Sur",        features=("dock_access", "riverfront")),
    ],
}


# ---------------------------------------------------------------------------
# Canonical Frame Connectivity
# ---------------------------------------------------------------------------
# Executable board-graph adjacency for Ciudad Viva.
# This is structural board topology only; gameplay rules do not currently
# consume these links directly.

FRAME_ADJACENCY_BY_PLAYER_COUNT = {
    4: (
        ("Plaza Central", "Ribera Noroeste"),
        ("Plaza Central", "Ribera Noreste"),
        ("Plaza Central", "Barrio Comercial"),
        ("Plaza Central", "Darsena Sur"),
        ("Barrio Comercial", "Ribera Noroeste"),
        ("Darsena Sur", "Ribera Noreste"),
        ("Barrio Comercial", "Periferia"),
        ("Darsena Sur", "Periferia"),
    ),
    3: (
        ("Plaza Central", "Ribera Noroeste"),
        ("Plaza Central", "Ribera Noreste"),
        ("Plaza Central", "Barrio Comercial"),
        ("Plaza Central", "Darsena Sur"),
        ("Barrio Comercial", "Ribera Noroeste"),
        ("Darsena Sur", "Ribera Noreste"),
    ),
    2: (
        ("Plaza Central", "Ribera Noroeste"),
        ("Plaza Central", "Barrio Comercial"),
        ("Plaza Central", "Darsena Sur"),
        ("Barrio Comercial", "Ribera Noroeste"),
        ("Barrio Comercial", "Darsena Sur"),
    ),
}


# ---------------------------------------------------------------------------
# The 8 District Tiles (16 faces) — modular inserts
# ---------------------------------------------------------------------------
# Each tile uses a 2x2 grid (4 lots). Slot indices 0-3.
# Row 0 = Fachada (L0, L1), Row 1 = Interior (L2, L3).

DISTRICT_TILES = [
    # Tile 1 — Financial / Executive
    DistrictTile(
        tile_id=1,
        side_a=DistrictTileFace(
            name="Centro Financiero",
            affinities=("Professional", "Retail"),
            bonuses=[
                SlotBonus(0, "Prestigio"),
                SlotBonus(1, "Trafico"),
                SlotBonus(2, "Descuento"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Zona Ejecutiva",
            affinities=("Professional", "Real Estate"),
            bonuses=[
                SlotBonus(0, "Prestigio"),
                SlotBonus(3, "Descuento"),
            ],
        ),
    ),

    # Tile 2 — Waterfront / Marina
    DistrictTile(
        tile_id=2,
        side_a=DistrictTileFace(
            name="Paseo Maritimo",
            affinities=("Food", "Real Estate"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(1, "Trafico"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Puerto Deportivo",
            affinities=("Service", "Real Estate"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(3, "Prestigio"),
            ],
        ),
    ),

    # Tile 3 — Commercial / Market
    DistrictTile(
        tile_id=3,
        side_a=DistrictTileFace(
            name="Calle Comercial",
            affinities=("Retail", "Food"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(1, "Trafico"),
                SlotBonus(2, "Descuento"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Plaza del Mercado",
            affinities=("Retail", "Service"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(2, "Descuento"),
            ],
        ),
    ),

    # Tile 4 — Tech / Innovation
    DistrictTile(
        tile_id=4,
        side_a=DistrictTileFace(
            name="Campus Tech",
            affinities=("Real Estate", "Professional"),
            bonuses=[
                SlotBonus(0, "Descuento"),
                SlotBonus(1, "Prestigio"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Hub de Innovacion",
            affinities=("Food", "Trades"),
            bonuses=[
                SlotBonus(0, "Descuento"),
                SlotBonus(1, "Trafico"),
            ],
        ),
    ),

    # Tile 5 — Industrial / Logistics
    DistrictTile(
        tile_id=5,
        side_a=DistrictTileFace(
            name="Poligono Industrial",
            affinities=("Trades", "Service"),
            bonuses=[
                SlotBonus(0, "Descuento"),
                SlotBonus(2, "Descuento"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Centro Logistico",
            affinities=("Trades", "Retail"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(3, "Descuento"),
            ],
        ),
    ),

    # Tile 6 — University / Medical
    DistrictTile(
        tile_id=6,
        side_a=DistrictTileFace(
            name="Barrio Universitario",
            affinities=("Professional", "Service"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(2, "Prestigio"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Centro Medico",
            affinities=("Professional", "Food"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(3, "Descuento"),
            ],
        ),
    ),

    # Tile 7 — Residential / Suburban
    DistrictTile(
        tile_id=7,
        side_a=DistrictTileFace(
            name="Zona Residencial",
            affinities=("Service", "Food"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(1, "Trafico"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Urbanizacion",
            affinities=("Service", "Trades"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(2, "Descuento"),
            ],
        ),
    ),

    # Tile 8 — Tourism / Cultural
    DistrictTile(
        tile_id=8,
        side_a=DistrictTileFace(
            name="Distrito Turistico",
            affinities=("Food", "Retail"),
            bonuses=[
                SlotBonus(0, "Trafico"),
                SlotBonus(1, "Prestigio"),
            ],
        ),
        side_b=DistrictTileFace(
            name="Barrio Cultural",
            affinities=("Professional", "Real Estate"),
            bonuses=[
                SlotBonus(0, "Prestigio"),
                SlotBonus(3, "Trafico"),
            ],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _clone_bonus_list(bonuses: list[SlotBonus]) -> list[SlotBonus]:
    return [replace(bonus) for bonus in bonuses]


def _clone_face(face: DistrictTileFace) -> DistrictTileFace:
    return DistrictTileFace(
        name=face.name,
        affinities=tuple(face.affinities),
        bonuses=_clone_bonus_list(face.bonuses),
    )


def _clone_tile(tile: DistrictTile) -> DistrictTile:
    return DistrictTile(
        tile_id=tile.tile_id,
        side_a=_clone_face(tile.side_a),
        side_b=_clone_face(tile.side_b),
    )


def _clone_frame(frame: CityFrame) -> CityFrame:
    return CityFrame(name=frame.name, features=tuple(frame.features))


def build_runtime_config() -> RuntimeBoardConfig:
    """Return a detached snapshot of the current board configuration."""
    return RuntimeBoardConfig(
        traffic_bonus_ir=TRAFFIC_BONUS_IR,
        prestige_bonus_marca=PRESTIGE_BONUS_MARCA,
        discount_bonus_k=DISCOUNT_BONUS_K,
        spatial_feature_values=dict(SPATIAL_FEATURE_VALUES),
        row_modifiers=dict(ROW_MODIFIERS),
        frame_affinity_preference={
            name: tuple(inds) for name, inds in FRAME_AFFINITY_PREFERENCE.items()
        },
        frame_compatibility_bonus=FRAME_COMPATIBILITY_BONUS,
        location_fee=LOCATION_FEE,
        barrio_slots=BARRIO_SLOTS,
        barrio_open_slots=BARRIO_OPEN_SLOTS,
        barrio_unlock_slots=BARRIO_UNLOCK_SLOTS,
        lots_per_frame=LOTS_PER_FRAME,
        frame_cols=FRAME_COLS,
        frame_rows=FRAME_ROWS,
        slot_size_by_tier=dict(SLOT_SIZE_BY_TIER),
        plaza_variants=[_clone_face(face) for face in PLAZA_VARIANTS],
        plaza_frame=_clone_frame(PLAZA_FRAME),
        city_frames={
            player_count: [_clone_frame(frame) for frame in frames]
            for player_count, frames in CITY_FRAMES.items()
        },
        frame_adjacency_by_player_count={
            player_count: tuple((left, right) for left, right in edges)
            for player_count, edges in FRAME_ADJACENCY_BY_PLAYER_COUNT.items()
        },
        district_tiles=[_clone_tile(tile) for tile in DISTRICT_TILES],
    )


def generate_board_from_runtime(runtime: RuntimeBoardConfig, player_count: int, seed: int) -> dict:
    """Assemble a board from an in-memory runtime config."""
    modular_frames = runtime.city_frames[player_count]
    n_modular = len(modular_frames)

    digest = hashlib.sha256(f"board::{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))

    plaza = _clone_face(rng.choice(runtime.plaza_variants))

    tile_indices = list(range(len(runtime.district_tiles)))
    rng.shuffle(tile_indices)
    selected_indices = tile_indices[:n_modular]

    selected_faces: list[DistrictTileFace] = []
    for idx in sorted(selected_indices):
        tile = runtime.district_tiles[idx]
        face = _clone_face(rng.choice([tile.side_a, tile.side_b]))
        selected_faces.append(face)

    frame_order = list(range(n_modular))
    rng.shuffle(frame_order)

    slots: list[dict] = []

    plaza_bonus_map = {b.slot_index: b.bonus_type for b in plaza.bonuses}
    for slot_idx in range(runtime.lots_per_frame):
        row = slot_idx // runtime.frame_cols
        col = slot_idx % runtime.frame_cols
        slots.append({
            "tile_pos": -1,
            "tile_name": plaza.name,
            "frame_name": runtime.plaza_frame.name,
            "frame_features": runtime.plaza_frame.features,
            "slot_idx": slot_idx,
            "row": row,
            "col": col,
            "row_modifier": runtime.row_modifiers.get(row, 0.0),
            "affinities": plaza.affinities,
            "bonus_type": plaza_bonus_map.get(slot_idx),
            "is_plaza": True,
            "frame_compatible": False,
        })

    for insert_idx, frame_idx in enumerate(frame_order):
        face = selected_faces[insert_idx]
        frame = modular_frames[frame_idx]
        bonus_map = {b.slot_index: b.bonus_type for b in face.bonuses}

        frame_prefs = runtime.frame_affinity_preference.get(frame.name, ())
        has_compatibility = bool(set(face.affinities) & set(frame_prefs)) if frame_prefs else False

        for slot_idx in range(runtime.lots_per_frame):
            row = slot_idx // runtime.frame_cols
            col = slot_idx % runtime.frame_cols
            slots.append({
                "tile_pos": insert_idx,
                "tile_name": face.name,
                "frame_name": frame.name,
                "frame_features": frame.features,
                "slot_idx": slot_idx,
                "row": row,
                "col": col,
                "row_modifier": runtime.row_modifiers.get(row, 0.0),
                "affinities": face.affinities,
                "bonus_type": bonus_map.get(slot_idx),
                "is_plaza": False,
                "frame_compatible": has_compatibility,
            })

    barrio_slots = [
        {
            "slot_idx": i,
            "bonus_type": None,
            "affinities": (),
            "frame_features": (),
            "row_modifier": 0.0,
            "frame_compatible": False,
        }
        for i in range(runtime.barrio_slots)
    ]

    return {
        "tiles": [plaza, *selected_faces],
        "frames": modular_frames,
        "frame_adjacency": list(runtime.frame_adjacency_by_player_count[player_count]),
        "plaza": plaza,
        "slots": slots,
        "barrio_slots": barrio_slots,
        "player_count": player_count,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Board Generation
# ---------------------------------------------------------------------------

def generate_board(player_count: int, seed: int) -> dict:
    """Assemble a board using the current module-level config."""
    return generate_board_from_runtime(build_runtime_config(), player_count, seed)
