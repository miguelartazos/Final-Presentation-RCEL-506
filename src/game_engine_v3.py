"""
game_engine_v3.py — Quarterly Business Empire simulator
=======================================================
Models the active v3 build around:

- 5 department actions
- quarterly planning with a public Reserve card
- 1 paid Giro per quarter
- permanent department levels + public focus cubes
- simple finance model with loans and bank liquidation fallback
- Condiciones de Mercado as the only macro system

The public API is intentionally kept compatible with the optimizer and
slide export scripts.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

DEPARTMENTS = (
    "OPORTUNIDADES",
    "LANZAR",
    "TALENTO",
    "CRECER",
    "FINANZAS",
)

MARKET_ROW_SIZE = 4


@dataclass
class Loan:
    loan_type: str
    amount: float
    interest: float
    refinanced_this_quarter: bool = False


@dataclass
class BoostCard:
    id: str
    name: str
    tags: list = field(default_factory=list)


@dataclass
class MarketCondition:
    id: str
    name: str
    tags: list = field(default_factory=list)


@dataclass
class BusinessCard:
    id: str
    name: str
    industry: str
    tier: str
    cost: float
    income: float
    valuation_points: float
    exit_value: float
    tempo: str
    mode: str
    staff_min: int
    income_scaled: float
    synergy_gives: str
    synergy_receives: str
    staff_opt: int = 0
    income_opt: float = 0.0
    tags: list = field(default_factory=list)
    time_delay: int = 0
    effort: int = 0
    likelihood: int = 10


@dataclass
class ActiveBusiness:
    card: BusinessCard
    employees_assigned: int
    city_placed: bool = False
    placement_area: str | None = None
    placement_frame: str | None = None
    placement_slot_ids: tuple[str, ...] = field(default_factory=tuple)
    placement_bonus_types: tuple[str, ...] = field(default_factory=tuple)
    district_affinity_match: bool = False
    has_manager: bool = False
    growth_cubes: int = 0
    trend_bonus: int = 0
    refreshed_this_quarter: bool = False
    break_bonus: float = 0.0

    @property
    def staffing_value(self) -> int:
        return self.employees_assigned + (3 if self.has_manager else 0)

    @property
    def _base_income(self) -> float:
        if (
            self.card.staff_opt > 0
            and self.staffing_value >= self.card.staff_opt
            and self.card.income_opt > 0
        ):
            return self.card.income_opt
        return self.card.income

    @property
    def current_income(self) -> float:
        base = self._base_income
        if self.card.tempo == "Escala":
            if self.growth_cubes >= 2 and self.card.income_scaled > 0:
                return self.card.income_scaled
            return base
        if self.card.tempo == "Tendencia":
            return base + self.trend_bonus
        return base


@dataclass
class QuarterPlan:
    reserve_action: str
    action_order: tuple[str, str, str, str]
    focus_actions: tuple[str, ...]


@dataclass
class StrategyProfile:
    name: str
    action_bias: dict[str, float]
    card_bias: dict[str, float]


@dataclass
class BoardSlot:
    slot_id: str
    area: str
    owner: str | None
    frame_name: str
    tile_name: str
    row: int
    col: int
    bonus_type: str | None
    affinities: tuple[str, ...]
    frame_features: tuple[str, ...]
    is_plaza: bool = False
    occupied_by: str | None = None


@dataclass
class PlacementDecision:
    area: str
    frame_name: str
    slot_ids: tuple[str, ...]
    slot_bonus_types: tuple[str, ...]
    district_affinity_match: bool
    city_placed: bool
    total_cost: float
    score: float


@dataclass
class BoardState:
    board_seed: int
    city_slots: list[BoardSlot]
    barrio_slots_by_player: dict[str, list[BoardSlot]]
    source_layout: dict[str, object] = field(default_factory=dict)


@dataclass
class GameConfig:
    starting_cash: float = 10.0
    starting_draw_size: int = 5
    starting_hand_size: int = 4
    max_hand_size: int = 8
    starting_brand: int = 5
    starting_employees: int = 9
    max_total_employees: int = 15
    turns_per_quarter: int = 4
    max_quarters: int = 8
    location_fee: float = 2.75
    focus_cube_cost: float = 2.0
    giro_cost: float = 2.0
    upgrade_cost_i_to_ii: float = 4.0
    upgrade_cost_ii_to_iii: float = 7.0
    end_business_threshold: int = 8
    profit_machine_threshold: float = 15.0
    enable_loans: bool = True
    enable_market_conditions: bool = True
    enable_synergies: bool = True
    enable_boosts: bool = True
    board_enabled: bool = False
    player_count: int = 2
    board_seed: int | None = None
    board_policy: str = "heuristic"
    barrio_unlock_thresholds: tuple[int, int] = (3, 5)
    runtime_board_config: Any | None = None


@dataclass
class Player:
    name: str
    strategy: str
    cash: float = 10.0
    brand: int = 5
    businesses: list[ActiveBusiness] = field(default_factory=list)
    hand: list[BusinessCard] = field(default_factory=list)
    employees_reserve: int = 9
    total_employees: int = 9
    department_levels: dict[str, int] = field(
        default_factory=lambda: {dept: 1 for dept in DEPARTMENTS}
    )
    cards_played_ids: list[str] = field(default_factory=list)
    industries_played: dict[str, int] = field(default_factory=dict)
    loans: list[Loan] = field(default_factory=list)
    boost_hand: list[BoostCard] = field(default_factory=list)
    pending_launch_discounts: list[float] = field(default_factory=list)
    rng: random.Random | None = field(default=None, repr=False, compare=False)
    turns_played: int = 0
    giro_uses: int = 0
    focus_bought: int = 0
    focus_spent: int = 0
    blocked_city_launches: int = 0
    city_launches: int = 0
    barrio_launches: int = 0
    slot_bonus_triggers: int = 0
    district_affinity_triggers: int = 0
    first_city_frame: str | None = None
    quarter_history: list[dict] = field(default_factory=list)

    @property
    def recurring_income(self) -> float:
        return sum(b.current_income for b in self.businesses)

    @property
    def loan_interest_cost(self) -> float:
        return sum(loan.interest for loan in self.loans)

    @property
    def total_debt(self) -> float:
        return sum(loan.amount for loan in self.loans)

    @property
    def active_loan_count(self) -> int:
        return len(self.loans)

    @property
    def net_profit(self) -> float:
        return self.recurring_income - self.loan_interest_cost

    @property
    def business_count(self) -> int:
        return len(self.businesses)

    @property
    def employees_deployed(self) -> int:
        return sum(b.employees_assigned for b in self.businesses)

    @property
    def score(self) -> float:
        asset_vp = sum(b.card.valuation_points for b in self.businesses)
        exit_bonus = _exit_bonus_lookup(self.net_profit, self.brand)
        cash_vp = int(max(self.cash, 0) // 5)
        debt_penalty = int(self.total_debt // 5) + self.active_loan_count * 2
        return asset_vp + exit_bonus + cash_vp - debt_penalty


EXIT_BONUS_TABLE = [
    [0, 2, 4],
    [4, 8, 12],
    [8, 13, 18],
    [12, 18, 24],
]

CITY_FRIENDLY_INDUSTRIES = {"Food", "Retail", "Professional", "Real Estate", "Service"}
PLATFORM_DEPENDENT_IDS = {"BUS-RETAIL-002", "BUS-RETAIL-004", "BUS-TECH-005"}
AI_SHIELD_IDS = {"BUS-SERVICE-006"}


def _brand_tier(brand: int) -> int:
    if brand <= 5:
        return 0
    if brand <= 11:
        return 1
    return 2


def _profit_band(net_profit: float) -> int:
    if net_profit < 5:
        return 0
    if net_profit < 10:
        return 1
    if net_profit < 15:
        return 2
    return 3


def _exit_bonus_lookup(net_profit: float, brand: int) -> int:
    return EXIT_BONUS_TABLE[_profit_band(net_profit)][_brand_tier(brand)]


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val == val and val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val)) if val == val and val is not None else default
    except (TypeError, ValueError):
        return default


def _build_board_state(player_names: tuple[str, str], cfg: GameConfig, seed: int | None) -> BoardState | None:
    if not cfg.board_enabled:
        return None

    import board_config

    board_seed = seed if seed is not None else (cfg.board_seed or _stable_seed("board", cfg.player_count))
    if cfg.runtime_board_config is not None:
        layout = board_config.generate_board_from_runtime(cfg.runtime_board_config, cfg.player_count, board_seed)
    else:
        layout = board_config.generate_board(cfg.player_count, board_seed)

    city_slots = [
        BoardSlot(
            slot_id=f"city:{idx}",
            area="Ciudad",
            owner=None,
            frame_name=str(slot["frame_name"]),
            tile_name=str(slot["tile_name"]),
            row=int(slot.get("row", 0)),
            col=int(slot.get("col", 0)),
            bonus_type=slot.get("bonus_type"),
            affinities=tuple(slot.get("affinities", ())),
            frame_features=tuple(slot.get("frame_features", ())),
            is_plaza=bool(slot.get("is_plaza", False)),
        )
        for idx, slot in enumerate(layout["slots"])
    ]

    barrio_slots_by_player: dict[str, list[BoardSlot]] = {}
    for player_index, player_name in enumerate(player_names):
        slots = []
        for idx in range(board_config.BARRIO_SLOTS):
            slots.append(
                BoardSlot(
                    slot_id=f"barrio:{player_index}:{idx}",
                    area="Barrio",
                    owner=player_name,
                    frame_name=f"Barrio {player_name}",
                    tile_name="Barrio",
                    row=idx // 2,
                    col=idx % 2,
                    bonus_type=None,
                    affinities=(),
                    frame_features=(),
                )
            )
        barrio_slots_by_player[player_name] = slots

    return BoardState(
        board_seed=board_seed,
        city_slots=city_slots,
        barrio_slots_by_player=barrio_slots_by_player,
        source_layout=layout,
    )


def _slot_size(card: BusinessCard) -> int:
    return 2 if card.tier in {"Premium", "Empire"} else 1


def _barrio_open_slots(player: Player, cfg: GameConfig, slots: list[BoardSlot]) -> list[BoardSlot]:
    open_count = 4
    thresholds = cfg.barrio_unlock_thresholds
    if player.business_count >= thresholds[0]:
        open_count += 1
    if player.business_count >= thresholds[1]:
        open_count += 1
    return slots[: min(open_count, len(slots))]


def _find_open_runs(slots: list[BoardSlot], size: int) -> list[tuple[BoardSlot, ...]]:
    open_slots = [slot for slot in slots if slot.occupied_by is None]
    if size == 1:
        return [(slot,) for slot in open_slots]

    runs: list[tuple[BoardSlot, ...]] = []
    slot_by_pos = {(slot.row, slot.col): slot for slot in open_slots}
    seen: set[tuple[str, str]] = set()
    for slot in open_slots:
        for delta_row, delta_col in ((0, 1), (1, 0)):
            other = slot_by_pos.get((slot.row + delta_row, slot.col + delta_col))
            if other is None:
                continue
            key = tuple(sorted((slot.slot_id, other.slot_id)))
            if key in seen:
                continue
            seen.add(key)
            runs.append((slot, other))
    return runs


def _slot_bonus_value(bonus_type: str | None, profile: StrategyProfile) -> float:
    if bonus_type == "Trafico":
        return 1.0 * profile.card_bias["income"]
    if bonus_type == "Prestigio":
        return 0.8 * profile.card_bias["brand"]
    if bonus_type == "Descuento":
        return 0.8 * profile.card_bias["cheap"]
    return 0.0


def _placement_candidates(
    player: Player,
    card: BusinessCard,
    level: int,
    profile: StrategyProfile,
    cfg: GameConfig,
    board_state: BoardState | None,
) -> list[PlacementDecision]:
    if card.mode != "Physical" or board_state is None:
        return []

    size = _slot_size(card)
    candidates: list[PlacementDecision] = []

    city_legal = _available_city_placement(card, level)
    if city_legal:
        city_by_frame: dict[str, list[BoardSlot]] = {}
        for slot in board_state.city_slots:
            city_by_frame.setdefault(slot.frame_name, []).append(slot)
        for frame_name, slots in city_by_frame.items():
            for run in _find_open_runs(slots, size):
                bonus_types = tuple(slot.bonus_type for slot in run if slot.bonus_type)
                discount_count = sum(1 for bonus in bonus_types if bonus == "Descuento")
                total_cost = card.cost + cfg.location_fee - discount_count
                district_match = any(card.industry in slot.affinities for slot in run)
                scenic_value = 0.05 * sum(len(slot.frame_features) for slot in run)
                score = (
                    sum(_slot_bonus_value(slot.bonus_type, profile) for slot in run)
                    + (1.1 if district_match else 0.0)
                    + scenic_value
                    - max(total_cost - card.cost, 0.0) * 0.35
                )
                candidates.append(
                    PlacementDecision(
                        area="Ciudad",
                        frame_name=frame_name,
                        slot_ids=tuple(slot.slot_id for slot in run),
                        slot_bonus_types=bonus_types,
                        district_affinity_match=district_match,
                        city_placed=True,
                        total_cost=total_cost,
                        score=score,
                    )
                )

    barrio_slots = _barrio_open_slots(player, cfg, board_state.barrio_slots_by_player[player.name])
    for run in _find_open_runs(barrio_slots, size):
        candidates.append(
            PlacementDecision(
                area="Barrio",
                frame_name=f"Barrio {player.name}",
                slot_ids=tuple(slot.slot_id for slot in run),
                slot_bonus_types=(),
                district_affinity_match=False,
                city_placed=False,
                total_cost=card.cost,
                score=0.25 * profile.card_bias["cheap"],
            )
        )
    return candidates


def _occupy_slots(board_state: BoardState, slot_ids: tuple[str, ...], business_id: str) -> None:
    all_slots = board_state.city_slots + [
        slot
        for slots in board_state.barrio_slots_by_player.values()
        for slot in slots
    ]
    slot_map = {slot.slot_id: slot for slot in all_slots}
    for slot_id in slot_ids:
        slot_map[slot_id].occupied_by = business_id


def _parse_synergy_category(synergy_str: str) -> str:
    if not synergy_str or not synergy_str.strip():
        return ""
    return synergy_str.split("->")[0].strip() if "->" in synergy_str else synergy_str.strip()


def _empty_card_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "card_id",
            "times_played",
            "usage_rate",
            "wins_played",
            "losses_played",
            "draws_played",
            "win_deck_rate",
            "loss_deck_rate",
            "draw_deck_rate",
            "win_bias",
        ]
    )


def _synergy_match_score(player: Player, card: BusinessCard) -> int:
    score = 0
    card_gives = _parse_synergy_category(card.synergy_gives)
    card_receives = _parse_synergy_category(card.synergy_receives)
    for biz in player.businesses:
        biz_gives = _parse_synergy_category(biz.card.synergy_gives)
        biz_receives = _parse_synergy_category(biz.card.synergy_receives)
        if card_gives and biz_receives and card_gives == biz_receives:
            score += 2
        if card_receives and biz_gives and card_receives == biz_gives:
            score += 2
    if card_gives:
        score += 1
    return score


def _resolve_synergies_at_break(player: Player) -> None:
    gives_categories: dict[str, list[ActiveBusiness]] = {}
    for biz in player.businesses:
        cat = _parse_synergy_category(biz.card.synergy_gives)
        if cat:
            gives_categories.setdefault(cat, []).append(biz)

    for biz in player.businesses:
        recv_cat = _parse_synergy_category(biz.card.synergy_receives)
        if not recv_cat:
            continue
        givers = gives_categories.get(recv_cat, [])
        if not any(g is not biz for g in givers):
            continue
        if recv_cat == "Tráfico":
            biz.break_bonus += 1
        elif recv_cat == "Prestigio":
            player.brand = min(18, player.brand + 1)


def _resolve_synergies_on_launch(player: Player, new_biz: ActiveBusiness) -> None:
    new_gives = _parse_synergy_category(new_biz.card.synergy_gives)
    new_receives = _parse_synergy_category(new_biz.card.synergy_receives)

    if new_gives:
        for biz in player.businesses:
            if biz is new_biz:
                continue
            recv_cat = _parse_synergy_category(biz.card.synergy_receives)
            if recv_cat == new_gives and new_gives == "Tecnología" and biz.card.tempo == "Escala":
                biz.growth_cubes = min(2, biz.growth_cubes + 1)
        if new_gives in {"Suministro", "Experiencia"}:
            player.pending_launch_discounts.append(1.0)

    if new_receives:
        for biz in player.businesses:
            if biz is new_biz:
                continue
            gives_cat = _parse_synergy_category(biz.card.synergy_gives)
            if gives_cat == new_receives and new_receives == "Tecnología" and new_biz.card.tempo == "Escala":
                new_biz.growth_cubes = min(2, new_biz.growth_cubes + 1)


def _resolve_market_condition_at_break(condition: MarketCondition, player: Player) -> None:
    if condition.id == "MKT-ANY-002":
        cost = sum(1 for b in player.businesses if b.card.staff_min >= 4)
        has_shield = any(b.card.id in AI_SHIELD_IDS for b in player.businesses)
        if has_shield and cost > 0:
            cost -= 1
        player.cash -= min(cost, 2)
    elif condition.id == "MKT-ANY-004":
        cost = sum(1 for b in player.businesses if b.card.id in PLATFORM_DEPENDENT_IDS)
        player.cash -= min(cost, 3)
    elif condition.id == "MKT-ANY-006":
        player.cash -= player.active_loan_count
    elif condition.id == "MKT-ANY-007":
        for biz in player.businesses:
            if biz.card.tempo == "Tendencia" and biz.card.mode == "Physical":
                biz.break_bonus += 1


def _best_boost_target(player: Player, boost: BoostCard) -> int:
    if not player.businesses:
        if boost.id == "BOO-ANY-006":
            return 0 if player.business_count <= 2 else -1
        if boost.id == "BOO-ANY-012":
            return 0
        return -1

    if boost.id == "BOO-ANY-001":
        for i, b in enumerate(player.businesses):
            if b.card.tempo == "Tendencia" and not b.refreshed_this_quarter:
                return i
    elif boost.id == "BOO-ANY-002":
        return len(player.businesses) - 1
    elif boost.id == "BOO-ANY-003":
        if sum(1 for b in player.businesses if b.card.income >= 3) >= 2:
            return 0
    elif boost.id == "BOO-ANY-004":
        for i, b in enumerate(player.businesses):
            if b.card.tempo == "Escala" and b.card.mode == "Digital" and b.growth_cubes < 2:
                return i
    elif boost.id == "BOO-ANY-005":
        industries = {b.card.industry for b in player.businesses}
        return 0 if len(industries) >= 2 else -1
    elif boost.id == "BOO-ANY-006":
        return 0 if player.business_count <= 2 else -1
    elif boost.id in ("BOO-ANY-007", "BOO-ANY-002"):
        return 0
    elif boost.id == "BOO-ANY-009":
        for i, b in enumerate(player.businesses):
            if b.card.tempo == "Escala" and b.growth_cubes < 2:
                return i
        return 0
    elif boost.id == "BOO-ANY-012":
        return 0
    elif boost.id == "BOO-ANY-013":
        for i, b in enumerate(player.businesses):
            if b.card.industry in ("Food", "Retail"):
                return i
    return -1


def _resolve_boost(
    player: Player,
    boost: BoostCard,
    target_idx: int,
    deck: list[BusinessCard],
    config: GameConfig,
) -> None:
    biz = player.businesses[target_idx] if 0 <= target_idx < len(player.businesses) else None

    if boost.id == "BOO-ANY-001":
        if biz and biz.card.tempo == "Tendencia":
            biz.trend_bonus = 2
            biz.refreshed_this_quarter = True
            if biz.card.mode == "Digital":
                biz.break_bonus += 1
    elif boost.id == "BOO-ANY-002":
        if biz:
            biz.break_bonus += 2
    elif boost.id == "BOO-ANY-003":
        if sum(1 for b in player.businesses if b.card.income >= 3) >= 2 and biz:
            biz.break_bonus += 2
    elif boost.id == "BOO-ANY-004":
        if biz and biz.card.tempo == "Escala" and biz.card.mode == "Digital":
            biz.growth_cubes = min(2, biz.growth_cubes + 1)
    elif boost.id == "BOO-ANY-005":
        industries_seen = set()
        targets = []
        for i, active in enumerate(player.businesses):
            if active.card.industry not in industries_seen:
                industries_seen.add(active.card.industry)
                targets.append(i)
                if len(targets) == 2:
                    break
        for ti in targets:
            player.businesses[ti].break_bonus += 1
    elif boost.id == "BOO-ANY-006":
        if player.business_count <= 2:
            player.cash += 4
    elif boost.id == "BOO-ANY-007":
        if biz:
            biz.break_bonus += 2
            if biz.card.tempo == "Tendencia":
                biz.trend_bonus = 2
                biz.refreshed_this_quarter = True
    elif boost.id == "BOO-ANY-009":
        if biz:
            if biz.card.tempo == "Escala":
                biz.growth_cubes = min(2, biz.growth_cubes + 1)
            else:
                biz.break_bonus += 2
    elif boost.id == "BOO-ANY-012":
        for _ in range(3):
            if deck:
                player.hand.append(deck.pop())
        _trim_hand(player, profile=_profile_for(player.strategy))
    elif boost.id == "BOO-ANY-013":
        if biz and biz.card.industry in ("Food", "Retail"):
            biz.break_bonus += 2


STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "Random": StrategyProfile(
        name="Random",
        action_bias={dept: 1.0 for dept in DEPARTMENTS},
        card_bias={"income": 1.0, "vp": 1.0, "scale": 1.0, "trend": 1.0, "brand": 1.0, "synergy": 1.0, "cheap": 1.0},
    ),
    "Greedy_VP": StrategyProfile(
        name="Greedy_VP",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.2, "TALENTO": 0.8, "CRECER": 0.9, "FINANZAS": 0.8},
        card_bias={"income": 0.8, "vp": 1.8, "scale": 1.0, "trend": 0.9, "brand": 1.0, "synergy": 1.0, "cheap": 0.7},
    ),
    "Cash_Machine": StrategyProfile(
        name="Cash_Machine",
        action_bias={"OPORTUNIDADES": 0.9, "LANZAR": 1.4, "TALENTO": 0.8, "CRECER": 0.7, "FINANZAS": 1.0},
        card_bias={"income": 1.8, "vp": 0.7, "scale": 0.9, "trend": 0.9, "brand": 0.7, "synergy": 0.9, "cheap": 1.0},
    ),
    "Stable_Heavy": StrategyProfile(
        name="Stable_Heavy",
        action_bias={"OPORTUNIDADES": 0.9, "LANZAR": 1.3, "TALENTO": 0.8, "CRECER": 0.7, "FINANZAS": 1.0},
        card_bias={"income": 1.5, "vp": 0.9, "scale": 0.4, "trend": 0.5, "brand": 0.7, "synergy": 0.8, "cheap": 1.2},
    ),
    "Scale_Rush": StrategyProfile(
        name="Scale_Rush",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.2, "TALENTO": 0.8, "CRECER": 1.3, "FINANZAS": 0.9},
        card_bias={"income": 1.1, "vp": 1.0, "scale": 1.9, "trend": 0.6, "brand": 0.8, "synergy": 1.0, "cheap": 0.8},
    ),
    "Trend_Surfer": StrategyProfile(
        name="Trend_Surfer",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.1, "TALENTO": 0.8, "CRECER": 1.4, "FINANZAS": 0.8},
        card_bias={"income": 1.0, "vp": 0.9, "scale": 0.7, "trend": 1.8, "brand": 1.1, "synergy": 0.9, "cheap": 1.0},
    ),
    "Balanced_Tempo": StrategyProfile(
        name="Balanced_Tempo",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.1, "TALENTO": 0.9, "CRECER": 1.1, "FINANZAS": 0.9},
        card_bias={"income": 1.1, "vp": 1.1, "scale": 1.1, "trend": 1.1, "brand": 1.0, "synergy": 1.1, "cheap": 1.0},
    ),
    "Premium_Builder": StrategyProfile(
        name="Premium_Builder",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.3, "TALENTO": 0.8, "CRECER": 0.9, "FINANZAS": 1.2},
        card_bias={"income": 0.9, "vp": 1.5, "scale": 1.1, "trend": 0.8, "brand": 1.0, "synergy": 0.9, "cheap": 0.5},
    ),
    "Leveraged_Growth": StrategyProfile(
        name="Leveraged_Growth",
        action_bias={"OPORTUNIDADES": 0.9, "LANZAR": 1.2, "TALENTO": 0.8, "CRECER": 1.0, "FINANZAS": 1.5},
        card_bias={"income": 1.3, "vp": 1.0, "scale": 1.4, "trend": 0.8, "brand": 0.8, "synergy": 0.9, "cheap": 0.6},
    ),
    "Boost_Opportunist": StrategyProfile(
        name="Boost_Opportunist",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.0, "TALENTO": 0.8, "CRECER": 1.5, "FINANZAS": 0.8},
        card_bias={"income": 1.0, "vp": 0.9, "scale": 1.1, "trend": 1.2, "brand": 1.1, "synergy": 1.0, "cheap": 0.9},
    ),
    "Synergy_Builder": StrategyProfile(
        name="Synergy_Builder",
        action_bias={"OPORTUNIDADES": 1.1, "LANZAR": 1.1, "TALENTO": 0.8, "CRECER": 1.2, "FINANZAS": 0.8},
        card_bias={"income": 0.9, "vp": 1.0, "scale": 1.0, "trend": 1.0, "brand": 1.0, "synergy": 1.8, "cheap": 0.8},
    ),
    "Brand_Rush": StrategyProfile(
        name="Brand_Rush",
        action_bias={"OPORTUNIDADES": 0.9, "LANZAR": 1.0, "TALENTO": 0.8, "CRECER": 1.8, "FINANZAS": 0.7},
        card_bias={"income": 0.8, "vp": 1.0, "scale": 0.9, "trend": 1.1, "brand": 1.8, "synergy": 1.1, "cheap": 0.8},
    ),
    "Bootstrap": StrategyProfile(
        name="Bootstrap",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.2, "TALENTO": 1.0, "CRECER": 0.9, "FINANZAS": 0.9},
        card_bias={"income": 1.2, "vp": 0.9, "scale": 0.9, "trend": 0.8, "brand": 0.9, "synergy": 1.0, "cheap": 1.4},
    ),
    "Industry_Focus": StrategyProfile(
        name="Industry_Focus",
        action_bias={"OPORTUNIDADES": 1.1, "LANZAR": 1.1, "TALENTO": 0.9, "CRECER": 1.0, "FINANZAS": 0.8},
        card_bias={"income": 1.0, "vp": 1.0, "scale": 1.0, "trend": 1.0, "brand": 1.0, "synergy": 1.4, "cheap": 0.9},
    ),
    "Early_Blitz": StrategyProfile(
        name="Early_Blitz",
        action_bias={"OPORTUNIDADES": 1.0, "LANZAR": 1.5, "TALENTO": 0.8, "CRECER": 0.9, "FINANZAS": 0.8},
        card_bias={"income": 1.2, "vp": 0.9, "scale": 0.8, "trend": 1.1, "brand": 0.9, "synergy": 0.9, "cheap": 1.5},
    ),
    "Defensive": StrategyProfile(
        name="Defensive",
        action_bias={"OPORTUNIDADES": 0.9, "LANZAR": 1.0, "TALENTO": 1.0, "CRECER": 0.9, "FINANZAS": 1.4},
        card_bias={"income": 1.0, "vp": 1.0, "scale": 0.9, "trend": 0.7, "brand": 0.9, "synergy": 0.9, "cheap": 1.1},
    ),
}


def _profile_for(strategy_name: str) -> StrategyProfile:
    return STRATEGY_PROFILES.get(strategy_name, STRATEGY_PROFILES["Random"])


def _strategy_plan_random(player: Player, market_row: list[BusinessCard], active_mc: MarketCondition | None, cfg: GameConfig) -> QuarterPlan:
    rng = player.rng or random.Random(0)
    actions = list(DEPARTMENTS)
    rng.shuffle(actions)
    reserve = actions.pop()
    ordered = tuple(actions[: cfg.turns_per_quarter])
    focus = tuple(ordered[: min(2, len(ordered))])
    return QuarterPlan(reserve_action=reserve, action_order=ordered, focus_actions=focus)


def _business_desirability(player: Player, card: BusinessCard, profile: StrategyProfile) -> float:
    synergy_score = _synergy_match_score(player, card)
    trend_score = 1.0 if card.tempo == "Tendencia" else 0.0
    scale_score = max(card.income_scaled, card.income) if card.tempo == "Escala" else 0.0
    brand_score = 1.0 if _parse_synergy_category(card.synergy_receives) == "Prestigio" else 0.0
    cheap_score = 1.0 / max(card.cost, 1.0)

    score = (
        profile.card_bias["income"] * card.income
        + profile.card_bias["vp"] * card.valuation_points
        + profile.card_bias["scale"] * scale_score
        + profile.card_bias["trend"] * trend_score * (card.income + 1.0)
        + profile.card_bias["brand"] * brand_score
        + profile.card_bias["synergy"] * synergy_score
        + profile.card_bias["cheap"] * cheap_score * 4.0
    )

    if player.strategy == "Industry_Focus":
        owned = player.industries_played.get(card.industry, 0)
        score += 1.5 * owned
    if player.strategy == "Defensive":
        if card.id in PLATFORM_DEPENDENT_IDS:
            score -= 2.0
        if card.staff_min >= 4:
            score -= 1.0
    if player.strategy == "Bootstrap":
        score += 0.4 * max(0, 10 - card.cost)
    return score


def _trim_hand(player: Player, profile: StrategyProfile) -> None:
    while len(player.hand) > GameConfig().max_hand_size:
        worst_i = min(
            range(len(player.hand)),
            key=lambda i: _business_desirability(player, player.hand[i], profile),
        )
        player.hand.pop(worst_i)


def _department_utility(
    player: Player,
    department: str,
    market_row: list[BusinessCard],
    active_mc: MarketCondition | None,
    profile: StrategyProfile,
    cfg: GameConfig,
) -> float:
    launchable = _launchable_cards(player, cfg)
    investable_scale = any(b.card.tempo == "Escala" and b.growth_cubes < 2 for b in player.businesses)
    investable_trend = any(b.card.tempo == "Tendencia" and not b.refreshed_this_quarter for b in player.businesses)
    promotable = any((not b.has_manager) and b.employees_assigned >= 3 for b in player.businesses)
    low_cash = player.cash < 5

    utility = profile.action_bias[department]
    if department == "OPORTUNIDADES":
        utility += max(0, 4 - len(player.hand)) * 0.6
        if market_row:
            utility += max(_business_desirability(player, c, profile) for c in market_row) / 10.0
    elif department == "LANZAR":
        utility += len(launchable) * 1.1
        if low_cash:
            utility -= 0.8
    elif department == "TALENTO":
        utility += (cfg.max_total_employees - player.total_employees) * 0.08
        utility += 1.2 if promotable else 0.0
        if any(card.staff_min > player.employees_reserve for card in player.hand):
            utility += 0.9
    elif department == "CRECER":
        utility += len(player.boost_hand) * 0.8
        utility += 1.0 if investable_scale else 0.0
        utility += 1.0 if investable_trend else 0.0
        utility += 0.2 * max(0, 10 - player.brand)
        if active_mc and active_mc.id == "MKT-ANY-008":
            utility += 0.7
    elif department == "FINANZAS":
        utility += 1.2 if low_cash else 0.0
        utility += player.active_loan_count * 0.7
        utility += 0.8 if player.cash < 0 else 0.0
    return utility


def _plan_quarter(
    player: Player,
    market_row: list[BusinessCard],
    active_mc: MarketCondition | None,
    cfg: GameConfig,
) -> QuarterPlan:
    if player.strategy == "Random":
        return _strategy_plan_random(player, market_row, active_mc, cfg)

    profile = _profile_for(player.strategy)
    scored = [
        (
            dept,
            _department_utility(player, dept, market_row, active_mc, profile, cfg)
            + 0.05 * player.department_levels[dept],
        )
        for dept in DEPARTMENTS
    ]
    scored.sort(key=lambda item: (item[1], item[0]))
    reserve = scored[0][0]
    ordered = tuple(
        dept for dept, _ in sorted(scored[1:], key=lambda item: item[1], reverse=True)
    )[: cfg.turns_per_quarter]
    focus = tuple(
        dept
        for dept in ordered
        if player.cash >= cfg.focus_cube_cost
    )[:2]
    return QuarterPlan(reserve_action=reserve, action_order=ordered, focus_actions=focus)


def _choose_permanent_upgrade(player: Player, profile: StrategyProfile) -> str | None:
    candidates = [dept for dept in DEPARTMENTS if player.department_levels[dept] < 3]
    if not candidates:
        return None
    return max(candidates, key=lambda dept: (profile.action_bias[dept], -player.department_levels[dept], dept))


def _execute_investment_phase(player: Player, cfg: GameConfig) -> tuple[str | None, tuple[str, ...]]:
    profile = _profile_for(player.strategy)
    upgraded = None
    candidate = _choose_permanent_upgrade(player, profile)
    if candidate is not None:
        current_level = player.department_levels[candidate]
        cost = cfg.upgrade_cost_i_to_ii if current_level == 1 else cfg.upgrade_cost_ii_to_iii
        if player.cash >= cost and player.cash - cost >= 2:
            player.cash -= cost
            player.department_levels[candidate] += 1
            upgraded = candidate

    desired_focus = sorted(DEPARTMENTS, key=lambda dept: profile.action_bias[dept], reverse=True)
    bought: list[str] = []
    for dept in desired_focus:
        if len(bought) >= 2:
            break
        if player.cash >= cfg.focus_cube_cost + 1:
            player.cash -= cfg.focus_cube_cost
            player.focus_bought += 1
            bought.append(dept)
    return upgraded, tuple(bought)


def _launch_cost(card: BusinessCard, city_placed: bool, cfg: GameConfig) -> float:
    return card.cost + (cfg.location_fee if city_placed and card.mode == "Physical" else 0.0)


def _available_city_placement(card: BusinessCard, level: int) -> bool:
    if card.mode != "Physical":
        return False
    if card.tier in {"Premium", "Empire"}:
        return True
    return level >= 2


def _city_bonus_on_launch(active: ActiveBusiness) -> None:
    if not active.city_placed:
        return
    active.break_bonus += 1
    if active.card.industry in {"Professional", "Real Estate"}:
        active.break_bonus += 1


def _launchable_cards(player: Player, config: GameConfig) -> list[tuple[int, BusinessCard]]:
    result = []
    for i, card in enumerate(player.hand):
        if card.staff_min <= player.employees_reserve:
            base_cost = card.cost
            if base_cost <= player.cash + sum(player.pending_launch_discounts):
                result.append((i, card))
    return result


def _draw_cards(deck: list[BusinessCard], amount: int) -> list[BusinessCard]:
    drawn = []
    for _ in range(amount):
        if deck:
            drawn.append(deck.pop())
    return drawn


def _refill_market_row(deck: list[BusinessCard], market_row: list[BusinessCard]) -> None:
    while len(market_row) < MARKET_ROW_SIZE and deck:
        market_row.append(deck.pop())


def _take_best_from_market(player: Player, market_row: list[BusinessCard], profile: StrategyProfile) -> BusinessCard | None:
    if not market_row:
        return None
    best_index = max(
        range(len(market_row)),
        key=lambda i: _business_desirability(player, market_row[i], profile),
    )
    return market_row.pop(best_index)


def _best_market_index(player: Player, market_row: list[BusinessCard], profile: StrategyProfile) -> int | None:
    if not market_row:
        return None
    return max(
        range(len(market_row)),
        key=lambda i: _business_desirability(player, market_row[i], profile),
    )


def _opportunities_action(
    player: Player,
    level: int,
    deck: list[BusinessCard],
    market_row: list[BusinessCard],
) -> None:
    profile = _profile_for(player.strategy)
    chosen: list[BusinessCard] = []

    if level == 1:
        options = _draw_cards(deck, 2)
        if options:
            best = max(options, key=lambda c: _business_desirability(player, c, profile))
            chosen.append(best)
    elif level == 2:
        market_index = _best_market_index(player, market_row, profile)
        market_pick = market_row[market_index] if market_index is not None else None
        draw_options = _draw_cards(deck, 3)
        if market_pick and draw_options:
            drawn_best = max(draw_options, key=lambda c: _business_desirability(player, c, profile))
            if _business_desirability(player, market_pick, profile) >= _business_desirability(player, drawn_best, profile):
                chosen.append(market_row.pop(market_index))
            else:
                draw_options.sort(key=lambda c: _business_desirability(player, c, profile), reverse=True)
                chosen.extend(draw_options[:2])
        elif market_pick:
            chosen.append(market_row.pop(market_index))
        else:
            draw_options.sort(key=lambda c: _business_desirability(player, c, profile), reverse=True)
            chosen.extend(draw_options[:2])
    else:
        market_index = _best_market_index(player, market_row, profile)
        market_pick = market_row[market_index] if market_index is not None else None
        draw_options = _draw_cards(deck, 4 if market_pick is None else 2)
        if market_pick:
            chosen.append(market_row.pop(market_index))
            if draw_options:
                best_draw = max(draw_options, key=lambda c: _business_desirability(player, c, profile))
                chosen.append(best_draw)
        else:
            draw_options.sort(key=lambda c: _business_desirability(player, c, profile), reverse=True)
            chosen.extend(draw_options[:2])

    player.hand.extend(chosen)
    _trim_hand(player, profile)
    _refill_market_row(deck, market_row)


def _choose_launch_target(
    player: Player,
    profile: StrategyProfile,
    cfg: GameConfig,
    board_state: BoardState | None,
    level: int,
) -> tuple[int, BusinessCard, PlacementDecision | None] | None:
    launchable = _launchable_cards(player, cfg)
    if not launchable:
        return None

    scored: list[tuple[int, BusinessCard, PlacementDecision | None, float]] = []
    for hand_index, card in launchable:
        placement = None
        placement_bonus = 0.0
        if cfg.board_enabled and card.mode == "Physical":
            candidates = _placement_candidates(player, card, level, profile, cfg, board_state)
            affordable = [
                candidate
                for candidate in candidates
                if candidate.total_cost <= player.cash + sum(player.pending_launch_discounts)
            ]
            if not affordable:
                player.blocked_city_launches += int(_available_city_placement(card, level))
                continue
            placement = max(affordable, key=lambda candidate: (candidate.score, candidate.area == "Ciudad"))
            placement_bonus = placement.score
        desirability = _business_desirability(player, card, profile) + placement_bonus
        scored.append((hand_index, card, placement, desirability))

    if not scored:
        return None
    if player.strategy == "Random" and player.rng:
        hand_index, card, placement, _ = player.rng.choice(scored)
        return hand_index, card, placement
    hand_index, card, placement, _ = max(scored, key=lambda item: item[3])
    return hand_index, card, placement


def _apply_placement_effects(player: Player, active: ActiveBusiness, placement: PlacementDecision | None) -> None:
    if placement is None:
        return
    if placement.area == "Ciudad":
        player.city_launches += 1
        if player.first_city_frame is None:
            player.first_city_frame = placement.frame_name
    else:
        player.barrio_launches += 1

    for bonus_type in placement.slot_bonus_types:
        player.slot_bonus_triggers += 1
        if bonus_type == "Trafico":
            active.break_bonus += 1
        elif bonus_type == "Prestigio":
            player.brand = min(18, player.brand + 1)

    if placement.district_affinity_match:
        player.district_affinity_triggers += 1
        active.break_bonus += 1


def _execute_launch(
    player: Player,
    hand_index: int,
    level: int,
    cfg: GameConfig,
    active_mc: MarketCondition | None,
    board_state: BoardState | None,
    placement: PlacementDecision | None,
) -> None:
    card = player.hand.pop(hand_index)
    if cfg.board_enabled and card.mode == "Physical":
        if placement is None:
            player.hand.insert(hand_index, card)
            return
        city_placed = placement.city_placed
        launch_cost = placement.total_cost
    else:
        city_placed = _available_city_placement(card, level) and (
            card.industry in CITY_FRIENDLY_INDUSTRIES or card.tier in {"Premium", "Empire"}
        )
        if city_placed and player.cash + sum(player.pending_launch_discounts) < _launch_cost(card, True, cfg):
            city_placed = False
        launch_cost = _launch_cost(card, city_placed, cfg)
    while player.pending_launch_discounts and launch_cost > 0:
        launch_cost = max(0.0, launch_cost - player.pending_launch_discounts.pop(0))
    if active_mc and active_mc.id == "MKT-ANY-003":
        gives = _parse_synergy_category(card.synergy_gives)
        receives = _parse_synergy_category(card.synergy_receives)
        if "Suministro" in {gives, receives}:
            launch_cost += 1.0

    player.cash -= launch_cost
    player.employees_reserve -= card.staff_min
    trend_bonus = 2 if card.tempo == "Tendencia" else 0
    active = ActiveBusiness(
        card=card,
        employees_assigned=card.staff_min,
        city_placed=city_placed,
        placement_area=placement.area if placement else ("Ciudad" if city_placed else "Barrio" if card.mode == "Physical" else None),
        placement_frame=placement.frame_name if placement else None,
        placement_slot_ids=placement.slot_ids if placement else (),
        placement_bonus_types=placement.slot_bonus_types if placement else (),
        district_affinity_match=placement.district_affinity_match if placement else False,
        trend_bonus=trend_bonus,
    )
    if cfg.board_enabled and placement is not None and board_state is not None:
        _occupy_slots(board_state, placement.slot_ids, card.id)
        _apply_placement_effects(player, active, placement)
    else:
        _city_bonus_on_launch(active)
    player.businesses.append(active)
    player.cards_played_ids.append(card.id)
    player.industries_played[card.industry] = player.industries_played.get(card.industry, 0) + 1

    if level >= 3:
        if active.card.tempo == "Tendencia":
            active.refreshed_this_quarter = True
            active.trend_bonus = 2
        elif active.card.tempo == "Escala":
            active.growth_cubes = min(2, active.growth_cubes + 1)
        else:
            active.break_bonus += 1

    _resolve_synergies_on_launch(player, active)


def _hire_employees(player: Player, amount: int, cfg: GameConfig) -> int:
    hired = 0
    while hired < amount and player.total_employees < cfg.max_total_employees:
        player.total_employees += 1
        player.employees_reserve += 1
        hired += 1
    return hired


def _promote_manager(player: Player) -> bool:
    candidates = [
        biz
        for biz in player.businesses
        if not biz.has_manager and biz.employees_assigned >= 3
    ]
    if not candidates:
        return False
    target = max(candidates, key=lambda biz: (biz.card.staff_min, biz.card.valuation_points))
    target.has_manager = True
    target.employees_assigned -= 2
    player.employees_reserve += 2
    return True


def _talent_action(player: Player, level: int, cfg: GameConfig, active_mc: MarketCondition | None) -> None:
    if active_mc and active_mc.id == "MKT-ANY-005":
        player.cash -= 1

    if level == 1:
        if not _promote_manager(player):
            _hire_employees(player, 1, cfg)
    elif level == 2:
        if not _promote_manager(player):
            _hire_employees(player, 2, cfg)
        else:
            _hire_employees(player, 1, cfg)
    else:
        promoted = _promote_manager(player)
        _hire_employees(player, 2 if not promoted else 1, cfg)


def _best_growth_target(player: Player, prefer_scale: bool) -> int | None:
    candidates = []
    for i, biz in enumerate(player.businesses):
        if prefer_scale and biz.card.tempo == "Escala" and biz.growth_cubes < 2:
            candidates.append((i, biz))
        elif not prefer_scale and biz.card.tempo == "Tendencia":
            candidates.append((i, biz))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1].card.valuation_points)[0]


def _play_best_boost(player: Player, deck: list[BusinessCard], cfg: GameConfig) -> bool:
    for bi, boost in enumerate(player.boost_hand):
        target = _best_boost_target(player, boost)
        if target >= 0:
            boost = player.boost_hand.pop(bi)
            _resolve_boost(player, boost, target, deck, cfg)
            return True
    return False


def _grow_action(
    player: Player,
    level: int,
    deck: list[BusinessCard],
    cfg: GameConfig,
    active_mc: MarketCondition | None,
    consumer_confidence_used: dict[str, bool],
) -> None:
    played_boost = False
    if cfg.enable_boosts and player.boost_hand:
        played_boost = _play_best_boost(player, deck, cfg)

    player.brand = min(18, player.brand + 1)
    if active_mc and active_mc.id == "MKT-ANY-008" and level >= 2 and not consumer_confidence_used.get(player.name):
        player.brand = min(18, player.brand + 1)
        consumer_confidence_used[player.name] = True

    if level >= 2:
        trend_i = _best_growth_target(player, prefer_scale=False)
        if trend_i is not None:
            player.businesses[trend_i].trend_bonus = 2
            player.businesses[trend_i].refreshed_this_quarter = True

    if level >= 3:
        scale_i = _best_growth_target(player, prefer_scale=True)
        if scale_i is not None:
            player.businesses[scale_i].growth_cubes = min(2, player.businesses[scale_i].growth_cubes + 1)
        elif not played_boost:
            player.brand = min(18, player.brand + 1)


def _take_loan(player: Player, loan_type: str) -> bool:
    if loan_type == "Bridge":
        bridge_count = sum(1 for l in player.loans if l.loan_type == "Bridge")
        if player.active_loan_count < 2 and bridge_count < 1:
            player.loans.append(Loan("Bridge", 5.0, 1.0))
            player.cash += 5.0
            return True
        return False
    if player.active_loan_count < 2:
        player.loans.append(Loan("Growth", 10.0, 2.0))
        player.cash += 10.0
        return True
    return False


def _repay_loan(player: Player) -> bool:
    if not player.loans:
        return False
    loan_index = min(range(len(player.loans)), key=lambda idx: (-player.loans[idx].interest, -player.loans[idx].amount))
    loan = player.loans[loan_index]
    if player.cash >= loan.amount:
        player.cash -= loan.amount
        player.loans.pop(loan_index)
        return True
    return False


def _refinance_loan(player: Player) -> bool:
    if not player.loans:
        return False
    loan_index = max(range(len(player.loans)), key=lambda idx: player.loans[idx].interest)
    loan = player.loans[loan_index]
    if loan.interest > 0 and not loan.refinanced_this_quarter:
        loan.interest = max(0.0, loan.interest - 1.0)
        loan.refinanced_this_quarter = True
        return True
    return False


def _sell_business_to_bank(player: Player) -> bool:
    if not player.businesses:
        return False
    weakest_idx = min(
        range(len(player.businesses)),
        key=lambda idx: (
            player.businesses[idx].current_income,
            player.businesses[idx].card.valuation_points,
        ),
    )
    sold = player.businesses.pop(weakest_idx)
    player.cash += max(1.0, sold.card.exit_value / 2.0)
    player.employees_reserve += sold.employees_assigned
    return True


def _finance_action(player: Player, level: int, cfg: GameConfig) -> None:
    operations_done = 0
    target_ops = 2 if level >= 3 else 1

    while operations_done < target_ops:
        if player.cash < 4 and cfg.enable_loans:
            if _take_loan(player, "Growth" if level >= 2 else "Bridge"):
                operations_done += 1
                if level < 2:
                    break
                continue
        if player.loans and player.cash >= max(7.0, player.loans[0].amount + 3):
            if _repay_loan(player):
                operations_done += 1
                continue
        if player.loans and _refinance_loan(player):
            operations_done += 1
            if level == 1:
                break
            continue
        if level >= 2 and player.cash < 3 and _sell_business_to_bank(player):
            operations_done += 1
            continue
        if level >= 2 and player.cash < 5:
            player.cash += 2
            operations_done += 1
            continue
        break


def _resolve_department(
    player: Player,
    department: str,
    level: int,
    deck: list[BusinessCard],
    market_row: list[BusinessCard],
    cfg: GameConfig,
    active_mc: MarketCondition | None,
    consumer_confidence_used: dict[str, bool],
    board_state: BoardState | None,
) -> None:
    if department == "OPORTUNIDADES":
        _opportunities_action(player, level, deck, market_row)
        return
    if department == "LANZAR":
        profile = _profile_for(player.strategy)
        target = _choose_launch_target(player, profile, cfg, board_state, level)
        if target is None:
            _opportunities_action(player, 1, deck, market_row)
            return
        _execute_launch(player, target[0], level, cfg, active_mc, board_state, target[2])
        return
    if department == "TALENTO":
        _talent_action(player, level, cfg, active_mc)
        return
    if department == "CRECER":
        _grow_action(player, level, deck, cfg, active_mc, consumer_confidence_used)
        return
    if department == "FINANZAS":
        _finance_action(player, level, cfg)


def _giro_candidate_utility(
    player: Player,
    action: str,
    market_row: list[BusinessCard],
    active_mc: MarketCondition | None,
    cfg: GameConfig,
) -> float:
    return _department_utility(player, action, market_row, active_mc, _profile_for(player.strategy), cfg)


def _maybe_use_giro(
    player: Player,
    plan: QuarterPlan,
    current_action: str,
    current_index: int,
    market_row: list[BusinessCard],
    active_mc: MarketCondition | None,
    cfg: GameConfig,
) -> str:
    if player.giro_uses >= 1 or player.cash < cfg.giro_cost:
        return current_action

    current_utility = _giro_candidate_utility(player, current_action, market_row, active_mc, cfg)
    reserve_utility = _giro_candidate_utility(player, plan.reserve_action, market_row, active_mc, cfg)
    if reserve_utility <= current_utility + 1.1:
        return current_action

    player.cash -= cfg.giro_cost
    player.giro_uses += 1
    remaining = list(plan.action_order)
    remaining[current_index] = plan.reserve_action
    reserve_replacement = current_action
    plan.action_order = tuple(remaining)
    plan.reserve_action = reserve_replacement
    return plan.action_order[current_index]


def _reset_quarter_flags(player: Player) -> None:
    for biz in player.businesses:
        biz.refreshed_this_quarter = False
    for loan in player.loans:
        loan.refinanced_this_quarter = False


STRATEGIES: dict[str, Callable[[Player, list[BusinessCard], MarketCondition | None, GameConfig], QuarterPlan]] = {
    "Random": _strategy_plan_random,
    "Greedy_VP": _plan_quarter,
    "Cash_Machine": _plan_quarter,
    "Stable_Heavy": _plan_quarter,
    "Scale_Rush": _plan_quarter,
    "Trend_Surfer": _plan_quarter,
    "Balanced_Tempo": _plan_quarter,
    "Premium_Builder": _plan_quarter,
    "Leveraged_Growth": _plan_quarter,
    "Boost_Opportunist": _plan_quarter,
    "Synergy_Builder": _plan_quarter,
    "Brand_Rush": _plan_quarter,
    "Bootstrap": _plan_quarter,
    "Industry_Focus": _plan_quarter,
    "Early_Blitz": _plan_quarter,
    "Defensive": _plan_quarter,
}

DEFAULT_TOURNAMENT_STRATEGIES = [
    "Random",
    "Greedy_VP",
    "Cash_Machine",
    "Scale_Rush",
    "Trend_Surfer",
    "Balanced_Tempo",
    "Synergy_Builder",
    "Bootstrap",
]


class GameEngine:
    """Quarterly Business Empire simulator aligned with the active v3 build."""

    def __init__(self, cards_df: pd.DataFrame, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.business_cards = self._df_to_cards(cards_df)
        self.boost_cards = self._df_to_boost_cards(cards_df)
        self.market_conditions = self._df_to_market_conditions(cards_df)

    def _df_to_cards(self, df: pd.DataFrame) -> list[BusinessCard]:
        biz = df[df["type"].astype(str).str.strip().str.title() == "Business"].copy()
        cards = []
        for _, row in biz.iterrows():
            tempo = str(row.get("tempo", "Estable"))
            if tempo in ("nan", "", "None", "<NA>"):
                tempo = "Estable"
            mode = str(row.get("mode", "Physical"))
            if mode in ("nan", "", "None", "<NA>"):
                mode = "Physical"
            tags_raw = row.get("tags", [])
            cards.append(
                BusinessCard(
                    id=str(row.get("id", "")),
                    name=str(row.get("name", "")),
                    industry=str(row.get("industry", "Unknown")),
                    tier=str(row.get("tier", "Unknown")),
                    cost=float(row.get("cost", 0)),
                    income=float(row.get("income", 0)),
                    valuation_points=float(row.get("valuation_points", 0)),
                    exit_value=_safe_float(row.get("exit_value", 0)),
                    tempo=tempo,
                    mode=mode,
                    staff_min=max(_safe_int(row.get("staff_min", 1), 1), 0),
                    income_scaled=_safe_float(row.get("income_scaled", 0)),
                    synergy_gives=str(row.get("synergy_gives", "") or ""),
                    synergy_receives=str(row.get("synergy_receives", "") or ""),
                    staff_opt=max(_safe_int(row.get("staff_opt", 0)), 0),
                    income_opt=_safe_float(row.get("income_opt", 0)),
                    tags=tags_raw if isinstance(tags_raw, list) else [],
                    time_delay=_safe_int(row.get("time_delay", 0), 0),
                    effort=_safe_int(row.get("effort", 0), 0),
                    likelihood=_safe_int(row.get("likelihood", 10), 10),
                )
            )
        return cards

    def _df_to_boost_cards(self, df: pd.DataFrame) -> list[BoostCard]:
        boosts = df[df["type"].astype(str).str.strip().str.title() == "Boost"].copy()
        cards = []
        for _, row in boosts.iterrows():
            tags_raw = row.get("tags", [])
            cards.append(
                BoostCard(
                    id=str(row.get("id", "")),
                    name=str(row.get("name", "")),
                    tags=tags_raw if isinstance(tags_raw, list) else [],
                )
            )
        return cards

    def _df_to_market_conditions(self, df: pd.DataFrame) -> list[MarketCondition]:
        mkt = df[df["type"].astype(str).str.strip().str.title() == "Market Condition"].copy()
        cards = []
        for _, row in mkt.iterrows():
            tags_raw = row.get("tags", [])
            cards.append(
                MarketCondition(
                    id=str(row.get("id", "")),
                    name=str(row.get("name", "")),
                    tags=tags_raw if isinstance(tags_raw, list) else [],
                )
            )
        return cards

    def simulate_game(
        self,
        strategy_a: str = "Random",
        strategy_b: str = "Random",
        seed: int | None = None,
    ) -> dict:
        rng = random.Random(seed)
        cfg = self.config

        strat_fn_a = STRATEGIES.get(strategy_a, _plan_quarter)
        strat_fn_b = STRATEGIES.get(strategy_b, _plan_quarter)

        business_deck = copy.deepcopy(self.business_cards)
        boost_deck = copy.deepcopy(self.boost_cards)
        market_deck = copy.deepcopy(self.market_conditions)
        rng.shuffle(business_deck)
        rng.shuffle(boost_deck)
        rng.shuffle(market_deck)

        market_row: list[BusinessCard] = []
        _refill_market_row(business_deck, market_row)

        p1 = Player(
            name="Player_1",
            strategy=strategy_a,
            cash=cfg.starting_cash,
            brand=cfg.starting_brand,
            employees_reserve=cfg.starting_employees,
            total_employees=cfg.starting_employees,
            rng=rng,
        )
        p2 = Player(
            name="Player_2",
            strategy=strategy_b,
            cash=cfg.starting_cash,
            brand=cfg.starting_brand,
            employees_reserve=cfg.starting_employees,
            total_employees=cfg.starting_employees,
            rng=rng,
        )
        board_state = _build_board_state((p1.name, p2.name), cfg, seed if cfg.board_seed is None else cfg.board_seed)

        for player in (p1, p2):
            drawn = _draw_cards(business_deck, cfg.starting_draw_size)
            profile = _profile_for(player.strategy)
            drawn.sort(key=lambda card: _business_desirability(player, card, profile), reverse=True)
            player.hand = drawn[: cfg.starting_hand_size]
            for discarded in drawn[cfg.starting_hand_size :]:
                business_deck.insert(0, discarded)
            if boost_deck:
                player.boost_hand.append(boost_deck.pop())

        first_player = p1
        quarters_elapsed = 0
        total_turns = 0
        game_over = False

        for quarter in range(1, cfg.max_quarters + 1):
            quarters_elapsed = quarter
            active_mc = market_deck.pop() if (cfg.enable_market_conditions and market_deck) else None
            consumer_confidence_used = {p1.name: False, p2.name: False}
            _reset_quarter_flags(p1)
            _reset_quarter_flags(p2)

            if quarter % 2 == 0 and cfg.enable_boosts:
                for player in (p1, p2):
                    if boost_deck:
                        player.boost_hand.append(boost_deck.pop())

            investments = {}
            for player in (p1, p2):
                upgraded, focus = _execute_investment_phase(player, cfg)
                plan_fn = strat_fn_a if player is p1 else strat_fn_b
                plan = plan_fn(player, market_row, active_mc, cfg)
                plan.focus_actions = tuple(
                    dept for dept in plan.focus_actions if dept in focus
                )
                if not plan.focus_actions:
                    plan.focus_actions = focus[:2]
                investments[player.name] = {"upgrade": upgraded, "focus": plan.focus_actions}
                player.quarter_history.append(
                    {
                        "quarter": quarter,
                        "upgrade": upgraded,
                        "reserve": plan.reserve_action,
                        "order": list(plan.action_order),
                        "focus": list(plan.focus_actions),
                        "market_condition": active_mc.id if active_mc else None,
                    }
                )
                if player is p1:
                    plan_a = plan
                else:
                    plan_b = plan

            if p1.brand < p2.brand or (p1.brand == p2.brand and p1.cash < p2.cash):
                first_player = p1
            elif p2.brand < p1.brand or (p1.brand == p2.brand and p2.cash < p1.cash):
                first_player = p2

            turn_order = (p1, p2) if first_player is p1 else (p2, p1)
            plans = {p1.name: plan_a, p2.name: plan_b}
            focus_remaining = {
                p1.name: list(plan_a.focus_actions),
                p2.name: list(plan_b.focus_actions),
            }

            for turn_index in range(cfg.turns_per_quarter):
                for player in turn_order:
                    plan = plans[player.name]
                    department = plan.action_order[turn_index]
                    department = _maybe_use_giro(
                        player,
                        plan,
                        department,
                        turn_index,
                        market_row,
                        active_mc,
                        cfg,
                    )
                    level = player.department_levels[department]
                    if department in focus_remaining[player.name]:
                        level = min(3, level + 1)
                        focus_remaining[player.name].remove(department)
                        player.focus_spent += 1

                    _resolve_department(
                        player,
                        department,
                        level,
                        business_deck,
                        market_row,
                        cfg,
                        active_mc,
                        consumer_confidence_used,
                        board_state,
                    )

                    total_turns += 1
                    player.turns_played = total_turns

            for player in (p1, p2):
                player.cash += player.recurring_income
                for biz in player.businesses:
                    player.cash += biz.break_bonus
                    biz.break_bonus = 0.0

                if cfg.enable_loans:
                    player.cash -= player.loan_interest_cost

                if cfg.enable_market_conditions and active_mc:
                    _resolve_market_condition_at_break(active_mc, player)

                if cfg.enable_synergies:
                    _resolve_synergies_at_break(player)
                    for biz in player.businesses:
                        player.cash += biz.break_bonus
                        biz.break_bonus = 0.0

                algorithm_change_applied = False
                for biz in player.businesses:
                    if biz.card.tempo != "Tendencia":
                        continue
                    if not biz.refreshed_this_quarter:
                        decay = 1
                        if (
                            active_mc
                            and active_mc.id == "MKT-ANY-001"
                            and biz.card.mode == "Digital"
                            and not algorithm_change_applied
                        ):
                            decay = 2
                            algorithm_change_applied = True
                        biz.trend_bonus = max(0, biz.trend_bonus - decay)
                    biz.refreshed_this_quarter = False

                if player.business_count >= cfg.end_business_threshold:
                    game_over = True
                if player.employees_reserve <= 0:
                    game_over = True
                if player.net_profit >= cfg.profit_machine_threshold:
                    game_over = True

            if game_over:
                break

        score_1 = p1.score
        score_2 = p2.score
        if score_1 > score_2:
            winner = "Player_1"
        elif score_2 > score_1:
            winner = "Player_2"
        else:
            winner = "Draw"

        return {
            "winner": winner,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "score_1": round(score_1, 2),
            "score_2": round(score_2, 2),
            "businesses_1": p1.business_count,
            "businesses_2": p2.business_count,
            "cash_1": round(p1.cash, 2),
            "cash_2": round(p2.cash, 2),
            "turns": total_turns,
            "cards_played_1": p1.cards_played_ids.copy(),
            "cards_played_2": p2.cards_played_ids.copy(),
            "industries_1": dict(p1.industries_played),
            "industries_2": dict(p2.industries_played),
            "brand_1": p1.brand,
            "brand_2": p2.brand,
            "net_profit_1": round(p1.net_profit, 2),
            "net_profit_2": round(p2.net_profit, 2),
            "exit_bonus_1": _exit_bonus_lookup(p1.net_profit, p1.brand),
            "exit_bonus_2": _exit_bonus_lookup(p2.net_profit, p2.brand),
            "asset_vp_1": sum(b.card.valuation_points for b in p1.businesses),
            "asset_vp_2": sum(b.card.valuation_points for b in p2.businesses),
            "quarters": quarters_elapsed,
            "loans_1": p1.active_loan_count,
            "loans_2": p2.active_loan_count,
            "debt_1": round(p1.total_debt, 2),
            "debt_2": round(p2.total_debt, 2),
            "reserve_1": plan_a.reserve_action,
            "reserve_2": plan_b.reserve_action,
            "giro_uses_1": p1.giro_uses,
            "giro_uses_2": p2.giro_uses,
            "focus_spent_1": p1.focus_spent,
            "focus_spent_2": p2.focus_spent,
            "board_enabled": cfg.board_enabled,
            "board_seed": board_state.board_seed if board_state else None,
            "starting_zone_1": p1.first_city_frame,
            "starting_zone_2": p2.first_city_frame,
            "city_launches_1": p1.city_launches,
            "city_launches_2": p2.city_launches,
            "barrio_launches_1": p1.barrio_launches,
            "barrio_launches_2": p2.barrio_launches,
            "blocked_city_launches_1": p1.blocked_city_launches,
            "blocked_city_launches_2": p2.blocked_city_launches,
            "slot_bonus_triggers_1": p1.slot_bonus_triggers,
            "slot_bonus_triggers_2": p2.slot_bonus_triggers,
            "district_affinity_triggers_1": p1.district_affinity_triggers,
            "district_affinity_triggers_2": p2.district_affinity_triggers,
            "city_slot_usage": [slot.slot_id for slot in (board_state.city_slots if board_state else []) if slot.occupied_by],
            "city_frame_usage": [slot.frame_name for slot in (board_state.city_slots if board_state else []) if slot.occupied_by],
        }


def run_monte_carlo(
    engine: GameEngine,
    strategy_a: str = "Random",
    strategy_b: str = "Random",
    n_simulations: int = 1000,
    verbose: bool = False,
    seed: int | None = None,
) -> dict:
    results = []
    for i in range(n_simulations):
        game_seed = None if seed is None else _stable_seed(seed, strategy_a, strategy_b, i)
        results.append(engine.simulate_game(strategy_a=strategy_a, strategy_b=strategy_b, seed=game_seed))

    df = pd.DataFrame(results)
    wins_1 = (df["winner"] == "Player_1").sum()
    wins_2 = (df["winner"] == "Player_2").sum()
    draws = (df["winner"] == "Draw").sum()

    summary = {
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "n_simulations": n_simulations,
        "seed": seed,
        "win_rate_a": round(wins_1 / n_simulations, 4),
        "win_rate_b": round(wins_2 / n_simulations, 4),
        "draw_rate": round(draws / n_simulations, 4),
        "avg_score_a": round(df["score_1"].mean(), 2),
        "avg_score_b": round(df["score_2"].mean(), 2),
        "std_score_a": round(df["score_1"].std(), 2),
        "std_score_b": round(df["score_2"].std(), 2),
        "avg_turns": round(df["turns"].mean(), 1),
        "avg_businesses_a": round(df["businesses_1"].mean(), 1),
        "avg_businesses_b": round(df["businesses_2"].mean(), 1),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Monte Carlo: {strategy_a} vs {strategy_b}")
        print(f"Simulations: {n_simulations}")
        print(f"{'=' * 50}")
        print(f"Win rate A ({strategy_a}): {summary['win_rate_a']:.1%}")
        print(f"Win rate B ({strategy_b}): {summary['win_rate_b']:.1%}")
        print(f"Draw rate: {summary['draw_rate']:.1%}")
        print(f"Avg score A: {summary['avg_score_a']}")
        print(f"Avg score B: {summary['avg_score_b']}")
        print(f"Avg turns: {summary['avg_turns']}")

    return {"summary": summary, "results": df}


def summarize_card_outcomes(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return _empty_card_summary()

    records = []
    winner_slots = loser_slots = draw_slots = 0

    for _, row in results_df.iterrows():
        winner = row["winner"]
        for player_name, cards in [("Player_1", row["cards_played_1"]), ("Player_2", row["cards_played_2"])]:
            if winner == "Draw":
                outcome = "draw"
                draw_slots += 1
            elif winner == player_name:
                outcome = "win"
                winner_slots += 1
            else:
                outcome = "loss"
                loser_slots += 1
            for card_id in cards:
                records.append({"card_id": card_id, "outcome": outcome})

    if not records:
        return _empty_card_summary()

    plays = pd.DataFrame(records)
    counts = plays.groupby(["card_id", "outcome"]).size().unstack(fill_value=0).reset_index()
    counts["times_played"] = counts[[c for c in ["win", "loss", "draw"] if c in counts.columns]].sum(axis=1)
    for column in ["win", "loss", "draw"]:
        if column not in counts.columns:
            counts[column] = 0

    total_player_games = len(results_df) * 2
    counts["usage_rate"] = (counts["times_played"] / max(total_player_games, 1)).round(4)
    counts["wins_played"] = counts["win"]
    counts["losses_played"] = counts["loss"]
    counts["draws_played"] = counts["draw"]
    counts["win_deck_rate"] = (counts["wins_played"] / max(winner_slots, 1)).round(4)
    counts["loss_deck_rate"] = (counts["losses_played"] / max(loser_slots, 1)).round(4)
    counts["draw_deck_rate"] = (counts["draws_played"] / max(draw_slots, 1)).round(4)
    counts["win_bias"] = (counts["win_deck_rate"] - counts["loss_deck_rate"]).round(4)

    return counts[
        [
            "card_id",
            "times_played",
            "usage_rate",
            "wins_played",
            "losses_played",
            "draws_played",
            "win_deck_rate",
            "loss_deck_rate",
            "draw_deck_rate",
            "win_bias",
        ]
    ].sort_values(["win_bias", "usage_rate", "times_played"], ascending=[False, False, False]).reset_index(drop=True)


def summarize_board_outcomes(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if results_df.empty or "board_enabled" not in results_df.columns or not bool(results_df["board_enabled"].any()):
        empty_zone = pd.DataFrame(columns=["starting_zone", "games", "wins", "win_rate"])
        empty_usage = pd.DataFrame(columns=["slot_id", "times_used"])
        empty_frame = pd.DataFrame(columns=["frame_name", "times_used"])
        empty_industry = pd.DataFrame(columns=["industry", "starting_zone", "count"])
        return {
            "win_rate_by_starting_zone": empty_zone,
            "slot_usage": empty_usage,
            "frame_usage": empty_frame,
            "industry_zone_matrix": empty_industry,
            "summary": pd.DataFrame([{
                "avg_blocked_city_launches": 0.0,
                "avg_city_launches": 0.0,
                "avg_barrio_launches": 0.0,
                "avg_slot_bonus_triggers": 0.0,
                "avg_district_affinity_triggers": 0.0,
            }]),
        }

    zone_rows = []
    usage_rows = []
    frame_rows = []
    industry_rows = []
    for _, row in results_df.iterrows():
        for player_idx, player_name in (("1", "Player_1"), ("2", "Player_2")):
            zone = row.get(f"starting_zone_{player_idx}")
            if zone:
                zone_rows.append({
                    "starting_zone": zone,
                    "winner": row["winner"] == player_name,
                })
                for industry, count in (row.get(f"industries_{player_idx}") or {}).items():
                    industry_rows.append({
                        "industry": industry,
                        "starting_zone": zone,
                        "count": count,
                    })
        for slot_id in row.get("city_slot_usage", []) or []:
            usage_rows.append({"slot_id": slot_id})
        for frame_name in row.get("city_frame_usage", []) or []:
            frame_rows.append({"frame_name": frame_name})

    zone_df = pd.DataFrame(zone_rows)
    if zone_df.empty:
        win_rate_by_zone = pd.DataFrame(columns=["starting_zone", "games", "wins", "win_rate"])
    else:
        win_rate_by_zone = (
            zone_df.groupby("starting_zone")["winner"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={"count": "games", "sum": "wins", "mean": "win_rate"})
            .sort_values(["win_rate", "games"], ascending=[False, False])
            .reset_index(drop=True)
        )
        win_rate_by_zone["win_rate"] = win_rate_by_zone["win_rate"].round(4)

    slot_usage = (
        pd.DataFrame(usage_rows).value_counts().reset_index(name="times_used")
        if usage_rows else pd.DataFrame(columns=["slot_id", "times_used"])
    )
    frame_usage = (
        pd.DataFrame(frame_rows).value_counts().reset_index(name="times_used")
        if frame_rows else pd.DataFrame(columns=["frame_name", "times_used"])
    )
    industry_zone_matrix = (
        pd.DataFrame(industry_rows).groupby(["industry", "starting_zone"], as_index=False)["count"].sum()
        if industry_rows else pd.DataFrame(columns=["industry", "starting_zone", "count"])
    )

    summary = pd.DataFrame([{
        "avg_blocked_city_launches": round(
            float(
                (
                    results_df["blocked_city_launches_1"].fillna(0)
                    + results_df["blocked_city_launches_2"].fillna(0)
                ).mean() / 2
            ),
            4,
        ),
        "avg_city_launches": round(
            float((results_df["city_launches_1"].fillna(0) + results_df["city_launches_2"].fillna(0)).mean() / 2),
            4,
        ),
        "avg_barrio_launches": round(
            float((results_df["barrio_launches_1"].fillna(0) + results_df["barrio_launches_2"].fillna(0)).mean() / 2),
            4,
        ),
        "avg_slot_bonus_triggers": round(
            float((results_df["slot_bonus_triggers_1"].fillna(0) + results_df["slot_bonus_triggers_2"].fillna(0)).mean() / 2),
            4,
        ),
        "avg_district_affinity_triggers": round(
            float((results_df["district_affinity_triggers_1"].fillna(0) + results_df["district_affinity_triggers_2"].fillna(0)).mean() / 2),
            4,
        ),
    }])

    return {
        "win_rate_by_starting_zone": win_rate_by_zone,
        "slot_usage": slot_usage,
        "frame_usage": frame_usage,
        "industry_zone_matrix": industry_zone_matrix,
        "summary": summary,
    }


def evaluate_strategy_tournament(
    engine: GameEngine,
    strategies: list[str] | None = None,
    n_simulations: int = 1000,
    base_seed: int | None = None,
    include_card_usage: bool = True,
    include_matchup_results: bool = True,
    verbose: bool = True,
) -> dict:
    if strategies is None:
        strategies = DEFAULT_TOURNAMENT_STRATEGIES

    matchup_summaries = []
    matchup_results = []
    total = len(strategies) ** 2
    done = 0

    for sa in strategies:
        for sb in strategies:
            matchup_seed = None if base_seed is None else _stable_seed(base_seed, sa, sb)
            mc = run_monte_carlo(
                engine,
                sa,
                sb,
                n_simulations=n_simulations,
                verbose=False,
                seed=matchup_seed,
            )
            summary = dict(mc["summary"])
            summary["matchup_seed"] = matchup_seed
            matchup_summaries.append(summary)
            if include_matchup_results or include_card_usage:
                results = mc["results"].copy()
                results["matchup_seed"] = matchup_seed
                matchup_results.append(results)
            done += 1
            if verbose and done % 10 == 0:
                print(f"  Progress: {done}/{total} matchups ({done / total:.0%})")

    matchups_df = pd.DataFrame(matchup_summaries)
    results_df = pd.concat(matchup_results, ignore_index=True) if matchup_results else pd.DataFrame()

    non_mirror = matchups_df[matchups_df["strategy_a"] != matchups_df["strategy_b"]].copy()
    if non_mirror.empty:
        balance_score = 0.0
        max_abs_deviation = 0.0
    else:
        balance_score = float(non_mirror["win_rate_a"].std())
        max_abs_deviation = float((non_mirror["win_rate_a"] - 0.5).abs().max())

    strategy_stats = (
        non_mirror.groupby("strategy_a")["win_rate_a"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "strategy_a": "strategy",
                "mean": "avg_win_rate",
                "std": "win_rate_std",
                "count": "matchup_count",
            }
        )
        .sort_values("avg_win_rate", ascending=False)
        .reset_index(drop=True)
    )
    if not strategy_stats.empty:
        strategy_stats["avg_win_rate"] = strategy_stats["avg_win_rate"].round(4)
        strategy_stats["win_rate_std"] = strategy_stats["win_rate_std"].fillna(0).round(4)

    card_usage = summarize_card_outcomes(results_df) if include_card_usage and not results_df.empty else _empty_card_summary()
    board_telemetry = summarize_board_outcomes(results_df) if not results_df.empty else summarize_board_outcomes(pd.DataFrame())

    return {
        "matchups": matchups_df,
        "results": results_df if include_matchup_results else pd.DataFrame(),
        "non_mirror": non_mirror,
        "strategy_stats": strategy_stats,
        "card_usage": card_usage,
        "balance_score": round(balance_score, 4),
        "max_abs_deviation": round(max_abs_deviation, 4),
        "board_telemetry": board_telemetry,
        "strategies": list(strategies),
        "n_simulations": n_simulations,
        "base_seed": base_seed,
    }


def run_all_strategy_matchups(engine, strategies=None, n_simulations=1000, base_seed=None):
    return evaluate_strategy_tournament(
        engine=engine,
        strategies=strategies,
        n_simulations=n_simulations,
        base_seed=base_seed,
        include_card_usage=False,
        include_matchup_results=False,
    )["matchups"]


def analyze_card_usage(results_df):
    usage = summarize_card_outcomes(results_df)
    return usage[["card_id", "times_played", "usage_rate"]].copy()


def main():
    from card_parser import parse_all_cards

    print("Loading card data...")
    df = parse_all_cards()

    engine = GameEngine(df)
    print(f"Business cards: {len(engine.business_cards)}")
    print(f"Boost cards: {len(engine.boost_cards)}")
    print(f"Market Conditions: {len(engine.market_conditions)}")

    single = engine.simulate_game("Balanced_Tempo", "Bootstrap", seed=42)
    print(f"Winner: {single['winner']}")
    print(f"Reserve P1/P2: {single['reserve_1']} / {single['reserve_2']}")
    print(f"Giro uses P1/P2: {single['giro_uses_1']} / {single['giro_uses_2']}")

    mc = run_monte_carlo(engine, "Synergy_Builder", "Balanced_Tempo", n_simulations=200, verbose=True, seed=42)
    print(mc["summary"])


if __name__ == "__main__":
    main()
