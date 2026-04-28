"""
Streamlit front end for the Business Empire balance pipeline.

The app has two tabs. The balance demo runs the same seeded simulator
used by the fast simulation check. The card gallery joins compressed
card thumbnails to the frozen card metadata committed with this repo.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
DATA_PATH = REPO_ROOT / "data" / "cards_dataset.csv"
ASSETS_DIR = APP_DIR / "streamlit_assets" / "cards"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from game_engine_v3 import (  # noqa: E402
    DEFAULT_TOURNAMENT_STRATEGIES,
    GameEngine,
    evaluate_strategy_tournament,
)


DEFAULT_STRATEGY_COUNT = 4
DEFAULT_BASE_SEED = 20260401
MIN_STRATEGIES = 2
GALLERY_COLUMNS = 4
CARD_ID_PATTERN = re.compile(r"^[A-Z]{3}-[A-Z]+-\d{3}$")

HERO_CARDS = [
    ("BUS-FOOD-001", "Coffee Cart"),
    ("BUS-TECH-004", "SaaS Build"),
    ("STF-ANY-012", "Senior Hire"),
    ("BOO-ANY-006", "Brand Boost"),
]


@st.cache_data(show_spinner=False)
def load_cards_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing card dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def build_engine(card_count: int, column_signature: str) -> GameEngine:
    _ = (card_count, column_signature)
    return GameEngine(load_cards_dataframe())


def cards_signature(cards_df: pd.DataFrame) -> tuple[int, str]:
    return len(cards_df), "|".join(cards_df.columns.astype(str))


def clamp_strategies(selected: list[str]) -> list[str]:
    if len(selected) < MIN_STRATEGIES:
        raise ValueError(
            f"Pick at least {MIN_STRATEGIES} strategies so the tournament can compare opponents."
        )
    return list(selected)


@st.cache_data(show_spinner=True)
def run_tournament(
    strategies: tuple[str, ...],
    n_simulations: int,
    base_seed: int,
    include_card_usage: bool,
    include_matchup_results: bool,
) -> dict[str, Any]:
    cards_df = load_cards_dataframe()
    card_count, column_signature = cards_signature(cards_df)
    engine = build_engine(card_count, column_signature)
    started = perf_counter()
    metrics = evaluate_strategy_tournament(
        engine=engine,
        strategies=list(strategies),
        n_simulations=n_simulations,
        base_seed=base_seed,
        include_card_usage=include_card_usage,
        include_matchup_results=include_matchup_results,
        verbose=False,
    )
    return {
        "balance_score": metrics["balance_score"],
        "max_abs_deviation": metrics["max_abs_deviation"],
        "elapsed_seconds": round(float(perf_counter() - started), 4),
        "matchups": metrics["matchups"],
        "non_mirror": metrics["non_mirror"],
        "strategy_stats": metrics["strategy_stats"],
        "card_usage": metrics["card_usage"],
    }


def industry_breakdown(cards_df: pd.DataFrame) -> pd.DataFrame:
    business = cards_df[cards_df["type"].astype(str).str.title() == "Business"]
    counts = Counter(business["industry"].fillna("Unknown"))
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return pd.DataFrame(rows, columns=["industry", "business_cards"])


def matchup_heatmap_data(non_mirror: pd.DataFrame) -> pd.DataFrame:
    if non_mirror.empty:
        return pd.DataFrame()
    pivot = non_mirror.pivot_table(
        index="strategy_a",
        columns="strategy_b",
        values="win_rate_a",
        aggfunc="mean",
    )
    return pivot.round(3)


@st.cache_data(show_spinner=False)
def discover_card_assets() -> pd.DataFrame:
    cards_df = load_cards_dataframe()
    metadata = cards_df.set_index("id") if "id" in cards_df.columns else pd.DataFrame()
    rows: list[dict[str, Any]] = []

    if ASSETS_DIR.exists():
        image_paths = sorted(ASSETS_DIR.glob("*.png"))
    else:
        image_paths = []

    for path in image_paths:
        card_id = path.stem
        if not CARD_ID_PATTERN.match(card_id):
            continue
        meta = metadata.loc[card_id].to_dict() if card_id in metadata.index else {}
        rows.append(
            {
                "id": card_id,
                "image_path": str(path),
                "name": meta.get("name", card_id),
                "type": meta.get("type", "Unknown"),
                "industry": meta.get("industry", "Any"),
                "tier": meta.get("tier", ""),
                "cost": meta.get("cost", None),
                "income": meta.get("income", None),
                "valuation_points": meta.get("valuation_points", None),
                "tags": meta.get("tags", ""),
            }
        )

    columns = [
        "id",
        "image_path",
        "name",
        "type",
        "industry",
        "tier",
        "cost",
        "income",
        "valuation_points",
        "tags",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["type", "industry", "id"]).reset_index(drop=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --be-bg: #0f1210;
            --be-panel: #171b18;
            --be-panel-soft: #1f241f;
            --be-border: rgba(226, 215, 188, 0.18);
            --be-text: #f8f1df;
            --be-muted: #d6cbb2;
            --be-accent: #f0c15b;
            --be-green: #76c893;
        }
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(118, 200, 147, 0.16), transparent 28rem),
                radial-gradient(circle at 90% 12%, rgba(240, 193, 91, 0.12), transparent 26rem),
                var(--be-bg);
            color: var(--be-text);
        }
        html,
        body,
        main,
        .main,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stElementContainer"] {
            background-color: transparent;
            color: var(--be-text);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(118, 200, 147, 0.16), transparent 28rem),
                radial-gradient(circle at 90% 12%, rgba(240, 193, 91, 0.12), transparent 26rem),
                var(--be-bg);
        }
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 3rem;
            background: transparent;
        }
        h1, h2, h3, h4, h5, h6,
        p, label, span, div, li,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            color: var(--be-text);
        }
        [data-testid="stCaptionContainer"],
        .stCaption,
        small {
            color: var(--be-muted);
        }
        section[data-testid="stSidebar"] {
            background: #121511;
            border-right: 1px solid var(--be-border);
        }
        section[data-testid="stSidebar"] * {
            color: var(--be-text);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.1rem;
            border-bottom: 1px solid var(--be-border);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--be-muted);
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            color: var(--be-accent);
        }
        [data-testid="stMetric"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"] {
            background: rgba(23, 27, 24, 0.88);
            border: 1px solid var(--be-border);
            border-radius: 10px;
        }
        [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
        }
        [data-testid="stMetric"] {
            color: var(--be-text);
        }
        [data-testid="stMetric"] *,
        [data-testid="stExpander"] *,
        [data-testid="stDataFrame"] * {
            color: var(--be-text);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"] {
            color: var(--be-muted);
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
        }
        div[data-testid="stImageContainer"] img {
            border-radius: 10px;
            border: 1px solid var(--be-border);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.32);
        }
        div[data-testid="stImageCaption"] {
            color: var(--be-muted);
        }
        input,
        textarea,
        button,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background-color: #101310;
            color: var(--be-text);
            border-color: var(--be-border);
        }
        [data-baseweb="tag"] {
            background: rgba(240, 193, 91, 0.18);
            color: var(--be-text);
        }
        .stAlert {
            background-color: rgba(240, 193, 91, 0.12);
            color: var(--be-text);
        }
        a {
            color: var(--be-green);
        }
        .be-card-caption {
            color: var(--be-muted);
            font-size: 0.86rem;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_cards() -> None:
    available = []
    for card_id, label in HERO_CARDS:
        candidate = ASSETS_DIR / f"{card_id}.png"
        if candidate.exists():
            available.append((card_id, label, candidate))
    if not available:
        return

    columns = st.columns(len(available))
    for column, (card_id, label, path) in zip(columns, available):
        with column:
            st.image(str(path), width="stretch", caption=f"{label} - {card_id}")


def render_balance_tab(cards_df: pd.DataFrame) -> None:
    render_hero_cards()

    with st.sidebar:
        st.header("Tournament configuration")
        strategies = st.multiselect(
            "Strategies",
            options=list(DEFAULT_TOURNAMENT_STRATEGIES),
            default=list(DEFAULT_TOURNAMENT_STRATEGIES[:DEFAULT_STRATEGY_COUNT]),
            help="Each pair plays a seeded matchup. Pick at least two strategies.",
        )
        n_simulations = st.slider(
            "Simulations per matchup",
            min_value=1,
            max_value=20,
            value=1,
            help="Higher values are slower but make the balance score more stable.",
        )
        base_seed = st.number_input(
            "Base seed",
            min_value=1,
            max_value=99_999_999,
            value=DEFAULT_BASE_SEED,
            step=1,
            help="The default seed matches src/live_smoke_demo.py.",
        )
        include_matchup_results = st.checkbox(
            "Include matchup matrix",
            value=True,
            help="Required for the heatmap below.",
        )
        include_card_usage = st.checkbox(
            "Include card-usage stats",
            value=False,
            help="Slower; aggregates per-card outcomes across matchups.",
        )

    if len(strategies) < MIN_STRATEGIES:
        st.warning(f"Pick at least {MIN_STRATEGIES} strategies to run a tournament.")
        return

    result = run_tournament(
        strategies=tuple(strategies),
        n_simulations=int(n_simulations),
        base_seed=int(base_seed),
        include_card_usage=include_card_usage,
        include_matchup_results=include_matchup_results,
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Balance score", f"{result['balance_score']:.4f}", help="Lower is better.")
    metric_b.metric("Max deviation", f"{result['max_abs_deviation']:.4f}")
    metric_c.metric("Tournament time", f"{result['elapsed_seconds']:.2f}s")
    metric_d.metric("Cards parsed", f"{len(cards_df)}")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Strategy stats")
        st.dataframe(result["strategy_stats"], hide_index=True, width="stretch")
    with right:
        st.subheader("Card pool by industry")
        st.dataframe(industry_breakdown(cards_df), hide_index=True, width="stretch")

    if include_matchup_results:
        st.subheader("Matchup heatmap")
        heatmap = matchup_heatmap_data(result["non_mirror"])
        if heatmap.empty:
            st.info("No non-mirror matchups to plot. Pick at least two strategies.")
        else:
            st.dataframe(
                heatmap.style.background_gradient(cmap="RdYlGn", vmin=0.0, vmax=1.0).format("{:.0%}"),
                width="stretch",
            )
    else:
        st.caption("Enable the matchup matrix in the sidebar to see the heatmap.")

    with st.expander("How to read this tab"):
        st.markdown(
            """
            This tab runs the same simulator used by `src/live_smoke_demo.py`
            and the deck metrics.

            - Change the seed to test reproducibility.
            - Widen the strategy mix to test a broader metagame.
            - Increase simulations per matchup to trade speed for stability.

            The machine-learning pieces are in the ridge notebook and the
            Expert bot scorer. This app shows the simulation results that
            those models are built around.
            """
        )


def render_card_gallery() -> None:
    st.subheader("Card gallery")
    st.caption(
        "Generated card art joined to the parsed metadata in cards_dataset.csv. "
        "Filter by type, industry, or name."
    )

    catalog = discover_card_assets()
    if catalog.empty:
        st.info("No card thumbnails were found in app/streamlit_assets/cards/.")
        return

    types = sorted(catalog["type"].dropna().astype(str).unique().tolist())
    industries = sorted(catalog["industry"].dropna().astype(str).unique().tolist())

    filter_a, filter_b, filter_c = st.columns([1.0, 1.0, 1.4])
    with filter_a:
        chosen_types = st.multiselect("Type", options=types, default=types)
    with filter_b:
        chosen_industries = st.multiselect("Industry", options=industries, default=industries)
    with filter_c:
        search_term = st.text_input("Search by name or ID", value="").strip().lower()

    filtered = catalog[
        catalog["type"].astype(str).isin(chosen_types)
        & catalog["industry"].astype(str).isin(chosen_industries)
    ]
    if search_term:
        haystack = (
            filtered["id"].astype(str).str.lower()
            + " "
            + filtered["name"].fillna("").astype(str).str.lower()
        )
        filtered = filtered[haystack.str.contains(search_term, regex=False)]

    counter_a, counter_b, counter_c = st.columns(3)
    counter_a.metric("Cards shown", len(filtered))
    counter_b.metric("Industries", filtered["industry"].nunique() if not filtered.empty else 0)
    counter_c.metric("Types", filtered["type"].nunique() if not filtered.empty else 0)

    if filtered.empty:
        st.info("No cards match this filter.")
        return

    rows = filtered.to_dict(orient="records")
    for chunk_start in range(0, len(rows), GALLERY_COLUMNS):
        columns = st.columns(GALLERY_COLUMNS)
        for column, card in zip(columns, rows[chunk_start : chunk_start + GALLERY_COLUMNS]):
            with column:
                st.image(card["image_path"], width="stretch")
                title = card["name"] if card["name"] != card["id"] else card["id"]
                st.markdown(f"**{title}**  \n`{card['id']}`")

                badge_parts = [
                    str(card[key])
                    for key in ("type", "industry", "tier")
                    if card.get(key) not in (None, "", "Any", "Unknown") and not pd.isna(card.get(key))
                ]
                if badge_parts:
                    st.caption(" - ".join(badge_parts))

                metric_parts: list[str] = []
                for label, key in (("Cost", "cost"), ("Income", "income"), ("VP", "valuation_points")):
                    value = card.get(key)
                    if value not in (None, "") and not pd.isna(value):
                        metric_parts.append(f"{label} {value:g}")
                if metric_parts:
                    st.caption(" - ".join(metric_parts))


def render_app() -> None:
    st.set_page_config(page_title="Business Empire - Live Balance Demo", layout="wide")
    inject_styles()
    st.title("Business Empire - Live Balance Demo")
    st.caption(
        "Run a seeded balance check or inspect generated card art "
        "joined to the parsed dataset."
    )

    cards_df = load_cards_dataframe()
    balance_tab, gallery_tab = st.tabs(["Balance demo", "Card gallery"])
    with balance_tab:
        render_balance_tab(cards_df)
    with gallery_tab:
        render_card_gallery()


if __name__ == "__main__":
    render_app()
