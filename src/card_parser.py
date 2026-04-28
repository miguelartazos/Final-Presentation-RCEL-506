"""
card_parser.py — Business Empire Card Data Pipeline
====================================================
Parses YAML frontmatter from all card .md files in 04-Cards/ and builds
a structured pandas DataFrame for analysis and Monte Carlo simulation.

Author: Maiky
Project: Business Empire — Data Science for Board Game Balancing
"""

import ast
import json
import os
import re
import pandas as pd
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback is exercised in local envs without PyYAML
    yaml = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base project directory (auto-detected relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # BusinessEmpire/
CARDS_DIR = PROJECT_ROOT / "04-Cards"
OUTPUT_CSV = SCRIPT_DIR / "cards_dataset.csv"

# All expected YAML fields (superset across card types)
EXPECTED_FIELDS = [
    "type", "name", "id", "industry", "tier",
    "cost", "income", "valuation_points",
    "tags", "requirements",
    "immediate_effect", "ongoing_effect",
    "time_delay", "effort", "likelihood",
    "synergies", "flavor_text",
    "status", "version",
    "created_date", "last_modified",
    "balance_notes",
    "mode", "exit_value", "tempo",
    "staff_min", "staff_opt",
    "income_scaled", "income_opt",
    "trend_track",
    "synergy_gives", "synergy_receives",
    "license", "location_requirement", "special_rule",
    # Optional fields (Staff only)
    "passive_ability", "upkeep",
]


# ---------------------------------------------------------------------------
# YAML Frontmatter Parser
# ---------------------------------------------------------------------------

def _parse_simple_yaml(frontmatter: str) -> dict | None:
    """
    Parse the limited key/value YAML subset used in legacy card files.
    This fallback exists so the data pipeline does not depend on PyYAML.
    """
    data: dict[str, object] = {}
    for raw_line in frontmatter.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value == "":
            parsed: object = ""
        elif value in {"true", "True"}:
            parsed = True
        elif value in {"false", "False"}:
            parsed = False
        elif value in {"null", "None"}:
            parsed = None
        else:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value.strip('"').strip("'")

        data[key] = parsed

    return data if data else None


def parse_yaml_frontmatter(filepath: str) -> dict | None:
    """
    Extract YAML frontmatter from a markdown file.
    Frontmatter is delimited by --- at the start and end.
    Returns the parsed dict, or None if parsing fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: could not read {filepath}: {e}")
        return None

    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        print(f"  Warning: no YAML frontmatter found in {filepath}")
        return None

    raw_frontmatter = match.group(1)

    try:
        data = json.loads(raw_frontmatter)
    except json.JSONDecodeError:
        data = None

    if data is None and yaml is not None:
        try:
            data = yaml.safe_load(raw_frontmatter)
        except yaml.YAMLError as e:
            print(f"  Warning: YAML parse error in {filepath}: {e}")
            return None

    if data is None:
        data = _parse_simple_yaml(raw_frontmatter)

    if not isinstance(data, dict):
        print(f"  Warning: frontmatter is not a dict in {filepath}")
        return None

    # Add source file path for traceability
    data["_source_file"] = str(filepath)
    return data


# ---------------------------------------------------------------------------
# Card Discovery & Parsing
# ---------------------------------------------------------------------------

def discover_card_files(cards_dir: Path = CARDS_DIR) -> list[Path]:
    """
    Recursively find all .md files under the cards directory,
    excluding the Card-Template.md file.
    """
    card_files = []
    for root, dirs, files in os.walk(cards_dir):
        for fname in sorted(files):
            if fname.endswith(".md") and fname != "Card-Template.md":
                card_files.append(Path(root) / fname)
    return card_files


def parse_all_cards(cards_dir: Path = CARDS_DIR) -> pd.DataFrame:
    """
    Parse all card files and return a cleaned pandas DataFrame.

    If the source markdown directory is unavailable (for example when this
    module is shipped without the 04-Cards/ tree), fall
    back to the frozen cards_dataset.csv snapshot so downstream code still
    works on a fresh clone.
    """
    if not cards_dir.exists():
        for csv_path in (SCRIPT_DIR / "cards_dataset.csv",
                         SCRIPT_DIR.parent / "data" / "cards_dataset.csv",
                         SCRIPT_DIR.parent / "cards_dataset.csv"):
            if csv_path.exists():
                print(f"Cards directory {cards_dir} not found; loading frozen CSV at {csv_path}")
                return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Cards directory not found at {cards_dir} and no cards_dataset.csv fallback was located."
        )
    card_files = discover_card_files(cards_dir)
    print(f"Found {len(card_files)} card files in {cards_dir}")

    records = []
    errors = []

    for filepath in card_files:
        data = parse_yaml_frontmatter(str(filepath))
        if data is not None:
            records.append(data)
        else:
            errors.append(str(filepath))

    if errors:
        print(f"\n{len(errors)} files failed to parse:")
        for e in errors:
            print(f"  - {e}")

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} cards successfully")

    # --- Clean & type-cast ---
    numeric_cols = ["cost", "income", "valuation_points", "time_delay", "effort", "likelihood"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Fill optional fields
    if "upkeep" not in df.columns:
        df["upkeep"] = 0
    else:
        df["upkeep"] = pd.to_numeric(df["upkeep"], errors="coerce").fillna(0)

    if "passive_ability" not in df.columns:
        df["passive_ability"] = ""

    # Coerce v2 numeric fields (allow NaN for non-Business cards)
    v2_numeric_cols = ["staff_min", "staff_opt", "income_scaled", "income_opt", "exit_value"]
    for col in v2_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Compute derived features ---
    df = compute_derived_features(df)

    return df


# ---------------------------------------------------------------------------
# Derived Features (Hormozi Value Score, ROI, Efficiency)
# ---------------------------------------------------------------------------

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns for analysis.

    Legacy metrics (v1, kept for CSV reference — retired from optimizer):
    - hormozi_value, roi, efficiency, net_income, card_category,
      tag_count, requirement_count, synergy_count

    Tempo-aware metrics (v2):
    - total_launch_cost, staff_hire_cost, max_income, avg_income,
      effective_roi, payback_breaks, synergy_output, synergy_input,
      synergy_links, complexity_score, hormozi_v2
    """
    is_business = df["type"].astype(str).str.strip().str.title() == "Business"

    # --- Legacy metrics (preserved for backwards compatibility) -----------

    # Hormozi Value Score
    # NOTE: time_delay=0 for all current cards, so this collapses to
    # VP * likelihood / max(effort, 1).  See hormozi_v2 below for a
    # cost-aware alternative.
    delay_effort = (df["time_delay"] * df["effort"]).clip(lower=1)
    df["hormozi_value"] = (df["valuation_points"] * df["likelihood"]) / delay_effort
    df["hormozi_value"] = df["hormozi_value"].round(2)

    # Return on Investment
    cost_floor = df["cost"].clip(lower=1)
    df["roi"] = (df["income"] / cost_floor).round(3)

    # Combined Efficiency (income + VP per cost unit)
    df["efficiency"] = ((df["income"] + df["valuation_points"]) / cost_floor).round(3)

    # Net Income per Break
    df["net_income"] = df["income"] - df["upkeep"]

    # Card category (simplified type for analysis)
    df["card_category"] = df["type"].str.strip().str.title()

    # Tag count (complexity proxy)
    df["tag_count"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Requirement count (barrier to play)
    df["requirement_count"] = df["requirements"].apply(
        lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) or x == "" else 1)
    )

    # Legacy synergy count (from old synergies[] array)
    df["synergy_count"] = df["synergies"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # --- v2 Synergy metrics -----------------------------------------------

    def _has_content(val) -> int:
        """Return 1 if val is a non-empty, non-null string."""
        if pd.isna(val):
            return 0
        if isinstance(val, str) and val.strip():
            return 1
        return 0

    sg = df["synergy_gives"] if "synergy_gives" in df.columns else pd.Series("", index=df.index)
    sr = df["synergy_receives"] if "synergy_receives" in df.columns else pd.Series("", index=df.index)
    df["synergy_output"] = sg.apply(_has_content)
    df["synergy_input"] = sr.apply(_has_content)
    df["synergy_links"] = df["synergy_output"] + df["synergy_input"]

    # --- v2 Economic metrics (Business-only, NaN for other types) ---------

    # Staff hiring cost (staff_min × 1k per Frontline hire)
    staff_min_col = df["staff_min"].fillna(0) if "staff_min" in df.columns else pd.Series(0.0, index=df.index)
    df["staff_hire_cost"] = 0.0
    df.loc[is_business, "staff_hire_cost"] = staff_min_col[is_business].astype(float)

    # Total launch cost: printed cost + 2k location fee if Physical
    mode_col = df["mode"].fillna("") if "mode" in df.columns else pd.Series("", index=df.index)
    location_fee = (mode_col == "Physical").astype(float) * 2
    df["total_launch_cost"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df.loc[is_business, "total_launch_cost"] = (df.loc[is_business, "cost"] + location_fee[is_business])

    # Tempo-aware income metrics
    tempo_col = df["tempo"].fillna("") if "tempo" in df.columns else pd.Series("", index=df.index)
    inc_scaled = df["income_scaled"] if "income_scaled" in df.columns else pd.Series(pd.NA, index=df.index)

    estable_mask = is_business & (tempo_col == "Estable")
    escala_mask = is_business & (tempo_col == "Escala")
    trend_mask = is_business & (tempo_col == "Tendencia")

    # max_income: best-case recurring income at base staffing
    df["max_income"] = 0.0
    df.loc[estable_mask, "max_income"] = df.loc[estable_mask, "income"]
    df.loc[escala_mask, "max_income"] = inc_scaled[escala_mask].fillna(df.loc[escala_mask, "income"])
    df.loc[trend_mask, "max_income"] = df.loc[trend_mask, "income"] + 2

    # avg_income: expected average income over the game
    df["avg_income"] = 0.0
    df.loc[estable_mask, "avg_income"] = df.loc[estable_mask, "income"]
    df.loc[escala_mask, "avg_income"] = ((df.loc[escala_mask, "income"] + df.loc[escala_mask, "max_income"]) / 2).round(2)
    df.loc[trend_mask, "avg_income"] = df.loc[trend_mask, "income"] + 1

    # effective_roi: avg_income / total_launch_cost
    launch_safe = df["total_launch_cost"].fillna(0).clip(lower=1)
    df["effective_roi"] = 0.0
    df.loc[is_business, "effective_roi"] = (df.loc[is_business, "avg_income"] / launch_safe[is_business]).round(3)

    # payback_breaks: Breaks to recover investment
    avg_safe = df["avg_income"].clip(lower=0.1)
    df["payback_breaks"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df.loc[is_business, "payback_breaks"] = (df.loc[is_business, "total_launch_cost"].fillna(0) / avg_safe[is_business]).round(2)

    # --- Complexity score -------------------------------------------------

    complexity_fields = [
        "staff_opt", "income_scaled", "income_opt", "trend_track",
        "synergy_gives", "synergy_receives", "license",
        "location_requirement", "special_rule",
    ]
    complexity_parts = []
    for field_name in complexity_fields:
        if field_name in df.columns:
            col = df[field_name]
            has_value = col.apply(lambda v: 0 if (pd.isna(v) or (isinstance(v, str) and not v.strip())) else 1)
            complexity_parts.append(has_value)
        else:
            complexity_parts.append(pd.Series(0, index=df.index))
    df["complexity_score"] = sum(complexity_parts)

    # --- hormozi_v2: cost-aware Hormozi formula ---------------------------
    # Uses total_launch_cost (falling back to printed cost) as the effort
    # proxy, making this metric meaningful for v2 cards.
    launch_or_cost = df["total_launch_cost"].fillna(df["cost"]).clip(lower=1)
    df["hormozi_v2"] = ((df["valuation_points"] * df["likelihood"]) / launch_or_cost).round(2)

    return df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, output_path: Path = OUTPUT_CSV) -> None:
    """Export the DataFrame to CSV for reproducibility."""
    # Select key columns for a clean CSV
    export_cols = [
        "id", "name", "type", "card_category", "industry", "tier",
        "cost", "income", "valuation_points", "upkeep", "net_income",
        "mode", "exit_value", "tempo",
        "staff_min", "staff_opt", "income_scaled", "income_opt",
        "trend_track", "synergy_gives", "synergy_receives",
        "license", "location_requirement", "special_rule",
        "time_delay", "effort", "likelihood",
        "hormozi_value", "roi", "efficiency",
        # v2 derived metrics
        "total_launch_cost", "staff_hire_cost",
        "max_income", "avg_income", "effective_roi", "payback_breaks",
        "synergy_output", "synergy_input", "synergy_links",
        "complexity_score", "hormozi_v2",
        # legacy counts
        "tag_count", "requirement_count", "synergy_count",
        "status", "version",
        "immediate_effect", "ongoing_effect",
        "tags", "requirements", "synergies",
        "flavor_text", "balance_notes",
    ]
    # Only include columns that exist
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"Exported {len(df)} cards to {output_path}")


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("\n" + "=" * 60)
    print("BUSINESS EMPIRE — CARD DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal cards: {len(df)}")

    print("\n--- By Type ---")
    print(df["card_category"].value_counts().to_string())

    biz = df[df["card_category"] == "Business"]
    if len(biz) > 0:
        print("\n--- Business Cards by Industry ---")
        print(biz["industry"].value_counts().to_string())

        print("\n--- Business Cards by Tier ---")
        print(biz["tier"].value_counts().to_string())

        if "tempo" in biz.columns:
            print("\n--- Business Cards by Tempo ---")
            print(biz["tempo"].value_counts().to_string())

        if "mode" in biz.columns:
            print("\n--- Business Cards by Mode ---")
            print(biz["mode"].value_counts().to_string())

        print("\n--- Business Card Economics by Tier (mean) ---")
        tier_cols = ["cost", "income", "valuation_points", "total_launch_cost",
                     "avg_income", "effective_roi", "payback_breaks"]
        tier_cols = [c for c in tier_cols if c in biz.columns]
        tier_stats = biz.groupby("tier")[tier_cols].mean()
        tier_order = ["Starter", "Established", "Premium", "Empire"]
        tier_stats = tier_stats.reindex([t for t in tier_order if t in tier_stats.index])
        print(tier_stats.round(2).to_string())

        if "tempo" in biz.columns:
            print("\n--- Business Card Economics by Tempo (mean) ---")
            tempo_cols = ["cost", "income", "max_income", "avg_income",
                          "total_launch_cost", "effective_roi", "payback_breaks"]
            tempo_cols = [c for c in tempo_cols if c in biz.columns]
            tempo_stats = biz.groupby("tempo")[tempo_cols].mean()
            print(tempo_stats.round(2).to_string())

    print("\n--- Hormozi Value Score Distribution ---")
    print(df["hormozi_value"].describe().round(2).to_string())

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse all cards, compute features, export CSV, print summary."""
    print("🎲 Business Empire — Card Data Pipeline")
    print("-" * 40)

    df = parse_all_cards()
    export_csv(df)
    print_summary(df)

    return df


if __name__ == "__main__":
    main()
