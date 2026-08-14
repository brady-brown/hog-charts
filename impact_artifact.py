"""
impact_artifact.py — Build the Player Impact Hub artifact (player-impact.json).

WHY THIS FILE EXISTS
--------------------
The impact hub is the deepest individual-player view on the site.  It
combines two independently computed metrics into one table:

  RAPM (Regularized Adjusted Plus-Minus)
      Ridge regression over the stint design matrix; measures how many net
      points per 100 possessions a player contributes independent of his
      teammates.  Every player is shrunk toward zero.

  On/Off net rating differential
      Raw net rating when the player is on the court vs. off it.  This is
      not causal (team quality confounds it) but it's intuitive and widely
      understood.

Used two ways:
  • Imported by build_site.py during the nightly site rebuild.
  • Run directly to regenerate a single season:
        python3 impact_artifact.py
        OVERRIDE_SEASON=2025 python3 impact_artifact.py
"""
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(python_obj):
    """Recursively replace NaN/Inf floats with None for valid JSON output.

    Python's json.dumps emits bare `NaN` by default, which is not valid
    JSON — browsers will reject it with a SyntaxError.
    """
    if isinstance(python_obj, float):
        return None if (python_obj != python_obj
                        or python_obj in (float("inf"), float("-inf"))) else python_obj
    if isinstance(python_obj, dict):
        return {key: _sanitize_for_json(val) for key, val in python_obj.items()}
    if isinstance(python_obj, list):
        return [_sanitize_for_json(item) for item in python_obj]
    return python_obj


def rapm_filename_suffix(season_end_year):
    """Convert a season end-year to the RAPM CSV filename suffix.

    The RAPM pipeline outputs files named mbb_rapm_202526.csv (for the
    2024-25 season, which ends in 2026).  The suffix encodes both years.

    Example: 2026 → '202526'
    """
    return f"{season_end_year - 1}{str(season_end_year)[-2:]}"


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def build_impact_records(project_root, season, conf_map=None,
                         min_possessions=400, conf_variant=False):
    """Return a list of per-player impact records, or None if RAPM data is missing.

    Args:
        project_root:  Absolute path to the repo root directory.
        season:        Calendar year the season ends (e.g. 2026).
        conf_map:      Optional {team_display_name: conference_abbreviation} dict
                       for filling in conf. labels missing from the RAPM CSV.
        min_possessions: Minimum total possessions for a player to appear.
                       Filters out tiny-sample players whose RAPM is unstable.
        conf_variant:  If True, load the conference-only RAPM and on/off files.

    Returns:
        list of dicts or None if the RAPM file does not exist.
    """
    filename_suffix = rapm_filename_suffix(season)
    conference_file_suffix = "_conf" if conf_variant else ""

    rapm_csv_path = os.path.join(
        project_root, f"mbb_rapm_{filename_suffix}{conference_file_suffix}.csv"
    )
    if not os.path.exists(rapm_csv_path):
        return None

    rapm_df = pd.read_csv(rapm_csv_path)

    # ── Merge on/off detail ──────────────────────────────────────────────────
    onoff_csv_filename = f"mbb_onoff_{season}{'_conf' if conf_variant else ''}_v2.csv"
    onoff_csv_path     = os.path.join(project_root, onoff_csv_filename)
    on_off_stat_columns = [
        "nrtg_on", "nrtg_off", "on_off",
        "ortg_on", "drtg_on", "ortg_off", "drtg_off"
    ]
    if os.path.exists(onoff_csv_path):
        on_off_df = pd.read_csv(
            onoff_csv_path,
            usecols=["athlete_id", *on_off_stat_columns, "poss_off_on"]
        )
        rapm_df = rapm_df.merge(on_off_df, on="athlete_id", how="left")
    else:
        for column_name in (*on_off_stat_columns, "poss_off_on"):
            rapm_df[column_name] = np.nan

    # ── Merge position + conference from the player_stats CSV ────────────────
    player_stats_csv_path = os.path.join(project_root, f"player_stats_{season}.csv")
    if os.path.exists(player_stats_csv_path):
        player_identity_df = (
            pd.read_csv(player_stats_csv_path,
                        usecols=["athlete_id", "athlete_position_name", "conf."])
            .drop_duplicates("athlete_id")
        )
        rapm_df = rapm_df.merge(player_identity_df, on="athlete_id", how="left")
    else:
        rapm_df["athlete_position_name"] = None
        rapm_df["conf."] = None

    # Fill conference from the team-level conf_map where the CSV had a gap.
    if conf_map:
        rapm_df["conf."] = rapm_df["conf."].fillna(
            rapm_df["team_display_name"].map(conf_map)
        )

    # ── Apply minimum possessions filter ─────────────────────────────────────
    rapm_df = rapm_df[rapm_df["total_poss"].fillna(0) >= min_possessions].copy()

    # ── Round to display precision ────────────────────────────────────────────
    rapm_two_decimal_columns = ["o_rapm", "d_rapm", "rapm"]
    rapm_one_decimal_columns = [
        "nrtg_on", "nrtg_off", "on_off",
        "ortg_on", "drtg_on", "ortg_off", "drtg_off"
    ]
    for column_name in rapm_two_decimal_columns:
        rapm_df[column_name] = (
            pd.to_numeric(rapm_df[column_name], errors="coerce").round(2)
            if column_name in rapm_df.columns else None
        )
    for column_name in rapm_one_decimal_columns:
        rapm_df[column_name] = pd.to_numeric(rapm_df[column_name], errors="coerce").round(1)
    rapm_df["total_poss"] = pd.to_numeric(rapm_df["total_poss"], errors="coerce").round(0)

    # ── Build compact output DataFrame ───────────────────────────────────────
    compact_output_df = pd.DataFrame({
        "id":       rapm_df["athlete_id"],
        "n":        rapm_df["athlete_display_name"],
        "tid":      rapm_df["team_id"],
        "t":        rapm_df["team_display_name"],
        "conf":     rapm_df["conf."],
        "pos":      rapm_df["athlete_position_name"],
        "poss":     rapm_df["total_poss"],
        "orapm":    rapm_df["o_rapm"],
        "drapm":    rapm_df["d_rapm"],
        "rapm":     rapm_df["rapm"],
        "non":      rapm_df["nrtg_on"],
        "noff":     rapm_df["nrtg_off"],
        "onoff":    rapm_df["on_off"],
        "oon":      rapm_df["ortg_on"],
        "don":      rapm_df["drtg_on"],
    })
    compact_output_df = compact_output_df.sort_values("rapm", ascending=False)
    json_records = compact_output_df.where(compact_output_df.notna(), other=None).to_dict("records")

    # Convert numeric ID fields from float → int in the final JSON.
    for player_record in json_records:
        for integer_field in ("id", "tid", "poss"):
            if player_record[integer_field] is not None:
                player_record[integer_field] = int(player_record[integer_field])

    return json_records


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date as _date

    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    season_override  = os.environ.get("OVERRIDE_SEASON")
    if season_override:
        current_season = int(season_override)
    else:
        today = _date.today()
        current_season = today.year + 1 if today.month >= 11 else today.year

    # Build team→conference map from the overall player_stats CSV.
    overall_stats_path = os.path.join(project_root_dir, f"player_stats_{current_season}.csv")
    conference_label_map = {}
    if os.path.exists(overall_stats_path):
        raw_conf_df = pd.read_csv(overall_stats_path, usecols=["team_display_name", "conf."]).dropna()
        conference_label_map = (
            raw_conf_df.drop_duplicates("team_display_name")
            .set_index("team_display_name")["conf."].to_dict()
        )

    season_output_dir = os.path.join(project_root_dir, "site", "data", str(current_season))
    os.makedirs(season_output_dir, exist_ok=True)

    output_file_specs = [
        (False, "player-impact.json",      400),
        (True,  "player-impact-conf.json", 200),
    ]
    for is_conference_variant, output_filename, minimum_possessions in output_file_specs:
        impact_records = build_impact_records(
            project_root_dir, current_season,
            conf_map=conference_label_map,
            conf_variant=is_conference_variant,
            min_possessions=minimum_possessions,
        )
        if impact_records is None:
            variant_label = "conf" if is_conference_variant else "overall"
            print(f"  [skip] no RAPM file for {current_season} ({variant_label})")
            continue

        output_path = os.path.join(season_output_dir, output_filename)
        build_meta  = {"built_at": None, "season": current_season}
        with open(output_path, "w") as output_file:
            output_file.write(
                json.dumps(
                    _sanitize_for_json({"players": impact_records, "meta": build_meta}),
                    separators=(",", ":"), allow_nan=False
                )
            )
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"  {output_filename:<28s} {len(impact_records):5d} players  {file_size_kb:7.0f} KB")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# --- rapm_filename_suffix() ---
# season_end_year              int     Calendar year the season ends (e.g. 2026).
# return value                 str     "202526" — the suffix used in RAPM CSV filenames.
#
# --- build_impact_records() ---
# project_root                 str     Absolute path to the repo root.
# season                       int     Calendar year the season ends.
# conf_map                     dict    {team_display_name: conference_abbreviation}; optional.
# min_possessions              int     Minimum total_poss to include a player.
# conf_variant                 bool    True = load conference-only RAPM/on-off files.
# filename_suffix              str     e.g. "202526" (from rapm_filename_suffix).
# conference_file_suffix       str     "_conf" or "" depending on conf_variant.
# rapm_csv_path                str     Full path to the RAPM CSV file.
# rapm_df                      DataFrame  Loaded RAPM data; on/off and identity cols merged in.
# onoff_csv_path               str     Path to the on/off CSV file.
# on_off_df                    DataFrame  on_off, nrtg_on, nrtg_off, ortg/drtg on/off, poss_off_on.
# on_off_stat_columns          list    Column names read from the on/off CSV.
# player_identity_df           DataFrame  athlete_id → position + conf. from player_stats CSV.
# rapm_two_decimal_columns     list    RAPM value columns rounded to 2 decimal places.
# rapm_one_decimal_columns     list    On/off stat columns rounded to 1 decimal place.
# compact_output_df            DataFrame  Short-column-name output ready for JSON serialization.
# json_records                 list    to_dict("records") output; integers converted from float.
#
# --- __main__ ---
# project_root_dir             str     Repo root (same as project_root when imported).
# current_season               int     Season being built.
# overall_stats_path           str     Path to player_stats_{season}.csv.
# conference_label_map         dict    {team: conf} built from overall stats CSV.
# season_output_dir            str     site/data/{season}/ output directory.
# output_file_specs            list    [(conf_variant, filename, min_poss), …].
# is_conference_variant        bool    Whether this iteration is the conference-only version.
# output_filename              str     "player-impact.json" or "player-impact-conf.json".
# minimum_possessions          int     min_possessions for this variant.
# impact_records               list    Return value of build_impact_records().
# build_meta                   dict    Metadata dict embedded in the output JSON.
