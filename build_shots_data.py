"""
Run this locally whenever you want to refresh shot data:
    python build_shots_data.py

Generates shots_{SEASON}.parquet — commit that file and the cloud app
will load from it instead of downloading the full PBP at runtime.
"""

from datetime import date as _date

import pandas as pd
import sportsdataverse.mbb as mbb

_today   = _date.today()
SEASON   = _today.year + 1 if _today.month >= 11 else _today.year
OUT_FILE = f"shots_{SEASON}.parquet"

SHOT_COLS = [
    "game_id", "team_id", "athlete_id_1",
    "coordinate_x", "coordinate_y",
    "scoring_play", "type_text", "text",
]
GAME_COLS = ["game_id", "home_team_id", "away_team_id"]

print(f"Loading {SEASON} PBP data...")
raw_pbp = mbb.load_mbb_pbp(seasons=[SEASON], return_as_pandas=True)

date_col = "game_date" if "game_date" in raw_pbp.columns else "date"
GAME_COLS.append(date_col)

print("Filtering to shooting plays...")
# Don't include date_col here — it comes from game_index merge below
shots = raw_pbp.loc[
    raw_pbp["shooting_play"] == True,
    [c for c in SHOT_COLS if c in raw_pbp.columns]
].copy()

game_index = (
    raw_pbp[[c for c in GAME_COLS if c in raw_pbp.columns]]
    .drop_duplicates(subset=["game_id"])
    .copy()
)
game_index["_is_game_row"] = True

del raw_pbp

combined = pd.merge(shots, game_index, on="game_id", how="left")
combined["_date_col"] = date_col

print(f"Loading boxscore data...")
raw_box = mbb.load_mbb_player_boxscore(seasons=[SEASON], return_as_pandas=True)
box_cols = ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
box = raw_box[[c for c in box_cols if c in raw_box.columns]].copy()
del raw_box

box.to_parquet(f"box_{SEASON}.parquet", index=False)
combined.to_parquet(OUT_FILE, index=False)

print(f"\nDone.")
print(f"  shots_{SEASON}.parquet  — {len(combined):,} shot rows")
print(f"  box_{SEASON}.parquet    — {len(box):,} player rows")
