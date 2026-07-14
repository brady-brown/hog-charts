"""The lineup / stint engine — PBP substitution events → on-court stints.

ONE engine, feeding both build_onoff_rapm (on/off + RAPM, tracks both teams'
5-man lineups per stint) and build_lineups (per-team n-man combos). A stint is a
run of plays between two substitutions, during which both teams' lineups are
fixed.

Before this, the two builders reconstructed stints independently with different
rules (possession weights, sub timing, starter fallback), so the Lineups page
and the on/off table disagreed for the same player by construction. build_stints
below is lifted verbatim from build_onoff_rapm so on/off + RAPM are unchanged;
build_lineups now consumes the same stints.
"""
import numpy as np
import pandas as pd


def clock_to_seconds(clock_minutes_value, clock_seconds_value):
    """Convert separate clock_minutes / clock_seconds columns to total seconds remaining."""
    try:
        return int(clock_minutes_value) * 60 + int(clock_seconds_value)
    except Exception:
        return 0


def build_stints(play_by_play, starters_by_game_team):
    """Assign a stint_id to every play; return (play_by_play, stint_lineup_info).

    play_by_play must be sorted by (game_id, sequence_number) with a reset 0..n
    index, and carry: game_id, home_team_id, away_team_id, team_id, athlete_id_1,
    is_subbing_in, is_subbing_out. starters_by_game_team maps (game_id, team_id)
    → frozenset of starter athlete_ids.

    Returns the same play_by_play with an added integer "stint_id" column, and
    stint_lineup_info: one row per (game_id, stint_id) with home_lineup /
    away_lineup frozensets (the on-court 5 for each team during that stint).
    """
    all_game_ids   = play_by_play["game_id"].unique()
    play_stint_ids = np.zeros(len(play_by_play), dtype=np.int32)  # stint number for every play row
    lineup_change_records = []    # one record per lineup state (game × stint)

    for game_id in all_game_ids:
        game_mask         = play_by_play["game_id"] == game_id
        game_play_indices = play_by_play.index[game_mask].to_numpy()
        game_plays        = play_by_play.loc[game_play_indices]

        home_team_id = int(game_plays["home_team_id"].iloc[0])
        away_team_id = int(game_plays["away_team_id"].iloc[0])
        home_lineup  = set(starters_by_game_team.get((game_id, home_team_id), set()))
        away_lineup  = set(starters_by_game_team.get((game_id, away_team_id), set()))

        current_stint_number = 0
        game_stint_ids        = np.zeros(len(game_play_indices), dtype=np.int32)
        is_subbing_in_array   = game_plays["is_subbing_in"].to_numpy()
        is_subbing_out_array  = game_plays["is_subbing_out"].to_numpy()
        team_id_array         = game_plays["team_id"].to_numpy()
        primary_athlete_id_array = game_plays["athlete_id_1"].to_numpy()

        for play_index in range(len(game_play_indices)):
            if is_subbing_in_array[play_index] or is_subbing_out_array[play_index]:
                # A substitution ends the current stint — record the lineup state.
                lineup_change_records.append({
                    "game_id":      game_id,
                    "stint_id":     current_stint_number,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_lineup":  frozenset(home_lineup),
                    "away_lineup":  frozenset(away_lineup),
                })
                current_stint_number += 1
                player_id = primary_athlete_id_array[play_index]
                if pd.notna(player_id):
                    player_id = int(player_id)
                    team_id_of_sub = team_id_array[play_index]
                    if is_subbing_in_array[play_index]:
                        if team_id_of_sub == home_team_id:
                            home_lineup.add(player_id)
                        elif team_id_of_sub == away_team_id:
                            away_lineup.add(player_id)
                    else:
                        if team_id_of_sub == home_team_id:
                            home_lineup.discard(player_id)
                        elif team_id_of_sub == away_team_id:
                            away_lineup.discard(player_id)
            game_stint_ids[play_index] = current_stint_number

        # Record the final stint (after the last substitution).
        lineup_change_records.append({
            "game_id":      game_id,
            "stint_id":     current_stint_number,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_lineup":  frozenset(home_lineup),
            "away_lineup":  frozenset(away_lineup),
        })
        play_stint_ids[game_play_indices] = game_stint_ids

    play_by_play = play_by_play.copy()
    play_by_play["stint_id"] = play_stint_ids
    stint_lineup_info = (
        pd.DataFrame(lineup_change_records)
        .drop_duplicates(subset=["game_id", "stint_id"], keep="last")
        .reset_index(drop=True)
    )
    return play_by_play, stint_lineup_info
