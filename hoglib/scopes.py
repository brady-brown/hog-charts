"""Game-scope partitioning — one copy for the whole pipeline.

Splits a season's games into the five scopes the site toggles between. The
subtle rule (duplicated across build_player_stats / build_points_resp /
build_shot_diet before this): conference-TOURNAMENT games are ESPN-tagged
season_type 2 (regular season) but played in March, so they are reclassified
OUT of reg/conf/nonconf and INTO post.

    reg     = season_type 2 games that are NOT conference tournaments
    conf    = reg games flagged conference_competition (league games)
    nonconf = reg games that are NOT conference league games
    post    = season_type 3 (NCAA/NIT/…) PLUS the conference tournaments
    all     = reg ∪ post

A conference tournament is a conference_competition game whose notes_headline
names the event ("SEC Tournament", "ASUN Championship", "America East
Playoffs", …) and is played on or after CONF_TOURNEY_EARLIEST (Feb 15). Early-
season multi-team events are usually non-conference, but not always: the 2027
Players Era 8 final was Kansas vs Houston, two Big 12 teams, so ESPN flagged it
conference_competition and its "Championship" headline matched. The earliest real
conference tournament game 2021-26 was Feb 25 (2021 Horizon League).
"""
from collections import namedtuple

# Derived scopes plus the raw building-block sets (build_player_stats needs the
# conference_competition and conference_tournament sets directly for its masks).
ScopeSets = namedtuple(
    "ScopeSets",
    "reg all post conf nonconf conf_competition conf_tournament type2 type3",
)

_TOURNEY_PATTERN = "Tournament|Championship|Playoffs"
CONF_TOURNEY_EARLIEST = (2, 15)       # (month, day) in the calendar year the season ends


def game_id_sets(schedule_df):
    """Partition a season schedule DataFrame into ScopeSets (all fields are sets of int game_ids)."""
    sched = schedule_df
    conf_competition = set(
        sched.loc[sched["conference_competition"] == True, "game_id"].astype(int).unique()
    )

    notes = sched.get("notes_headline")
    if notes is not None:
        import pandas as pd
        game_day = pd.to_datetime(sched["game_date"], errors="coerce")
        season_end_year = game_day.dt.year.max()      # Nov/Dec games belong to next year's season
        month, day = CONF_TOURNEY_EARLIEST
        is_conf_tourney = (
            (sched["conference_competition"] == True)
            & notes.astype(str).str.contains(_TOURNEY_PATTERN, case=False, na=False)
            & (game_day >= pd.Timestamp(int(season_end_year), month, day))
        )
        conf_tournament = set(
            sched.loc[is_conf_tourney, "game_id"].astype(int).unique()
        )
    else:
        conf_tournament = set()

    type2 = set(sched.loc[sched["season_type"] == 2, "game_id"].astype(int).unique())
    type3 = set(sched.loc[sched["season_type"] == 3, "game_id"].astype(int).unique())

    reg = type2 - conf_tournament
    conf = (conf_competition - conf_tournament) & reg
    nonconf = reg - conf
    post = type3 | conf_tournament
    all_games = reg | post

    return ScopeSets(reg, all_games, post, conf, nonconf,
                     conf_competition, conf_tournament, type2, type3)
