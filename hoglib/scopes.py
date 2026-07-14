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
Playoffs", …). Early-season multi-team events (Maui Invitational, etc.) are
non-conference, so conference_competition is False and they are not caught.
"""
from collections import namedtuple

# Derived scopes plus the raw building-block sets (build_player_stats needs the
# conference_competition and conference_tournament sets directly for its masks).
ScopeSets = namedtuple(
    "ScopeSets",
    "reg all post conf nonconf conf_competition conf_tournament type2 type3",
)

_TOURNEY_PATTERN = "Tournament|Championship|Playoffs"


def game_id_sets(schedule_df):
    """Partition a season schedule DataFrame into ScopeSets (all fields are sets of int game_ids)."""
    sched = schedule_df
    conf_competition = set(
        sched.loc[sched["conference_competition"] == True, "game_id"].astype(int).unique()
    )

    notes = sched.get("notes_headline")
    if notes is not None:
        is_conf_tourney = (
            (sched["conference_competition"] == True)
            & notes.astype(str).str.contains(_TOURNEY_PATTERN, case=False, na=False)
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
