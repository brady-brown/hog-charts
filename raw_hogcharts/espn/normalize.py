"""Turn raw ESPN JSON into the four feed tables the build pipeline expects.

The existing builders read pbp / player_box / team_box / schedule frames whose
column names and dtypes come from sportsdataverse's parquet releases. Those
releases are themselves a normalization of the very endpoints this package
calls, so the goal here is a drop-in match: same column names, same dtypes, same
semantics, produced straight from ESPN.

Column coverage is deliberately a superset of what the pipeline actually reads
(24 of 59 pbp columns, 35 of 55 player_box, 28 of 59 team_box, 27 of 87
schedule) plus a handful of raw fields that cost nothing to carry and are useful
when debugging a mismatch — notably coordinate_x_raw / coordinate_y_raw and the
string play_id (sdv's `id` is a float64 and loses precision above 2^53).

Two conventions worth knowing, both verified against the sdv parquet:

  game_date   is the US/Eastern calendar date, not the UTC date. 3,406 of 6,318
              games in 2026 fall on a different UTC date, so this matters.

  coordinates ESPN gives half-court integers (x 0-50 across, y 0-~47 toward the
              basket). sdv maps them to a full-court frame:
                  coordinate_x = s * (y_raw - 41.75)
                  coordinate_y = s * (x_raw - 25)
              where s is -1 for the home team and +1 otherwise (no flip at
              halftime — both halves share one frame).
"""
import datetime as dt

import pandas as pd

EASTERN = "America/New_York"

# ── dtypes, matching the sportsdataverse parquet exactly ─────────────────────
_INT = "int32[pyarrow]"
_BIG = "int64[pyarrow]"
_DBL = "double[pyarrow]"
_STR = "large_string[pyarrow]"
_BOOL = "bool[pyarrow]"
_DATE = "date32[day][pyarrow]"

PBP_SCHEMA = {
    "game_id": _INT, "id": _DBL, "play_id": _STR, "sequence_number": _INT,
    "type_id": _INT, "type_text": _STR, "text": _STR, "short_description": _STR,
    "away_score": _INT, "home_score": _INT,
    "period_number": _INT, "period_display_value": _STR,
    "clock_display_value": _STR, "clock_minutes": _INT, "clock_seconds": _INT,
    "half": _INT,
    "scoring_play": _BOOL, "score_value": _INT,
    "shooting_play": _BOOL, "points_attempted": _INT,
    "coordinate_x_raw": _DBL, "coordinate_y_raw": _DBL,
    "coordinate_x": _DBL, "coordinate_y": _DBL,
    "team_id": _INT, "athlete_id_1": _INT, "athlete_id_2": _INT,
    "home_team_id": _INT, "away_team_id": _INT,
    "home_team_name": _STR, "away_team_name": _STR,
    "wallclock": _STR,
    "season": _INT, "season_type": _INT, "game_date": _DATE,
}

PLAYER_BOX_SCHEMA = {
    "game_id": _INT, "season": _INT, "season_type": _INT, "game_date": _DATE,
    "athlete_id": _INT, "athlete_display_name": _STR, "athlete_short_name": _STR,
    "athlete_jersey": _STR, "athlete_position_name": _STR,
    "athlete_position_abbreviation": _STR, "athlete_headshot_href": _STR,
    "team_id": _INT, "team_name": _STR, "team_location": _STR,
    "team_display_name": _STR, "team_short_display_name": _STR,
    "team_abbreviation": _STR, "team_slug": _STR, "team_logo": _STR,
    "team_color": _STR, "team_alternate_color": _STR,
    "minutes": _DBL, "field_goals_made": _INT, "field_goals_attempted": _INT,
    "three_point_field_goals_made": _INT, "three_point_field_goals_attempted": _INT,
    "free_throws_made": _INT, "free_throws_attempted": _INT,
    "offensive_rebounds": _INT, "defensive_rebounds": _INT, "rebounds": _INT,
    "assists": _INT, "steals": _INT, "blocks": _INT, "turnovers": _INT,
    "fouls": _INT, "points": _INT,
    "starter": _BOOL, "ejected": _BOOL, "did_not_play": _BOOL, "active": _BOOL,
    "home_away": _STR, "team_winner": _BOOL, "team_score": _INT,
    "opponent_team_id": _INT, "opponent_team_name": _STR,
    "opponent_team_location": _STR, "opponent_team_display_name": _STR,
    "opponent_team_score": _INT,
}

TEAM_BOX_SCHEMA = {
    "game_id": _INT, "season": _INT, "season_type": _INT, "game_date": _DATE,
    "team_id": _INT, "team_slug": _STR, "team_location": _STR, "team_name": _STR,
    "team_abbreviation": _STR, "team_display_name": _STR,
    "team_short_display_name": _STR, "team_color": _STR,
    "team_alternate_color": _STR, "team_logo": _STR,
    "team_home_away": _STR, "team_score": _INT, "team_winner": _BOOL,
    "assists": _INT, "blocks": _INT, "defensive_rebounds": _INT,
    "fast_break_points": _INT, "field_goal_pct": _DBL,
    "field_goals_made": _INT, "field_goals_attempted": _INT,
    "flagrant_fouls": _INT, "fouls": _INT, "free_throw_pct": _DBL,
    "free_throws_made": _INT, "free_throws_attempted": _INT,
    "largest_lead": _INT, "lead_changes": _INT, "lead_percentage": _DBL,
    "offensive_rebounds": _INT, "points_in_paint": _INT, "steals": _INT,
    "team_turnovers": _INT, "technical_fouls": _INT,
    "three_point_field_goal_pct": _DBL, "three_point_field_goals_made": _INT,
    "three_point_field_goals_attempted": _INT, "total_rebounds": _INT,
    "total_technical_fouls": _INT, "total_turnovers": _INT,
    "turnover_points": _INT, "turnovers": _INT,
    "opponent_team_id": _INT, "opponent_team_location": _STR,
    "opponent_team_name": _STR, "opponent_team_display_name": _STR,
    "opponent_team_abbreviation": _STR, "opponent_team_score": _INT,
}

SCHEDULE_SCHEMA = {
    "id": _INT, "game_id": _INT, "uid": _STR,
    "date": _STR, "game_date": _DATE, "game_date_time": _STR,
    "season": _INT, "season_type": _INT,
    "attendance": _INT, "neutral_site": _BOOL, "conference_competition": _BOOL,
    "play_by_play_available": _BOOL, "recent": _BOOL,
    "tournament_id": _INT, "notes_type": _STR, "notes_headline": _STR,
    "venue_id": _INT, "venue_full_name": _STR, "venue_address_city": _STR,
    "venue_address_state": _STR,
    "status_type_id": _INT, "status_type_name": _STR, "status_type_state": _STR,
    "status_type_completed": _BOOL, "status_type_detail": _STR,
    "home_id": _INT, "home_location": _STR, "home_name": _STR,
    "home_display_name": _STR, "home_abbreviation": _STR, "home_logo": _STR,
    "home_conference_id": _INT, "home_score": _INT, "home_winner": _BOOL,
    "home_current_rank": _INT,
    "away_id": _INT, "away_location": _STR, "away_name": _STR,
    "away_display_name": _STR, "away_abbreviation": _STR, "away_logo": _STR,
    "away_conference_id": _INT, "away_score": _INT, "away_winner": _BOOL,
    "away_current_rank": _INT,
}

SCHEMAS = {
    "pbp": PBP_SCHEMA,
    "player_box": PLAYER_BOX_SCHEMA,
    "team_box": TEAM_BOX_SCHEMA,
    "schedule": SCHEDULE_SCHEMA,
}


# ── small parsing helpers ────────────────────────────────────────────────────
def _int(value):
    if value is None or value == "" or value == "-":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value):
    if value is None or value == "" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _made_attempted(display_value):
    """'13-22' → (13, 22). Also tolerates '13/22' and blanks."""
    if not isinstance(display_value, str):
        return None, None
    parts = display_value.replace("/", "-").split("-")
    if len(parts) != 2:
        return None, None
    return _int(parts[0]), _int(parts[1])


def eastern_date(iso_timestamp):
    """ESPN timestamps are UTC ('2026-02-14T23:30Z'); the feeds key on ET date."""
    if not iso_timestamp:
        return None
    stamp = pd.to_datetime(iso_timestamp, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp.tz_convert(EASTERN).date()


def eastern_datetime_string(iso_timestamp):
    if not iso_timestamp:
        return None
    stamp = pd.to_datetime(iso_timestamp, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return str(stamp.tz_convert(EASTERN))


def split_clock(display_value):
    """'19:25' → (19, 25). Missing/odd clocks fall back to 0:00, as sdv does."""
    if not isinstance(display_value, str) or ":" not in display_value:
        return 0, 0
    minutes, _, seconds = display_value.partition(":")
    return _int(minutes) or 0, _int(float(seconds)) or 0


def to_frame(rows, feed):
    """Build a DataFrame with the feed's exact column order and dtypes."""
    schema = SCHEMAS[feed]
    frame = pd.DataFrame(rows, columns=list(schema))
    for column, dtype in schema.items():
        if dtype == _DATE:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").astype(_DATE)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


# ── schedule (scoreboard events) ─────────────────────────────────────────────
def normalize_event(event):
    """One scoreboard event → one schedule row."""
    competitions = event.get("competitions") or [{}]
    competition = competitions[0]
    season = event.get("season") or {}
    status = competition.get("status") or {}
    status_type = status.get("type") or {}
    venue = competition.get("venue") or {}
    address = venue.get("address") or {}
    notes = competition.get("notes") or []
    note = notes[0] if notes else {}

    sides = {}
    for competitor in competition.get("competitors") or []:
        sides[competitor.get("homeAway")] = competitor

    game_id = _int(event.get("id"))
    date_string = event.get("date")
    row = {
        "id": game_id,
        "game_id": game_id,
        "uid": event.get("uid"),
        "date": date_string,
        "game_date": eastern_date(date_string),
        "game_date_time": eastern_datetime_string(date_string),
        "season": _int(season.get("year")),
        "season_type": _int(season.get("type")),
        "attendance": _int(competition.get("attendance")),
        "neutral_site": competition.get("neutralSite"),
        "conference_competition": competition.get("conferenceCompetition"),
        "play_by_play_available": competition.get("playByPlayAvailable"),
        "recent": competition.get("recent"),
        "tournament_id": _int(competition.get("tournamentId")),
        # sdv stores "" rather than null when a game carries no notes, and
        # build_postseason keys off this column — match it exactly.
        "notes_type": note.get("type") or "",
        "notes_headline": note.get("headline") or "",
        "venue_id": _int(venue.get("id")),
        "venue_full_name": venue.get("fullName"),
        "venue_address_city": address.get("city"),
        "venue_address_state": address.get("state"),
        "status_type_id": _int(status_type.get("id")),
        "status_type_name": status_type.get("name"),
        "status_type_state": status_type.get("state"),
        "status_type_completed": status_type.get("completed"),
        "status_type_detail": status_type.get("detail"),
    }

    for side in ("home", "away"):
        competitor = sides.get(side) or {}
        team = competitor.get("team") or {}
        row.update({
            f"{side}_id": _int(team.get("id")),
            f"{side}_location": team.get("location"),
            f"{side}_name": team.get("name"),
            f"{side}_display_name": team.get("displayName"),
            f"{side}_abbreviation": team.get("abbreviation"),
            f"{side}_logo": team.get("logo"),
            f"{side}_conference_id": _int(team.get("conferenceId")),
            f"{side}_score": _int(competitor.get("score")),
            f"{side}_winner": competitor.get("winner"),
            f"{side}_current_rank": _int(competitor.get("curatedRank", {}).get("current")
                                         if isinstance(competitor.get("curatedRank"), dict)
                                         else competitor.get("rank")),
        })
    return row


# ── summary (one game → pbp + player box + team box) ─────────────────────────
def _game_context(payload):
    """Season, season_type, date and the two competitors, from the header."""
    header = payload.get("header") or {}
    competitions = header.get("competitions") or [{}]
    competition = competitions[0]
    season = header.get("season") or {}

    context = {
        "game_id": _int(header.get("id")),
        "season": _int(season.get("year")),
        "season_type": _int(season.get("type")),
        "date": competition.get("date"),
        "game_date": eastern_date(competition.get("date")),
        "teams": {},
    }
    for competitor in competition.get("competitors") or []:
        team = competitor.get("team") or {}
        side = competitor.get("homeAway")
        context["teams"][side] = {
            "id": _int(team.get("id")),
            "name": team.get("name"),
            "location": team.get("location"),
            "display_name": team.get("displayName"),
            "abbreviation": team.get("abbreviation"),
            "score": _int(competitor.get("score")),
            "winner": competitor.get("winner"),
        }
    return context


def normalize_plays(payload, context=None):
    """summary['plays'] → pbp rows (one per play), sdv-shaped."""
    context = context or _game_context(payload)
    plays = payload.get("plays") or []
    if not plays:
        return []

    home = context["teams"].get("home") or {}
    away = context["teams"].get("away") or {}
    home_team_id = home.get("id")
    away_team_id = away.get("id")

    rows = []
    for play in plays:
        play_type = play.get("type") or {}
        period = play.get("period") or {}
        clock = play.get("clock") or {}
        coordinate = play.get("coordinate") or {}
        participants = play.get("participants") or []

        team_id = _int((play.get("team") or {}).get("id"))
        athlete_ids = [
            _int(((participant or {}).get("athlete") or {}).get("id"))
            for participant in participants
        ]
        athlete_ids = [athlete_id for athlete_id in athlete_ids if athlete_id is not None]

        raw_x = _float(coordinate.get("x"))
        raw_y = _float(coordinate.get("y"))
        # -1 for the home team, +1 otherwise: puts both teams' shots in one
        # full-court frame. Verified against sdv on every play of two games.
        sign = -1.0 if (team_id is not None and team_id == home_team_id) else 1.0
        if raw_x is None or raw_y is None:
            coordinate_x = coordinate_y = None
        else:
            coordinate_x = sign * (raw_y - 41.75)
            coordinate_y = sign * (raw_x - 25.0)

        clock_minutes, clock_seconds = split_clock(clock.get("displayValue"))
        period_number = _int(period.get("number"))
        play_id = play.get("id")

        rows.append({
            "game_id": context["game_id"],
            "id": _float(play_id),
            "play_id": str(play_id) if play_id is not None else None,
            "sequence_number": _int(play.get("sequenceNumber")),
            "type_id": _int(play_type.get("id")),
            "type_text": play_type.get("text"),
            "text": play.get("text"),
            "short_description": play.get("shortDescription"),
            "away_score": _int(play.get("awayScore")),
            "home_score": _int(play.get("homeScore")),
            "period_number": period_number,
            "period_display_value": period.get("displayValue"),
            "clock_display_value": clock.get("displayValue"),
            "clock_minutes": clock_minutes,
            "clock_seconds": clock_seconds,
            "half": None if period_number is None else min(period_number, 2),
            "scoring_play": play.get("scoringPlay"),
            "score_value": _int(play.get("scoreValue")),
            "shooting_play": play.get("shootingPlay"),
            "points_attempted": _int(play.get("pointsAttempted")),
            "coordinate_x_raw": raw_x,
            "coordinate_y_raw": raw_y,
            "coordinate_x": coordinate_x,
            "coordinate_y": coordinate_y,
            "team_id": team_id,
            "athlete_id_1": athlete_ids[0] if len(athlete_ids) > 0 else None,
            "athlete_id_2": athlete_ids[1] if len(athlete_ids) > 1 else None,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team_name": home.get("name"),
            "away_team_name": away.get("name"),
            "wallclock": play.get("wallclock"),
            "season": context["season"],
            "season_type": context["season_type"],
            "game_date": context["game_date"],
        })
    return rows


# player box stat keys, as ESPN labels them in boxscore.players[].statistics[].keys
_SPLIT_STATS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("field_goals_made", "field_goals_attempted"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": (
        "three_point_field_goals_made", "three_point_field_goals_attempted"),
    "freeThrowsMade-freeThrowsAttempted": ("free_throws_made", "free_throws_attempted"),
}
_SCALAR_STATS = {
    "minutes": "minutes", "points": "points", "rebounds": "rebounds",
    "assists": "assists", "turnovers": "turnovers", "steals": "steals",
    "blocks": "blocks", "offensiveRebounds": "offensive_rebounds",
    "defensiveRebounds": "defensive_rebounds", "fouls": "fouls",
}


def normalize_player_box(payload, context=None):
    """summary['boxscore']['players'] → one row per player per game."""
    context = context or _game_context(payload)
    boxscore = payload.get("boxscore") or {}
    rows = []

    for team_block in boxscore.get("players") or []:
        team = team_block.get("team") or {}
        team_id = _int(team.get("id"))
        side = "home" if team_id == (context["teams"].get("home") or {}).get("id") else "away"
        own = context["teams"].get(side) or {}
        opponent = context["teams"].get("away" if side == "home" else "home") or {}

        for statistics in team_block.get("statistics") or []:
            keys = statistics.get("keys") or []
            for entry in statistics.get("athletes") or []:
                athlete = entry.get("athlete") or {}
                position = athlete.get("position") or {}
                headshot = athlete.get("headshot") or {}
                values = entry.get("stats") or []

                row = {
                    "game_id": context["game_id"],
                    "season": context["season"],
                    "season_type": context["season_type"],
                    "game_date": context["game_date"],
                    "athlete_id": _int(athlete.get("id")),
                    "athlete_display_name": athlete.get("displayName"),
                    "athlete_short_name": athlete.get("shortName"),
                    "athlete_jersey": athlete.get("jersey"),
                    "athlete_position_name": position.get("name"),
                    "athlete_position_abbreviation": position.get("abbreviation"),
                    "athlete_headshot_href": headshot.get("href"),
                    "team_id": team_id,
                    "team_name": team.get("name"),
                    "team_location": team.get("location"),
                    "team_display_name": team.get("displayName"),
                    "team_short_display_name": team.get("shortDisplayName"),
                    "team_abbreviation": team.get("abbreviation"),
                    "team_slug": team.get("slug"),
                    "team_logo": team.get("logo"),
                    "team_color": team.get("color"),
                    "team_alternate_color": team.get("alternateColor"),
                    "starter": entry.get("starter"),
                    "ejected": entry.get("ejected"),
                    "did_not_play": entry.get("didNotPlay"),
                    "active": entry.get("active"),
                    "home_away": side,
                    "team_winner": own.get("winner"),
                    "team_score": own.get("score"),
                    "opponent_team_id": opponent.get("id"),
                    "opponent_team_name": opponent.get("name"),
                    "opponent_team_location": opponent.get("location"),
                    "opponent_team_display_name": opponent.get("display_name"),
                    "opponent_team_score": opponent.get("score"),
                }
                # Players who didn't play carry an empty stats array; leaving the
                # counting stats null (rather than zero) matches sdv exactly.
                for key, value in zip(keys, values):
                    if key in _SCALAR_STATS:
                        target = _SCALAR_STATS[key]
                        row[target] = _float(value) if target == "minutes" else _int(value)
                    elif key in _SPLIT_STATS:
                        made_column, attempted_column = _SPLIT_STATS[key]
                        made, attempted = _made_attempted(value)
                        row[made_column] = made
                        row[attempted_column] = attempted
                rows.append(row)
    return rows


# team box stat names, as ESPN labels them in boxscore.teams[].statistics[].name
_TEAM_SPLIT_STATS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("field_goals_made", "field_goals_attempted"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": (
        "three_point_field_goals_made", "three_point_field_goals_attempted"),
    "freeThrowsMade-freeThrowsAttempted": ("free_throws_made", "free_throws_attempted"),
}
_TEAM_INT_STATS = {
    "totalRebounds": "total_rebounds", "offensiveRebounds": "offensive_rebounds",
    "defensiveRebounds": "defensive_rebounds", "assists": "assists",
    "steals": "steals", "blocks": "blocks", "turnovers": "turnovers",
    "teamTurnovers": "team_turnovers", "totalTurnovers": "total_turnovers",
    "technicalFouls": "technical_fouls", "totalTechnicalFouls": "total_technical_fouls",
    "flagrantFouls": "flagrant_fouls", "turnoverPoints": "turnover_points",
    "fastBreakPoints": "fast_break_points", "pointsInPaint": "points_in_paint",
    "fouls": "fouls", "largestLead": "largest_lead", "leadChanges": "lead_changes",
}
_TEAM_FLOAT_STATS = {
    "fieldGoalPct": "field_goal_pct", "threePointFieldGoalPct": "three_point_field_goal_pct",
    "freeThrowPct": "free_throw_pct", "leadPercentage": "lead_percentage",
}


def normalize_team_box(payload, context=None):
    """summary['boxscore']['teams'] → one row per team per game."""
    context = context or _game_context(payload)
    boxscore = payload.get("boxscore") or {}
    rows = []

    for team_block in boxscore.get("teams") or []:
        team = team_block.get("team") or {}
        team_id = _int(team.get("id"))
        side = team_block.get("homeAway") or (
            "home" if team_id == (context["teams"].get("home") or {}).get("id") else "away")
        own = context["teams"].get(side) or {}
        opponent = context["teams"].get("away" if side == "home" else "home") or {}

        row = {
            "game_id": context["game_id"],
            "season": context["season"],
            "season_type": context["season_type"],
            "game_date": context["game_date"],
            "team_id": team_id,
            "team_slug": team.get("slug"),
            "team_location": team.get("location"),
            "team_name": team.get("name"),
            "team_abbreviation": team.get("abbreviation"),
            "team_display_name": team.get("displayName"),
            "team_short_display_name": team.get("shortDisplayName"),
            "team_color": team.get("color"),
            "team_alternate_color": team.get("alternateColor"),
            "team_logo": team.get("logo"),
            "team_home_away": side,
            "team_score": own.get("score"),
            "team_winner": own.get("winner"),
            "opponent_team_id": opponent.get("id"),
            "opponent_team_location": opponent.get("location"),
            "opponent_team_name": opponent.get("name"),
            "opponent_team_display_name": opponent.get("display_name"),
            "opponent_team_abbreviation": opponent.get("abbreviation"),
            "opponent_team_score": opponent.get("score"),
        }
        for statistic in team_block.get("statistics") or []:
            name = statistic.get("name")
            value = statistic.get("displayValue")
            if name in _TEAM_SPLIT_STATS:
                made_column, attempted_column = _TEAM_SPLIT_STATS[name]
                made, attempted = _made_attempted(value)
                row[made_column] = made
                row[attempted_column] = attempted
            elif name in _TEAM_INT_STATS:
                row[_TEAM_INT_STATS[name]] = _int(value)
            elif name in _TEAM_FLOAT_STATS:
                row[_TEAM_FLOAT_STATS[name]] = _float(value)
        rows.append(row)
    return rows


def normalize_summary(payload):
    """One summary payload → {'pbp': [...], 'player_box': [...], 'team_box': [...]}."""
    context = _game_context(payload)
    return {
        "pbp": normalize_plays(payload, context),
        "player_box": normalize_player_box(payload, context),
        "team_box": normalize_team_box(payload, context),
    }
