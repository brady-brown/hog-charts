"""
predictor.py — tempo-adjusted college basketball game predictor.

Extracted verbatim from isotonic_predictor.ipynb (the production TempoPredictor,
FORM_MODE='residual') so it can be imported by the web app and the artifact
builder without running a notebook.

HIGH-LEVEL PIPELINE
-------------------
1.  For every team, compute an "adjusted efficiency" rating (off_eff, def_eff)
    by iterating over games and adjusting each performance for the quality of
    the opponent faced.  We blend this season's ratings with last season's to
    stabilize early-season estimates.

2.  Train a linear spread model on historical walk-forward data:
        predicted_margin = coef[0] * tempo_adj
                         + coef[1] * home_court
                         + coef[2] * form_diff

3.  Calibrate predicted margins to win probabilities using isotonic regression
    so the curve is always monotone (higher spread → higher win prob).

4.  At prediction time, look up efficiency, pace, and form for both teams and
    run the spread through the calibrator to get P(team1 wins).
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from hoglib import feeds

warnings.filterwarnings("ignore")

# Bayesian shrinkage constant for blending current-season and prior-season
# efficiency ratings.  A team with n games is given weight n / (n + k).
# Tuned empirically to 7 (74.24% held-out accuracy vs 74.13% at k=10 with FORM_GAMES=5).
GAME_WEIGHT_CONSTANT = 7


# ==============================================================================
# Efficiency helpers
# ==============================================================================

def calculate_efficiency(team_game_df):
    """Compute raw (unadjusted) offensive and defensive efficiency per team-season.

    Offensive efficiency = points scored per 100 possessions.
    Defensive efficiency = points allowed per 100 possessions.
    Possessions estimated with the project-wide formula:
        FGA + 0.44 * FTA - ORB + TOV

    Parameters
    ----------
    team_game_df : DataFrame
        One row per team per game (from the team boxscore loader).

    Returns
    -------
    DataFrame with columns: season, team_id, off_poss, def_poss, off_eff, def_eff.
    """
    # Only keep games where the opponent is also a Division I team.
    division_one_team_ids = team_game_df['team_id'].unique()
    team_game_df = team_game_df[team_game_df['opponent_team_id'].isin(division_one_team_ids)]

    # Season totals for the team.
    team_season_stats = team_game_df.groupby(['season', 'team_id']).agg({
        'team_score':              'sum',
        'opponent_team_score':     'sum',
        'field_goals_attempted':   'sum',
        'free_throws_attempted':   'sum',
        'offensive_rebounds':      'sum',
        'turnovers':               'sum',
    }).reset_index()

    # Season totals for opponents (needed for defensive efficiency denominator).
    opponent_season_stats = team_game_df.groupby(['season', 'opponent_team_id']).agg({
        'field_goals_attempted':  'sum',
        'free_throws_attempted':  'sum',
        'offensive_rebounds':     'sum',
        'turnovers':              'sum',
    }).reset_index()
    opponent_season_stats.columns = [
        'season', 'team_id',
        'opp_fga', 'opp_fta', 'opp_orb', 'opp_tov'
    ]

    combined_stats = team_season_stats.merge(opponent_season_stats, on=['season', 'team_id'], how='left')

    combined_stats['off_poss'] = (
        combined_stats['field_goals_attempted']
        + 0.44 * combined_stats['free_throws_attempted']
        - combined_stats['offensive_rebounds']
        + combined_stats['turnovers']
    ).replace(0, 1)    # prevent division by zero

    combined_stats['def_poss'] = (
        combined_stats['opp_fga']
        + 0.44 * combined_stats['opp_fta']
        - combined_stats['opp_orb']
        + combined_stats['opp_tov']
    ).replace(0, 1)

    combined_stats['off_eff'] = (combined_stats['team_score']     / combined_stats['off_poss'] * 100).clip(70, 140)
    combined_stats['def_eff'] = (combined_stats['opponent_team_score'] / combined_stats['def_poss'] * 100).clip(70, 140)
    return combined_stats


def regress_three_point_shooting(team_game_df, k_off=600, k_def=1500):
    """Re-score every game as if each team shot its *expected* 3P%.

    Three-point percentage is noisy, and opponents' 3P% against a defense is
    mostly luck; sharp lines regress both and raw efficiency doesn't. On the
    2021-26 line study the model lost to the closing line 43% of the time in
    power-conference games when it leaned toward the team with the luckier 3P%
    numbers. Here, over the games passed in (only games already played):

      own rate   = (3PM + k_off * lg) / (3PA + k_off)      shooting is partly skill
      allowed    = (opp 3PM + k_def * lg) / (opp 3PA + k_def)   almost all luck
      expected   = lg + (own - lg) + (opponent's allowed - lg)
      team_score = team_score - 3 * 3PM + 3 * 3PA * expected

    and opponent_team_score is the opponent row's adjusted score. Only the two
    score columns change, so efficiency/adjustment code downstream is untouched.
    """
    df = team_game_df.copy()
    made = pd.to_numeric(df['three_point_field_goals_made'], errors='coerce').fillna(0)
    att = pd.to_numeric(df['three_point_field_goals_attempted'], errors='coerce').fillna(0)
    league = made.sum() / max(att.sum(), 1)
    shots = pd.DataFrame({'team_id': df['team_id'], 'opp': df['opponent_team_id'], 'm': made, 'a': att})
    own = shots.groupby('team_id')[['m', 'a']].sum()
    own_rate = (own['m'] + k_off * league) / (own['a'] + k_off)
    allowed = shots.groupby('opp')[['m', 'a']].sum()
    allowed_rate = (allowed['m'] + k_def * league) / (allowed['a'] + k_def)
    expected = (league + (df['team_id'].map(own_rate).fillna(league) - league)
                + (df['opponent_team_id'].map(allowed_rate).fillna(league) - league))
    df['team_score'] = df['team_score'] + 3 * (att * expected - made)
    adjusted = df[['game_id', 'team_id', 'team_score']].rename(
        columns={'team_id': 'opponent_team_id', 'team_score': '_opp_adj'})
    df = df.merge(adjusted, on=['game_id', 'opponent_team_id'], how='left')
    df['opponent_team_score'] = df['_opp_adj'].fillna(df['opponent_team_score'])
    return df.drop(columns='_opp_adj')


def calculate_adjusted_efficiency(team_game_df, raw_efficiency_ratings, iterations=10,
                                  opponent_prior=None, fade_games=None):
    """Iteratively adjust efficiency ratings for opponent quality.

    Each iteration recomputes each game's offensive/defensive efficiency and
    adds the "quality bonus" from facing a tough/weak opponent.  After enough
    iterations (10 is standard) the ratings converge to opponent-adjusted values.

    Parameters
    ----------
    team_game_df         : DataFrame  One row per team per game.
    raw_efficiency_ratings: DataFrame  Starting point from calculate_efficiency().
    iterations            : int        Number of Markov-style adjustment passes.
    opponent_prior        : dict|None  team_id -> {'off_eff', 'def_eff'} (last season).
    fade_games            : int|None   With opponent_prior: each opponent is rated as
                                       share * this-season + (1 - share) * prior, where
                                       share = min(1, opponent's games / fade_games) —
                                       how Team Stats shows them early on. A one- or
                                       two-game solve otherwise goes circular (a 50-point
                                       loss is the only evidence about the loser, so the
                                       winner keeps nearly the whole margin). Once every
                                       team has fade_games games it's the plain solve.
                                       Omitted (the predictor): unchanged behaviour.

    Returns
    -------
    DataFrame with columns: season, team_id, off_eff, def_eff, net_eff.
    """
    current_ratings = raw_efficiency_ratings[['season', 'team_id', 'off_eff', 'def_eff']].copy()
    anchored = opponent_prior is not None and fade_games
    if anchored:
        games_played = team_game_df.groupby(['season', 'team_id']).size().to_dict()

    for _ in range(iterations):
        current_ratings_by_team_season = (
            current_ratings.set_index(['season', 'team_id'])[['off_eff', 'def_eff']]
            .to_dict('index')
        )
        game_adjustments = []
        for _, game_row in team_game_df.iterrows():
            opponent_key    = (game_row['season'], game_row['opponent_team_id'])
            opponent_rating = current_ratings_by_team_season.get(
                opponent_key, {'off_eff': 100, 'def_eff': 100}
            )
            if anchored:
                share = min(1.0, games_played.get(opponent_key, 0) / fade_games)
                if share < 1:
                    prior = opponent_prior.get(game_row['opponent_team_id'], {'off_eff': 100, 'def_eff': 100})
                    opponent_rating = {
                        side: share * opponent_rating[side] + (1 - share) * prior[side]
                        for side in ('off_eff', 'def_eff')
                    }
            team_possessions = (
                game_row['field_goals_attempted']
                + 0.44 * game_row['free_throws_attempted']
                - game_row['offensive_rebounds']
                + game_row['turnovers']
            )
            # "not > 0" (not "<= 0") also skips NaN: ESPN sometimes posts a final
            # score with no box stats (2027: Warren Wilson-NC A&T, 2026-11-03).
            # A team whose only games are like that drops out of the ratings
            # instead of carrying a NaN that breaks the rank casts downstream.
            if not team_possessions > 0:
                continue
            adjusted_offense = (game_row['team_score'] / team_possessions * 100
                                + (100 - opponent_rating['def_eff']))
            adjusted_defense = (game_row['opponent_team_score'] / team_possessions * 100
                                + (100 - opponent_rating['off_eff']))
            game_adjustments.append({
                'season':  game_row['season'],
                'team_id': game_row['team_id'],
                'adjusted_off_per_game': adjusted_offense,
                'adjusted_def_per_game': adjusted_defense,
            })

        adjusted_df = (
            pd.DataFrame(game_adjustments)
            .groupby(['season', 'team_id'])
            .mean().reset_index()
        )
        current_ratings = current_ratings[['season', 'team_id']].merge(
            adjusted_df, on=['season', 'team_id']
        )
        current_ratings.columns = ['season', 'team_id', 'off_eff', 'def_eff']

    current_ratings['net_eff'] = current_ratings['off_eff'] - current_ratings['def_eff']
    return current_ratings


def blend_efficiency_ratings(current_season_efficiency, prior_season_efficiency, game_counts_df,
                             prior_games=None):
    """Bayesian blend of current-season and prior-season efficiency ratings.

    A team with many games this season leans on this year's data; a team with
    few games leans on last year.  The blend weight is:
        w = n / (n + GAME_WEIGHT_CONSTANT)
    where n is the number of games played this season.

    Parameters
    ----------
    current_season_efficiency : DataFrame   season + team_id + off_eff + def_eff
    prior_season_efficiency   : DataFrame   Same structure for the previous season.
    game_counts_df            : DataFrame   Columns: team_id, n (games played this season).
    prior_games               : float | dict | None   k in n/(n+k): a number, a per-team
                                dict (team_id → k), or None for GAME_WEIGHT_CONSTANT.

    Returns
    -------
    DataFrame: team_id, off_eff, def_eff, net_eff (blended values).
    """
    blended = (
        current_season_efficiency
        .merge(game_counts_df, on='team_id')
        .merge(prior_season_efficiency, on='team_id', how='left', suffixes=('', '_prior'))
    )
    blended[['off_eff_prior', 'def_eff_prior']] = (
        blended[['off_eff_prior', 'def_eff_prior']].fillna(100)  # assume average if no prior data
    )
    if prior_games is None:
        k = GAME_WEIGHT_CONSTANT
    elif isinstance(prior_games, dict):
        k = blended['team_id'].map(prior_games).fillna(GAME_WEIGHT_CONSTANT)
    else:
        k = prior_games
    game_weight_fraction = blended['n'] / (blended['n'] + k)
    blended['blended_off_eff'] = (blended['off_eff']       * game_weight_fraction
                                   + blended['off_eff_prior'] * (1 - game_weight_fraction))
    blended['blended_def_eff'] = (blended['def_eff']       * game_weight_fraction
                                   + blended['def_eff_prior'] * (1 - game_weight_fraction))
    blended['net_eff'] = blended['blended_off_eff'] - blended['blended_def_eff']
    return blended[['team_id', 'blended_off_eff', 'blended_def_eff', 'net_eff']].rename(
        columns={'blended_off_eff': 'off_eff', 'blended_def_eff': 'def_eff'}
    )


_PLAYER_MINUTES = {}


def _player_minutes(season):
    """(team_id, athlete_id) → minutes for a season. Memoized: past seasons download
    from sportsdataverse on every load, and the returning/roster features read each
    season several times."""
    if season not in _PLAYER_MINUTES:
        box = feeds.load_player_box(season)
        box = box[box['athlete_id'].notna()].copy()
        box['min'] = pd.to_numeric(box['minutes'], errors='coerce').fillna(0)
        box['athlete_id'] = box['athlete_id'].astype('int64')
        box['team_id'] = box['team_id'].astype('int64')
        _PLAYER_MINUTES[season] = box.groupby(['team_id', 'athlete_id'])['min'].sum().reset_index()
    return _PLAYER_MINUTES[season].copy()


def returning_minutes_share(season):
    """team_id → share of last season's minutes played by players on this season's roster.

    "Roster" = everyone who logs minutes for the team this season — the stand-in
    for the preseason roster, which ESPN doesn't keep historically (a player who
    never suits up is missed). Only minutes for the SAME team count, so transfers
    out lower the share and transfers in don't raise it.
    """
    now, before = _player_minutes(season), _player_minutes(season - 1)
    roster = set(zip(now.loc[now['min'] > 0, 'team_id'], now.loc[now['min'] > 0, 'athlete_id']))
    before['back'] = [(t, a) in roster for t, a in zip(before['team_id'], before['athlete_id'])]
    total = before.groupby('team_id')['min'].sum()
    back = before[before['back']].groupby('team_id')['min'].sum().reindex(total.index, fill_value=0)
    return (back / total.replace(0, np.nan)).dropna().clip(0, 1).to_dict()


ROSTER_FEATURES = ('tin', 'o_in', 'd_in', 'o_ret', 'd_ret', 'o_left', 'd_left')


def roster_production_features(season, cap=None):
    """team_id → last-season production of this season's roster, incl. transfers in.

    All as shares of the team's own last-season minutes (T):
      tin            minutes incoming D-I transfers played last season (elsewhere) / T
      o_in,  d_in    Σ those minutes × the player's last-season O/D RAPM (box prior) / T
      o_ret, d_ret   same for players back from this team
      o_left,d_left  same for last season's players who are gone
    RAPM: mbb_rapm_{yyyy}{yy}.csv for last season (plain RAPM where the box-prior
    columns don't exist, i.e. 2020). Players without a RAPM row count as 0.
    cap: incoming minutes are kept, best player first (O+D RAPM), only until
    returners + transfers fill `cap` × T — a team can only play so many minutes.
    """
    import os
    root = os.path.dirname(os.path.abspath(__file__))

    last = season - 1
    path = os.path.join(root, f"mbb_rapm_{last - 1}{str(last)[2:]}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=['team_id', *ROSTER_FEATURES])
    rapm = pd.read_csv(path)
    o_col, d_col = ('o_rapm_bp', 'd_rapm_bp') if 'o_rapm_bp' in rapm else ('o_rapm', 'd_rapm')
    rapm = rapm.groupby('athlete_id').agg(o=(o_col, 'mean'), d=(d_col, 'mean')).reset_index()

    now, before = _player_minutes(season), _player_minutes(last)
    before = before.merge(rapm, on='athlete_id', how='left').fillna({'o': 0.0, 'd': 0.0})
    before['mo'], before['md'] = before['min'] * before['o'], before['min'] * before['d']
    team_minutes = before.groupby('team_id')['min'].sum()
    roster = now.loc[now['min'] > 0, ['team_id', 'athlete_id']]
    on_roster = set(zip(roster['team_id'], roster['athlete_id']))

    # A player's last season, summed over every team he played for.
    career = before.groupby('athlete_id')[['min', 'mo', 'md']].sum()
    last_teams = before.groupby('athlete_id')['team_id'].agg(set)
    roster = roster.join(career, on='athlete_id').fillna({'min': 0.0, 'mo': 0.0, 'md': 0.0})
    came_back = [t in last_teams.get(a, ()) for t, a in zip(roster['team_id'], roster['athlete_id'])]
    incoming = roster[(roster['min'] > 0) & ~np.array(came_back)]
    returning = before[[(t, a) in on_roster for t, a in zip(before['team_id'], before['athlete_id'])]]
    if cap is not None:
        room = (cap * team_minutes - returning.groupby('team_id')['min'].sum()
                .reindex(team_minutes.index, fill_value=0)).clip(lower=0)
        incoming = incoming.assign(value=(incoming['mo'] + incoming['md']) / incoming['min'].clip(lower=1))
        incoming = incoming.sort_values('value', ascending=False)
        used = incoming.groupby('team_id')['min'].cumsum() - incoming['min']
        keep = (incoming['team_id'].map(room).fillna(0) - used).clip(lower=0)
        frac = (keep / incoming['min']).clip(upper=1)
        incoming = incoming.assign(min=incoming['min'] * frac, mo=incoming['mo'] * frac, md=incoming['md'] * frac)
    departed = before[[(t, a) not in on_roster for t, a in zip(before['team_id'], before['athlete_id'])]]

    out = pd.DataFrame(index=team_minutes[team_minutes > 0].index)
    total = out.index.map(team_minutes)
    for prefix, frame in (('in', incoming), ('ret', returning), ('left', departed)):
        sums = frame.groupby('team_id')[['min', 'mo', 'md']].sum().reindex(out.index, fill_value=0)
        if prefix == 'in':
            out['tin'] = sums['min'].values / total
        out[f'o_{prefix}'] = sums['mo'].values / total
        out[f'd_{prefix}'] = sums['md'].values / total
    return out.reset_index().rename(columns={'index': 'team_id'})


# ==============================================================================
# BasePredictor
# ==============================================================================
class BasePredictor:
    """One spread model (LinearRegression) + one win-prob calibrator (subclass).

    Training is walk-forward: for each month of the calibration year, the model
    is given only games from prior months, so it never sees the future.
    """

    def __init__(self):
        self.spread_regression_model = None   # LinearRegression: features → predicted margin
        self.calibrator              = None   # e.g. IsotonicRegression: margin → P(win)
        self.calibration_years       = None   # list of seasons used to train
        self.current_efficiency      = None   # DataFrame indexed by team_id
        self.team_lookup             = None   # {team_display_name: team_id}
        self.team_id_to_name         = None   # {team_id: team_display_name}
        self.current_season          = None
        self.as_of_date              = None

    # Subclasses implement these two methods.
    def _fit_calibrator(self, predicted_margins, actual_outcomes):
        raise NotImplementedError

    def _calibrate(self, predicted_margin):
        raise NotImplementedError

    def _walk_forward_raw(self, season, prior_season):
        """Collect (actual_outcome, margin, net_eff_diff, location_code) for a season
        via a monthly walk-forward: only games from prior months are used as ratings inputs."""
        season_game_df = feeds.load_team_box_numpy(season)
        season_game_df['month'] = pd.to_datetime(season_game_df['game_date']).dt.to_period('M')
        chronological_months = sorted(season_game_df['month'].unique())

        prior_season_game_df   = feeds.load_team_box_numpy(prior_season)
        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_season_game_df, calculate_efficiency(prior_season_game_df)
        )

        actual_outcomes_list, actual_margins_list, net_efficiency_diffs_list, location_codes_list = (
            [], [], [], []
        )
        for month_index, current_month in enumerate(chronological_months):
            month_games  = season_game_df[season_game_df['month'] == current_month]
            games_before = season_game_df[season_game_df['month'].isin(chronological_months[:month_index])]

            if len(games_before) > 0:
                current_season_efficiency = calculate_adjusted_efficiency(
                    games_before, calculate_efficiency(games_before)
                )
                blended_efficiency = blend_efficiency_ratings(
                    current_season_efficiency, prior_season_efficiency,
                    games_before.groupby('team_id').size().reset_index(name='n')
                )
            else:
                blended_efficiency = prior_season_efficiency[
                    ['team_id', 'off_eff', 'def_eff', 'net_eff']
                ].copy()

            efficiency_lookup = blended_efficiency.set_index('team_id')
            for _, game_row in month_games.iterrows():
                team1_id = game_row['team_id']
                team2_id = game_row['opponent_team_id']
                if team1_id not in efficiency_lookup.index or team2_id not in efficiency_lookup.index:
                    continue
                net_efficiency_diffs_list.append(
                    efficiency_lookup.loc[team1_id]['net_eff']
                    - efficiency_lookup.loc[team2_id]['net_eff']
                )
                location_codes_list.append(
                    {'home': 1, 'neutral': 0, 'away': -1}.get(game_row['team_home_away'], 0)
                )
                actual_outcomes_list.append(int(game_row['team_winner']))
                actual_margins_list.append(game_row['team_score'] - game_row['opponent_team_score'])

        return (
            np.array(actual_outcomes_list),
            np.array(actual_margins_list),
            np.array(net_efficiency_diffs_list),
            np.array(location_codes_list),
        )

    def train(self, calibration_year):
        """Fit the spread model and calibrator on walk-forward data.

        calibration_year: int or list[int].  Multiple years concatenated.
        """
        start_time = time.time()
        model_class_name = type(self).__name__
        calibration_year_list = (
            [calibration_year] if isinstance(calibration_year, int) else list(calibration_year)
        )
        self.calibration_years = calibration_year_list
        print(f"Training {model_class_name} (walk-forward) on {calibration_year_list}...")

        all_actual_outcomes   = []
        all_actual_margins    = []
        all_net_eff_diffs     = []
        all_location_codes    = []
        for cal_year in calibration_year_list:
            year_actuals, year_margins, year_net_diffs, year_locs = (
                self._walk_forward_raw(cal_year, cal_year - 1)
            )
            all_actual_outcomes.append(year_actuals)
            all_actual_margins.append(year_margins)
            all_net_eff_diffs.append(year_net_diffs)
            all_location_codes.append(year_locs)

        all_actual_outcomes = np.concatenate(all_actual_outcomes)
        all_actual_margins  = np.concatenate(all_actual_margins)
        feature_matrix = np.column_stack([
            np.concatenate(all_net_eff_diffs),
            np.concatenate(all_location_codes),
        ])

        self.spread_regression_model = LinearRegression().fit(feature_matrix, all_actual_margins)
        print(
            f"   spread: margin = {self.spread_regression_model.coef_[0]:.3f}*net_diff "
            f"+ {self.spread_regression_model.coef_[1]:+.2f}*loc  "
            f"(home court = {self.spread_regression_model.coef_[1]:+.2f} pts)"
        )
        self._fit_calibrator(
            self.spread_regression_model.predict(feature_matrix), all_actual_outcomes
        )
        print(f"OK {model_class_name} trained in {time.time() - start_time:.1f}s "
              f"({len(all_actual_outcomes)} games)")

    @staticmethod
    def _infer_current_season(timestamp=None):
        """NCAA seasons are labeled by the calendar year they END (spring).
        A date in Jul–Dec belongs to next year's season; Jan–Jun to the current year."""
        timestamp = pd.Timestamp.now() if timestamp is None else pd.Timestamp(timestamp)
        return timestamp.year + 1 if timestamp.month >= 7 else timestamp.year

    @staticmethod
    def _match_team_name(search_name, name_to_id_lookup):
        """Resolve a team name string to its numeric ID.

        Resolution order:
          1. Exact match
          2. Case-insensitive exact match
          3. Best word-subset match (search words ⊆ candidate name words, fewest extra words wins)
          4. None if no hit
        """
        if search_name in name_to_id_lookup:
            return name_to_id_lookup[search_name]
        search_lower = search_name.lower().strip()
        for candidate_name, team_id in name_to_id_lookup.items():
            if candidate_name.lower() == search_lower:
                return team_id
        search_words = set(search_lower.split())
        word_subset_candidates = []
        for candidate_name, team_id in name_to_id_lookup.items():
            candidate_words = set(candidate_name.lower().split())
            if search_words.issubset(candidate_words):
                word_subset_candidates.append((
                    len(candidate_words) - len(search_words),  # fewer extra words = better
                    len(candidate_name),
                    candidate_name,
                    team_id,
                ))
        if word_subset_candidates:
            word_subset_candidates.sort()
            return word_subset_candidates[0][3]
        return None

    def _build_ratings(self, current_season, prior_season=None, as_of_date=None):
        """Load and blend efficiency ratings for the current season."""
        if prior_season is None:
            prior_season = current_season - 1

        current_season_game_df = feeds.load_team_box_numpy(current_season)
        if as_of_date is not None:
            current_season_game_df = current_season_game_df[
                pd.to_datetime(current_season_game_df['game_date']) <= pd.Timestamp(as_of_date)
            ]

        prior_season_game_df    = feeds.load_team_box_numpy(prior_season)
        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_season_game_df, calculate_efficiency(prior_season_game_df)
        )
        current_raw_efficiency = calculate_adjusted_efficiency(
            current_season_game_df, calculate_efficiency(current_season_game_df)
        )
        game_counts_df = (
            current_season_game_df.groupby('team_id').size().reset_index(name='n')
        )
        blended_efficiency = blend_efficiency_ratings(
            current_raw_efficiency, prior_season_efficiency, game_counts_df
        )
        self.current_efficiency = blended_efficiency.set_index('team_id')
        self.team_lookup = (
            current_season_game_df[['team_id', 'team_display_name']]
            .drop_duplicates()
            .set_index('team_display_name')['team_id'].to_dict()
        )
        self.team_id_to_name = {team_id: name for name, team_id in self.team_lookup.items()}
        self.current_season  = current_season
        self.as_of_date      = None if as_of_date is None else pd.Timestamp(as_of_date)
        freshness_note = f" as of {self.as_of_date.date()}" if self.as_of_date is not None else ""
        print(f"OK Loaded {current_season} ratings for {len(self.current_efficiency)} teams{freshness_note}")

    def load_current_ratings(self, prior_season=None):
        """Refresh ratings using every game played to date this season."""
        self._build_ratings(self._infer_current_season(), prior_season=prior_season)

    def get_team_id(self, team_name):
        return self._match_team_name(team_name, self.team_lookup)

    def validate_walk_forward(self, test_season, prior_season=None):
        """Run walk-forward validation and return accuracy/logloss/AUC metrics."""
        if prior_season is None:
            prior_season = test_season - 1
        actual_outcomes, actual_margins, net_eff_diffs, location_codes = (
            self._walk_forward_raw(test_season, prior_season)
        )
        feature_matrix    = np.column_stack([net_eff_diffs, location_codes])
        predicted_spreads = self.spread_regression_model.predict(feature_matrix)
        predicted_win_probs = np.array([self._calibrate(spread) for spread in predicted_spreads])
        validation_metrics = {
            'accuracy':    np.mean((predicted_win_probs > 0.5) == actual_outcomes),
            'logloss':     log_loss(actual_outcomes, predicted_win_probs),
            'auc':         roc_auc_score(actual_outcomes, predicted_win_probs),
            'spread_mae':  np.mean(np.abs(predicted_spreads - actual_margins)),
            'spread_rmse': np.sqrt(np.mean((predicted_spreads - actual_margins) ** 2)),
        }
        return validation_metrics, predicted_win_probs, actual_outcomes


# ==============================================================================
# IsotonicPredictor
# ==============================================================================
class IsotonicPredictor(BasePredictor):
    """Maps predicted margin → P(win) with isotonic (monotone) regression.

    Isotonic regression is guaranteed to be non-decreasing, so a larger
    predicted margin always produces a higher win probability.
    """

    def _fit_calibrator(self, predicted_margins, actual_outcomes):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(predicted_margins, actual_outcomes)

    def _calibrate(self, predicted_margin):
        return float(self.calibrator.predict([predicted_margin])[0])


# ==============================================================================
# TempoPredictor
# ==============================================================================
class TempoPredictor(IsotonicPredictor):
    """Full model adding tempo adjustment, recent form, and per-team home court.

    Three features used in the spread regression:
        tempo_adj  = net_efficiency_diff * pace_factor
                     (efficiency advantage scaled by how fast both teams play)
        home_court = home_team_advantage OR -away_team_advantage
                     (replaces the flat location coefficient in BasePredictor)
        form_diff  = team1_recent_form - team2_recent_form
                     (rolling mean of last FORM_GAMES residual margins)
    """

    SPREAD_FEATURES   = ['tempo_adj', 'home_court', 'form_diff']
    FORM_GAMES        = 5       # rolling window for recent form
    FORM_MODE         = 'residual'   # use opponent-adjusted form; tuned default
    CONF_GATE_GAMES   = 5       # conference games needed before gate/window form turns on
    HOME_ADV_SHRINK   = 30      # Bayesian shrink constant for home-court estimates
    # ESPN's team box labels neutral-site games home/away (only the schedule's
    # neutral_site knows). NEUTRAL_FIX relabels them 'neutral' so training and
    # home-court estimates stop treating them as home games. HOME_ADV_FIX takes the
    # league-average home edge over teams with both home and away games — non-D-I
    # teams that only play road games otherwise drag it to ~15 points.
    NEUTRAL_FIX  = False
    HOME_ADV_FIX = False

    def _load_box(self, season):
        df = feeds.load_team_box_numpy(season)
        if self.NEUTRAL_FIX:
            sched = feeds.load_schedule(season).dropna(subset=['game_id'])
            neutral = set(sched.loc[sched['neutral_site'].astype(bool), 'game_id'].astype('int64'))
            df = df.copy()
            df.loc[df['game_id'].astype('int64').isin(neutral), 'team_home_away'] = 'neutral'
        return df
    HOME_ADV_MIN_GP   = 20      # minimum games to appear in leaderboard

    _conference_game_id_cache = {}  # season → set(conference game_ids), class-level

    # (k_off, k_def) rates teams on 3P%-regressed scores (regress_three_point_shooting);
    # None = actual scores. The class default stays None so studies get a plain
    # baseline; build_artifacts.py turns it on for the site's predictions.
    THREE_PT_REGRESSION = None

    # Last season's weight in the rating blend, w_prior = k / (n + k).
    # PRIOR_GAMES = k (7 = production). RETURNING_PRIOR (experimental, None = off):
    #   {"floor": f, "scale_k": bool} — last season's rating is pulled toward the
    #   league mean by roster turnover: kept share g = f + (1 - f) * returning
    #   minutes share; with scale_k, k is also multiplied by g (fewer "games" of
    #   prior for a turned-over roster).
    PRIOR_GAMES = GAME_WEIGHT_CONSTANT
    RETURNING_PRIOR = None
    _returning_cache = {}

    #   {"mode": "projection", "shift_only": bool} — last season's rating is
    #   replaced by a projection fit on OTHER seasons (no leakage):
    #     this_dev = b*last_dev + c*last_dev*dret + e*dret   (Off and Def separately)
    #   dret = returning minutes share minus that season's average. shift_only
    #   fixes b=1, c=0 (only the up/down shift by returning share).
    PROJECTION_SEASONS = range(2021, 2027)
    SOLVE_CACHE_DIR = None      # set by line_study.py --fast
    _roster_cache = {}
    _ratings_cache = {}
    _projection_rows = {}

    def _season_ratings(self, season):
        df = self._rating_input(feeds.load_team_box_numpy(season))
        ratings = calculate_adjusted_efficiency(df, calculate_efficiency(df))
        ratings['n'] = ratings['team_id'].map(df.groupby('team_id').size())
        return ratings

    def _returning(self, season):
        if season not in self._returning_cache:
            TempoPredictor._returning_cache[season] = returning_minutes_share(season)
        return self._returning_cache[season]

    def _roster(self, season):
        cap = self.RETURNING_PRIOR.get('transfer_cap')
        key = (season, cap)
        if key not in self._roster_cache:
            TempoPredictor._roster_cache[key] = roster_production_features(season, cap=cap)
        return self._roster_cache[key]

    def _projection_terms(self, frame, col):
        """Columns of the projection's design matrix for one side (after the intercept)."""
        dev, dret = frame[col + '_dev'].values, frame['dret'].values
        terms = [dev, dev * dret, dret]
        if self.RETURNING_PRIOR.get('transfers'):
            side = col[0]                                # 'o' / 'd'
            terms += [frame['tin_c'].values, frame[f'{side}_in_c'].values,
                      frame[f'{side}_ret_c'].values, frame[f'{side}_left_c'].values]
        if self.RETURNING_PRIOR.get('history'):
            terms += [frame[col + '_dev2'].values]
        if self.RETURNING_PRIOR.get('elite'):
            # good programs regress less than a straight line says: last season's
            # Net above average (positive part), for both sides
            net = frame['off_eff_dev'].values - frame['def_eff_dev'].values
            terms += [np.clip(net, 0, None)]
        return np.column_stack(terms)

    def _dev_frame(self, prior, share, games, season=None):
        out = self._dev_frame_base(prior, share, games)
        if self.RETURNING_PRIOR and self.RETURNING_PRIOR.get('transfers'):
            roster = self._roster(season)
            out = out.merge(roster, on='team_id', how='left')
            has = out['ret'].notna() & out['tin'].notna()
            out['ret'] = out['ret'].where(has)
            for c in ROSTER_FEATURES:
                out[c + '_c'] = (out[c] - out.loc[has, c].mean()).fillna(0.0)
        if self.RETURNING_PRIOR and self.RETURNING_PRIOR.get('history'):
            # Program history: the rating two seasons back (stand-in for recruiting).
            two_back = self._season_ratings_cached(season - 2)
            d1 = two_back[two_back['n'] >= 15]
            for col in ('off_eff', 'def_eff'):
                dev2 = out['team_id'].map(d1.set_index('team_id')[col] - d1[col].median())
                out[col + '_dev2'] = dev2.fillna(0.0)
        return out

    def _season_ratings_cached(self, season):
        key = (self.THREE_PT_REGRESSION, season)
        if key not in self._ratings_cache:
            TempoPredictor._ratings_cache[key] = self._season_ratings(season)
        return self._ratings_cache[key]

    @staticmethod
    def _dev_frame_base(prior, share, games):
        # D-I only (15+ games last season, roster data): non-D-I opponents carry
        # extreme ratings that would drag the center. Others keep their rating.
        out = prior[['team_id']].copy()
        out['ret'] = out['team_id'].map(share).where(out['team_id'].map(games).fillna(0) >= 15)
        has = out['ret'].notna()
        out['dret'] = (out['ret'] - out.loc[has, 'ret'].mean()).fillna(0.0)
        for col in ('off_eff', 'def_eff'):
            center = prior.loc[has, col].median()
            out[col + '_mean'] = center
            out[col + '_dev'] = prior[col] - center
        return out

    def _projection_coefs(self, exclude):
        rows = []
        for s in self.PROJECTION_SEASONS:
            if s == exclude:
                continue
            ck = (self.THREE_PT_REGRESSION, bool(self.RETURNING_PRIOR.get('transfers')),
                  self.RETURNING_PRIOR.get('transfer_cap'), bool(self.RETURNING_PRIOR.get('history')),
                  bool(self.RETURNING_PRIOR.get('elite')), s)
            if ck not in self._projection_rows:
                prev, cur = self._season_ratings_cached(s - 1), self._season_ratings_cached(s)
                frame = self._dev_frame(prev, self._returning(s), prev.set_index('team_id')['n'], season=s)
                frame = frame.merge(cur[['team_id', 'off_eff', 'def_eff', 'n']], on='team_id')
                frame = frame[(frame['n'] >= 15) & frame['ret'].notna()]
                for col in ('off_eff', 'def_eff'):
                    frame[col + '_cur'] = frame[col] - frame[col].median()
                TempoPredictor._projection_rows[ck] = frame
            rows.append(self._projection_rows[ck])
        data = pd.concat(rows)
        coefs = {}
        for col in ('off_eff', 'def_eff'):
            dev, y = data[col + '_dev'].values, data[col + '_cur'].values
            one = np.ones(len(data))
            if self.RETURNING_PRIOR.get('shift_only'):
                a, e = np.linalg.lstsq(np.c_[one, data['dret'].values], y - dev, rcond=None)[0]
                coefs[col] = (a, 1.0, 0.0, e)
            else:
                X = np.c_[one, self._projection_terms(data, col)]
                coefs[col] = tuple(np.linalg.lstsq(X, y, rcond=None)[0])
        return coefs

    def _prior_adjustment(self, prior_efficiency, season):
        """(adjusted prior ratings, per-team k) for blending into `season`."""
        if not self.RETURNING_PRIOR:
            return prior_efficiency, self.PRIOR_GAMES
        if self.RETURNING_PRIOR.get('mode') == 'projection':
            coefs = self._projection_coefs(exclude=season)
            games = feeds.load_team_box_numpy(season - 1).groupby('team_id').size()
            frame = self._dev_frame(prior_efficiency, self._returning(season), games, season=season)
            adjusted = prior_efficiency.copy()
            has = frame['ret'].notna().values               # no roster data → unchanged
            for col in ('off_eff', 'def_eff'):
                beta = np.asarray(coefs[col])
                proj = (frame[col + '_mean'].values + beta[0]
                        + self._projection_terms(frame, col) @ beta[1:])
                adjusted[col] = np.where(has, proj, prior_efficiency[col].values)
            adjusted['net_eff'] = adjusted['off_eff'] - adjusted['def_eff']
            print(f"    [{season}] projection coefs (a, b, c, e, …): "
                  + "  ".join(f"{k[:3]} {tuple(round(float(v), 3) for v in coefs[k])}" for k in coefs))
            return adjusted, self.PRIOR_GAMES
        share = self._returning(season)
        floor = self.RETURNING_PRIOR.get("floor", 0.0)
        keep = prior_efficiency['team_id'].map(share).fillna(share and np.mean(list(share.values())) or 1.0)
        keep = floor + (1 - floor) * keep
        adjusted = prior_efficiency.copy()
        for col in ('off_eff', 'def_eff'):
            mean = adjusted[col].mean()
            adjusted[col] = mean + (adjusted[col] - mean) * keep
        adjusted['net_eff'] = adjusted['off_eff'] - adjusted['def_eff']
        if self.RETURNING_PRIOR.get("scale_k"):
            k = dict(zip(adjusted['team_id'], self.PRIOR_GAMES * keep))
        else:
            k = self.PRIOR_GAMES
        return adjusted, k

    def _rating_input(self, game_df):
        """The game rows efficiency ratings are solved from."""
        if self.THREE_PT_REGRESSION is None:
            return game_df
        return regress_three_point_shooting(game_df, *self.THREE_PT_REGRESSION)

    def __init__(self):
        super().__init__()
        self.current_pace       = {}    # {team_id: avg_possessions_per_game}
        self.current_form       = {}    # {team_id: recent_form_value}
        self.home_adv           = {}    # {team_id: estimated_home_court_pts}
        self._league_avg_pace   = 70.0  # fallback if no data
        self._league_home_adv   = 3.0   # fallback home-court advantage
        self.total_regression_model = None   # LinearRegression: raw projected total → actual total

    @classmethod
    def _conference_game_ids(cls, season):
        """Return the set of game_ids flagged as conference competition for a season."""
        if season not in cls._conference_game_id_cache:
            schedule = feeds.load_schedule_numpy(season)
            cls._conference_game_id_cache[season] = set(
                schedule.loc[schedule['conference_competition'] == True, 'game_id']
            )
        return cls._conference_game_id_cache[season]

    def _compute_form_value(self, game_log_for_team):
        """Reduce a team's chronological game log to a single recent-form number.

        The game log is a list of dicts: {'margin': float, 'conf': bool, 'resid': float}.
        Returns 0.0 when not enough data is available under the current mode.
        """
        if not game_log_for_team:
            return 0.0
        mode = self.FORM_MODE
        if mode == 'raw':
            recent_values = [entry['margin'] for entry in game_log_for_team][-self.FORM_GAMES:]
        elif mode == 'residual':
            # Residual = actual margin minus what the model predicted → opponent-adjusted form.
            recent_values = [entry['resid'] for entry in game_log_for_team][-self.FORM_GAMES:]
        elif mode == 'conf_gate':
            conf_game_count = sum(entry['conf'] for entry in game_log_for_team)
            if conf_game_count < self.CONF_GATE_GAMES:
                return 0.0
            recent_values = [entry['margin'] for entry in game_log_for_team][-self.FORM_GAMES:]
        elif mode == 'conf_window':
            conference_margins = [entry['margin'] for entry in game_log_for_team if entry['conf']]
            if len(conference_margins) < self.CONF_GATE_GAMES:
                return 0.0
            recent_values = conference_margins[-self.FORM_GAMES:]
        else:
            raise ValueError(f"unknown FORM_MODE: {mode!r}")
        return float(np.mean(recent_values)) if recent_values else 0.0

    @staticmethod
    def _compute_team_pace(team_game_df):
        """Average possessions per game for each team."""
        team_game_df = team_game_df.copy()
        team_game_df['possessions'] = (
            team_game_df['field_goals_attempted']
            + 0.44 * team_game_df['free_throws_attempted']
            - team_game_df['offensive_rebounds']
            + team_game_df['turnovers']
        )
        return team_game_df.groupby('team_id')['possessions'].mean()

    @staticmethod
    def _compute_league_avg_pace(team_game_df):
        """League-average possessions per team per game."""
        possessions = (
            team_game_df['field_goals_attempted']
            + 0.44 * team_game_df['free_throws_attempted']
            - team_game_df['offensive_rebounds']
            + team_game_df['turnovers']
        )
        return float(possessions.mean())

    def _compute_current_form(self, current_season_game_df, conference_game_id_set):
        """Build a form value for every team from the chronological game log."""
        sorted_game_df = current_season_game_df.sort_values('game_date')
        efficiency_lookup = self.current_efficiency
        team_game_logs = {}

        for _, game_row in sorted_game_df.iterrows():
            team1_id = game_row['team_id']
            team2_id = game_row['opponent_team_id']
            actual_margin = float(game_row['team_score'] - game_row['opponent_team_score'])

            if team1_id in efficiency_lookup.index and team2_id in efficiency_lookup.index:
                net_efficiency_diff = (efficiency_lookup.loc[team1_id]['net_eff']
                                       - efficiency_lookup.loc[team2_id]['net_eff'])
                pace_factor = (
                    (self.current_pace.get(team1_id, self._league_avg_pace)
                     + self.current_pace.get(team2_id, self._league_avg_pace)) / 2
                ) / self._league_avg_pace
                location_code = {'home': 1, 'neutral': 0, 'away': -1}.get(game_row['team_home_away'], 0)
                home_court_pts = self._home_court_feature(
                    team1_id, team2_id, location_code, self.home_adv, self._league_home_adv
                )
                model_expected_margin = net_efficiency_diff * pace_factor + home_court_pts
            else:
                model_expected_margin = 0.0

            team_game_logs.setdefault(team1_id, []).append({
                'margin': actual_margin,
                'conf':   bool(game_row['game_id'] in conference_game_id_set),
                'resid':  actual_margin - model_expected_margin,
            })

        return {team_id: self._compute_form_value(log) for team_id, log in team_game_logs.items()}

    @classmethod
    def _compute_home_adv(cls, team_game_df, shrink=None, fix=False):
        """Estimate per-team home-court advantage in points, shrunk toward the league mean.

        raw_adv = (avg home margin - avg away margin) / 2
        shrunk  = raw_adv * w + league_avg * (1 - w),  where w = n / (n + shrink)

        Returns (adv_dict, league_avg_adv).
        """
        if shrink is None:
            shrink = cls.HOME_ADV_SHRINK
        game_df = team_game_df.copy()
        game_df['margin'] = game_df['team_score'] - game_df['opponent_team_score']

        home_stats = (game_df[game_df['team_home_away'] == 'home']
                      .groupby('team_id')['margin']
                      .agg(home_mean='mean', n_home='count'))
        away_stats = (game_df[game_df['team_home_away'] == 'away']
                      .groupby('team_id')['margin']
                      .agg(away_mean='mean', n_away='count'))
        merged_home_away = home_stats.join(away_stats, how='outer').fillna(0)
        merged_home_away['raw_home_adv'] = (merged_home_away['home_mean'] - merged_home_away['away_mean']) / 2
        if fix:
            both = (merged_home_away['n_home'] > 0) & (merged_home_away['n_away'] > 0)
            league_avg_home_adv = float(merged_home_away.loc[both, 'raw_home_adv'].mean())
            merged_home_away.loc[~both, 'raw_home_adv'] = league_avg_home_adv
        else:
            league_avg_home_adv = float(merged_home_away['raw_home_adv'].mean())
        total_games = merged_home_away['n_home'] + merged_home_away['n_away']
        shrink_weight = total_games / (total_games + shrink)
        merged_home_away['shrunk_home_adv'] = (merged_home_away['raw_home_adv'] * shrink_weight
                                               + league_avg_home_adv * (1 - shrink_weight))
        return merged_home_away['shrunk_home_adv'].to_dict(), league_avg_home_adv

    def _home_court_feature(self, team1_id, team2_id, location_code,
                            home_adv_dict, league_avg_home_adv):
        """Return the point value of home court from team1's perspective.

        location_code: 1 = team1 is at home, -1 = team1 is away, 0 = neutral.
        """
        if location_code == 1:
            return home_adv_dict.get(team1_id, league_avg_home_adv)
        elif location_code == -1:
            return -home_adv_dict.get(team2_id, league_avg_home_adv)
        return 0.0

    def _walk_forward_frame(self, season, prior_season, refit='month'):
        """Produce a training DataFrame for one season via walk-forward.

        Returns a DataFrame with one row per game side containing the three model
        features (tempo_adj, home_court, form_diff) plus the actual margin and
        win indicator.

        refit='month' (training default): ratings re-solved at the start of each
        month from every earlier game. refit='day': re-solved before every game
        date, the way the live site rates teams — slower (~130 solves a season),
        used by line_study.py to compare point-in-time predictions with the book.
        """
        season_game_df = self._load_box(season)
        season_game_df['month'] = pd.to_datetime(season_game_df['game_date']).dt.to_period(
            'M' if refit == 'month' else 'D')   # the refit period ("month" kept as the name)
        season_game_df = season_game_df.sort_values('game_date')
        chronological_months = sorted(season_game_df['month'].unique())

        conference_game_id_set = self._conference_game_ids(season)

        prior_season_game_df    = self._load_box(prior_season)
        prior_rating_df         = self._rating_input(prior_season_game_df)
        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_rating_df, calculate_efficiency(prior_rating_df)
        )
        prior_season_efficiency, prior_k = self._prior_adjustment(prior_season_efficiency, season)
        prior_season_pace       = self._compute_team_pace(prior_season_game_df)
        league_avg_pace         = self._compute_league_avg_pace(prior_season_game_df)
        prior_home_adv_dict, league_home_adv = self._compute_home_adv(prior_season_game_df, fix=self.HOME_ADV_FIX)

        # Optional on-disk cache of the in-season solves (SOLVE_CACHE_DIR): they
        # depend only on this season's games and the 3P setting, never on the
        # prior, so prior-weight experiments can replay them instead of re-solving.
        solve_cache, solve_cache_path, solve_cache_dirty = {}, None, False
        if self.SOLVE_CACHE_DIR:
            import os, pickle
            os.makedirs(self.SOLVE_CACHE_DIR, exist_ok=True)
            three = self.THREE_PT_REGRESSION
            solve_cache_path = os.path.join(
                self.SOLVE_CACHE_DIR,
                f"solves_{season}_{refit}_{'none' if not three else f'{three[0]}-{three[1]}'}.pkl")
            if os.path.exists(solve_cache_path):
                with open(solve_cache_path, 'rb') as handle:
                    solve_cache = pickle.load(handle)

        team_form_logs = {}    # {team_id: list of game log entries}
        training_rows  = []
        # A game appears twice (once per side). Results are held until the date
        # changes so neither row's form can see this game's own margin.
        pending_log_entries = []   # (team_id, entry) for current_date
        current_date        = None

        def _flush_pending_log_entries():
            for pending_team_id, pending_entry in pending_log_entries:
                team_form_logs.setdefault(pending_team_id, []).append(pending_entry)
            pending_log_entries.clear()

        for month_index, current_month in enumerate(chronological_months):
            month_games  = season_game_df[season_game_df['month'] == current_month].sort_values('game_date')
            games_before = season_game_df[season_game_df['month'].isin(chronological_months[:month_index])]

            if len(games_before) > 0:
                if month_index in solve_cache:
                    current_raw_efficiency = solve_cache[month_index]
                else:
                    rating_df = self._rating_input(games_before)
                    current_raw_efficiency = calculate_adjusted_efficiency(
                        rating_df, calculate_efficiency(rating_df)
                    )
                    solve_cache[month_index], solve_cache_dirty = current_raw_efficiency, True
                games_played_per_team = games_before.groupby('team_id').size()
                blended_efficiency = blend_efficiency_ratings(
                    current_raw_efficiency, prior_season_efficiency,
                    games_played_per_team.reset_index(name='n'), prior_games=prior_k
                )
                current_season_pace = self._compute_team_pace(games_before)

                def _blended_pace(team_id,
                                  _curr=current_season_pace,
                                  _prior=prior_season_pace,
                                  _counts=games_played_per_team,
                                  _league=league_avg_pace):
                    n = _counts.get(team_id, 0)
                    w = n / (n + GAME_WEIGHT_CONSTANT)
                    return (float(_curr.get(team_id, _league)) * w
                            + float(_prior.get(team_id, _league)) * (1 - w))
            else:
                blended_efficiency = prior_season_efficiency[
                    ['team_id', 'off_eff', 'def_eff', 'net_eff']
                ].copy()

                def _blended_pace(team_id, _prior=prior_season_pace, _league=league_avg_pace):
                    return float(_prior.get(team_id, _league))

            efficiency_lookup = blended_efficiency.set_index('team_id')
            # Same league-average defense the site uses for projected scores
            # (js/predict.js leagueDef: the mean def_eff over rated teams).
            league_def_eff = float(efficiency_lookup['def_eff'].mean())

            for _, game_row in month_games.iterrows():
                if game_row['game_date'] != current_date:
                    _flush_pending_log_entries()
                    current_date = game_row['game_date']

                team1_id = game_row['team_id']
                team2_id = game_row['opponent_team_id']
                if team1_id not in efficiency_lookup.index or team2_id not in efficiency_lookup.index:
                    continue

                net_efficiency_diff = (efficiency_lookup.loc[team1_id]['net_eff']
                                       - efficiency_lookup.loc[team2_id]['net_eff'])
                expected_pace = (_blended_pace(team1_id) + _blended_pace(team2_id)) / 2
                pace_factor   = expected_pace / league_avg_pace
                tempo_adj     = net_efficiency_diff * pace_factor

                location_code = {'home': 1, 'neutral': 0, 'away': -1}.get(game_row['team_home_away'], 0)
                home_court_pts = self._home_court_feature(
                    team1_id, team2_id, location_code, prior_home_adv_dict, league_home_adv
                )

                team1_form = self._compute_form_value(team_form_logs.get(team1_id, []))
                team2_form = self._compute_form_value(team_form_logs.get(team2_id, []))

                actual_margin         = game_row['team_score'] - game_row['opponent_team_score']
                model_expected_margin = net_efficiency_diff * pace_factor + home_court_pts

                # Raw projected total, exactly as js/predict.js projScores builds it.
                team1_eff, team2_eff = efficiency_lookup.loc[team1_id], efficiency_lookup.loc[team2_id]
                projected_total = ((team1_eff['off_eff'] + team2_eff['def_eff'] - league_def_eff)
                                   + (team2_eff['off_eff'] + team1_eff['def_eff'] - league_def_eff)) \
                                  / 100 * expected_pace

                training_rows.append({
                    'game_id':     game_row['game_id'],
                    'team_id':     team1_id,
                    'opponent_team_id': team2_id,
                    'game_date':   game_row['game_date'],
                    'tempo_adj':   tempo_adj,
                    'net_diff':    net_efficiency_diff,
                    'loc':         location_code,
                    'home_court':  home_court_pts,
                    'pace_factor': pace_factor,
                    'form_diff':   team1_form - team2_form,
                    'margin':      actual_margin,
                    'won':         int(game_row['team_winner']),
                    'proj_total':  projected_total,
                    'total':       game_row['team_score'] + game_row['opponent_team_score'],
                    # First walk-forward month: ratings are last season's alone,
                    # which the live model never is (it blends in current games).
                    'prior_only':  len(games_before) == 0,
                })
                pending_log_entries.append((team1_id, {
                    'margin': float(actual_margin),
                    'conf':   bool(game_row['game_id'] in conference_game_id_set),
                    'resid':  float(actual_margin - model_expected_margin),
                }))

            _flush_pending_log_entries()

        if solve_cache_path and solve_cache_dirty:
            with open(solve_cache_path, 'wb') as handle:
                pickle.dump(solve_cache, handle)
        return pd.DataFrame(training_rows)

    def train(self, calibration_year):
        """Fit spread model + calibrator using walk-forward training data."""
        start = time.time()
        calibration_year_list = (
            [calibration_year] if isinstance(calibration_year, int) else list(calibration_year)
        )
        self.calibration_years = calibration_year_list
        print(f"Training TempoPredictor (walk-forward, form={self.FORM_MODE}) on {calibration_year_list}...")

        training_frame = pd.concat(
            [self._walk_forward_frame(cal_year, cal_year - 1) for cal_year in calibration_year_list],
            ignore_index=True
        )
        self.spread_regression_model = LinearRegression().fit(
            training_frame[self.SPREAD_FEATURES], training_frame['margin']
        )
        print("   spread: " + "  ".join(
            f"{feature_name}={coef:+.4f}"
            for feature_name, coef in zip(self.SPREAD_FEATURES, self.spread_regression_model.coef_)
        ))
        self._fit_calibrator(
            self.spread_regression_model.predict(training_frame[self.SPREAD_FEATURES]),
            training_frame['won'].values
        )
        # Projected totals ran ~6 pts low against finals in the live log (Nov 2026):
        # the raw formula is never fit to actual scores the way the spread is.
        # Prior-only rows are left out: their projections are far staler than any
        # live one, and including them flattens the fit (slope 0.63 vs 0.84; held-out
        # 2026 Dec-Apr MAE 14.26 vs 14.12, live log 13.55 either way vs 14.67 raw).
        totals_frame = training_frame[~training_frame['prior_only']].dropna(subset=['proj_total', 'total'])
        self.total_regression_model = LinearRegression().fit(totals_frame[['proj_total']], totals_frame['total'])
        print(f"   total: actual = {self.total_regression_model.intercept_:+.2f} "
              f"+ {self.total_regression_model.coef_[0]:.4f} * projected")
        print(f"OK TempoPredictor trained in {time.time() - start:.1f}s ({len(training_frame)} games)")

    def validate_walk_forward(self, test_season, prior_season=None):
        """Evaluate walk-forward accuracy on a held-out season."""
        if prior_season is None:
            prior_season = test_season - 1
        validation_frame = self._walk_forward_frame(test_season, prior_season)
        predicted_spreads     = self.spread_regression_model.predict(
            validation_frame[self.SPREAD_FEATURES]
        )
        predicted_win_probs   = np.array([self._calibrate(s) for s in predicted_spreads])
        actual_outcomes       = validation_frame['won'].values
        actual_margins        = validation_frame['margin'].values
        return {
            'accuracy':    np.mean((predicted_win_probs > 0.5) == actual_outcomes),
            'logloss':     log_loss(actual_outcomes, predicted_win_probs),
            'auc':         roc_auc_score(actual_outcomes, predicted_win_probs),
            'spread_mae':  np.mean(np.abs(predicted_spreads - actual_margins)),
            'spread_rmse': np.sqrt(np.mean((predicted_spreads - actual_margins) ** 2)),
            'total_mae_raw': np.mean(np.abs(validation_frame['proj_total'] - validation_frame['total'])),
            'total_mae':   np.mean(np.abs(self.total_regression_model.predict(validation_frame[['proj_total']])
                                          - validation_frame['total'])),
        }, predicted_win_probs, actual_outcomes

    def _build_ratings(self, current_season, prior_season=None, as_of_date=None):
        """Extend BasePredictor._build_ratings to also compute pace, home court, and form."""
        if prior_season is None:
            prior_season = current_season - 1

        current_season_game_df = self._load_box(current_season)
        if as_of_date is not None:
            current_season_game_df = current_season_game_df[
                pd.to_datetime(current_season_game_df['game_date']) <= pd.Timestamp(as_of_date)
            ]
        prior_season_game_df    = self._load_box(prior_season)

        prior_rating_df   = self._rating_input(prior_season_game_df)
        current_rating_df = self._rating_input(current_season_game_df)
        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_rating_df, calculate_efficiency(prior_rating_df)
        )
        prior_season_efficiency, prior_k = self._prior_adjustment(prior_season_efficiency, current_season)
        current_raw_efficiency  = calculate_adjusted_efficiency(
            current_rating_df, calculate_efficiency(current_rating_df)
        )
        game_counts_per_team = current_season_game_df.groupby('team_id').size()
        blended_efficiency   = blend_efficiency_ratings(
            current_raw_efficiency, prior_season_efficiency,
            game_counts_per_team.reset_index(name='n'), prior_games=prior_k
        )
        self.current_efficiency = blended_efficiency.set_index('team_id')
        self.team_lookup = (
            current_season_game_df[['team_id', 'team_display_name']]
            .drop_duplicates()
            .set_index('team_display_name')['team_id'].to_dict()
        )
        self.team_id_to_name = {team_id: name for name, team_id in self.team_lookup.items()}
        self.current_season  = current_season
        self.as_of_date      = None if as_of_date is None else pd.Timestamp(as_of_date)

        # Pace
        self.current_pace = {}
        self.home_adv     = {}
        prior_season_pace    = self._compute_team_pace(prior_season_game_df)
        current_season_pace  = self._compute_team_pace(current_season_game_df)
        self._league_avg_pace = self._compute_league_avg_pace(prior_season_game_df)

        for team_id in set(current_season_pace.index) | set(prior_season_pace.index):
            n = game_counts_per_team.get(team_id, 0)
            w = n / (n + GAME_WEIGHT_CONSTANT)
            self.current_pace[team_id] = (
                float(current_season_pace.get(team_id, self._league_avg_pace)) * w
                + float(prior_season_pace.get(team_id, self._league_avg_pace)) * (1 - w)
            )

        # Home-court advantage
        prior_home_adv_dict, prior_league_home_adv = self._compute_home_adv(prior_season_game_df, fix=self.HOME_ADV_FIX)
        curr_home_adv_dict,  curr_league_home_adv  = self._compute_home_adv(current_season_game_df, fix=self.HOME_ADV_FIX)
        self._league_home_adv = curr_league_home_adv
        for team_id in set(prior_home_adv_dict) | set(curr_home_adv_dict):
            n = game_counts_per_team.get(team_id, 0)
            w = n / (n + self.HOME_ADV_SHRINK)
            self.home_adv[team_id] = (
                curr_home_adv_dict.get(team_id, prior_league_home_adv) * w
                + prior_home_adv_dict.get(team_id, prior_league_home_adv) * (1 - w)
            )

        # Form
        conference_game_id_set = self._conference_game_ids(current_season)
        self.current_form = self._compute_current_form(current_season_game_df, conference_game_id_set)

        freshness_note = f" as of {self.as_of_date.date()}" if self.as_of_date is not None else ""
        print(
            f"OK Loaded {current_season} ratings + pace + form "
            f"(L{self.FORM_GAMES}, {self.FORM_MODE}) "
            f"+ home court for {len(self.current_efficiency)} teams{freshness_note}"
        )

    def predict_game(self, team1_name, team2_name, team1_home=True, neutral_site=False,
                     verbose=True, save_output=False):
        """Predict the outcome of a single game.

        Parameters
        ----------
        team1_name   : str   Display name of team 1 (listed first).
        team2_name   : str   Display name of team 2.
        team1_home   : bool  True if team 1 is at home.
        neutral_site : bool  True if playing at a neutral venue.

        Returns
        -------
        dict with win probabilities, projected spread, efficiency ratings, etc.
        """
        team1_id = self.get_team_id(team1_name)
        team2_id = self.get_team_id(team2_name)
        if team1_id is None:
            raise ValueError(f"Team not found: '{team1_name}'")
        if team2_id is None:
            raise ValueError(f"Team not found: '{team2_name}'")

        team1_display_name = self.team_id_to_name[team1_id]
        team2_display_name = self.team_id_to_name[team2_id]
        team1_efficiency   = self.current_efficiency.loc[team1_id]
        team2_efficiency   = self.current_efficiency.loc[team2_id]

        location_code       = 0 if neutral_site else (1 if team1_home else -1)
        net_efficiency_diff = team1_efficiency['net_eff'] - team2_efficiency['net_eff']
        team1_pace          = self.current_pace.get(team1_id, self._league_avg_pace)
        team2_pace          = self.current_pace.get(team2_id, self._league_avg_pace)
        expected_pace       = (team1_pace + team2_pace) / 2
        pace_factor         = expected_pace / self._league_avg_pace
        tempo_adj           = net_efficiency_diff * pace_factor

        team1_form = self.current_form.get(team1_id, 0.0)
        team2_form = self.current_form.get(team2_id, 0.0)
        form_diff  = team1_form - team2_form

        home_court_pts = self._home_court_feature(
            team1_id, team2_id, location_code, self.home_adv, self._league_home_adv
        )
        predicted_spread = self.spread_regression_model.predict(
            np.array([[tempo_adj, home_court_pts, form_diff]])
        )[0]
        team1_win_probability = self._calibrate(predicted_spread)

        prediction_result = {
            'team1': team1_display_name, 'team2': team2_display_name,
            'team1_home': team1_home, 'neutral_site': neutral_site,
            'expected_pace':   round(expected_pace, 1),
            'pace_factor':     round(pace_factor, 3),
            'team1_form':      round(team1_form, 1),
            'team2_form':      round(team2_form, 1),
            'home_court_pts':  round(home_court_pts, 2),
            'team1_win_prob':  team1_win_probability,
            'team2_win_prob':  1 - team1_win_probability,
            'team1_spread':    predicted_spread,
            'team2_spread':    -predicted_spread,
            'team1_net_eff':   team1_efficiency['net_eff'],
            'team2_net_eff':   team2_efficiency['net_eff'],
            'team1_off_eff':   team1_efficiency['off_eff'],
            'team1_def_eff':   team1_efficiency['def_eff'],
            'team2_off_eff':   team2_efficiency['off_eff'],
            'team2_def_eff':   team2_efficiency['def_eff'],
        }
        if verbose:
            venue_str  = ("Neutral" if neutral_site
                          else (f"{team1_display_name} (Home)" if team1_home
                                else f"{team2_display_name} (Home)"))
            favorite   = team1_display_name if team1_win_probability > 0.5 else team2_display_name
            fav_prob   = max(team1_win_probability, 1 - team1_win_probability)
            fav_spread = predicted_spread if team1_win_probability > 0.5 else -predicted_spread
            print(f"\n{'='*60}")
            print(f"{team1_display_name} vs {team2_display_name}  |  {venue_str}  "
                  f"|  pace {expected_pace:.1f} ({pace_factor:+.2%} vs avg)")
            print(f"   Ratings  - {team1_display_name}: {team1_efficiency['net_eff']:+.1f}  "
                  f"{team2_display_name}: {team2_efficiency['net_eff']:+.1f}")
            print(f"   Form L{self.FORM_GAMES} ({self.FORM_MODE}) - "
                  f"{team1_display_name}: {team1_form:+.1f}  {team2_display_name}: {team2_form:+.1f}")
            print(f"   {team1_display_name}: {team1_win_probability*100:.1f}%   "
                  f"{team2_display_name}: {(1-team1_win_probability)*100:.1f}%")
            print(f"   Favorite: {favorite} by {abs(fav_spread):.1f}  ({fav_prob*100:.1f}%)")
            print(f"{'='*60}\n")
        return prediction_result


# ==============================================================================
# VARIABLE GLOSSARY
# ==============================================================================
#
# GAME_WEIGHT_CONSTANT         int    Bayesian shrink constant k; weight = n/(n+k).
#                                     Higher k → lean more on prior season data.
#
# --- calculate_efficiency() ---
# division_one_team_ids        ndarray  All team_ids in the dataset (proxy for D-I membership).
# team_season_stats            DataFrame  Season totals: pts, FGA, FTA, ORB, TOV per team.
# opponent_season_stats        DataFrame  Same columns but from the opponent's perspective.
# combined_stats               DataFrame  Merged frame with both team and opponent totals.
# off_poss                     Series  Estimated offensive possessions = FGA + 0.44*FTA - ORB + TOV.
# def_poss                     Series  Estimated defensive possessions (same formula, opponent's stats).
# off_eff                      Series  Points scored per 100 offensive possessions.
# def_eff                      Series  Points allowed per 100 defensive possessions.
#
# --- calculate_adjusted_efficiency() ---
# current_ratings              DataFrame  Ratings updated each iteration.
# current_ratings_by_team_season dict    Lookup {(season, team_id): {off_eff, def_eff}} for speed.
# opponent_key                 tuple   (season, opponent_team_id) for dict lookup.
# opponent_rating              dict    The opponent's current off/def efficiency.
# team_possessions             float   Possessions for a single game row.
# adjusted_offense             float   adj_off = actual_off_rtg + (100 - opp_def_eff).
# adjusted_defense             float   adj_def = actual_def_rtg + (100 - opp_off_eff).
# game_adjustments             list    One dict per eligible game row.
# adjusted_df                  DataFrame  Per-team mean adjusted values for this iteration.
#
# --- blend_efficiency_ratings() ---
# game_weight_fraction         Series  w = n / (n + k); fraction of weight on current-season data.
# blended_off_eff              Series  Weighted average of current and prior offensive efficiency.
# blended_def_eff              Series  Weighted average of current and prior defensive efficiency.
#
# --- BasePredictor ---
# spread_regression_model      LinearRegression  Predicts actual margin from features.
# calibrator                   IsotonicRegression  Maps predicted margin → P(win).
# calibration_years            list[int]  Seasons the model was trained on.
# current_efficiency           DataFrame (indexed by team_id)  Blended eff ratings.
# team_lookup                  dict  {team_display_name: team_id}.
# team_id_to_name              dict  {team_id: team_display_name}.
# current_season               int   The season for which ratings are loaded.
# as_of_date                   Timestamp or None  Rating cutoff date.
#
# _walk_forward_raw():
# season_game_df               DataFrame  All games in the target season.
# chronological_months         list[Period]  Months in sorted order.
# prior_season_efficiency      DataFrame  Adjusted efficiency from the prior season.
# actual_outcomes_list         list[int]   1 = team1 won, 0 = team1 lost.
# actual_margins_list          list[int]   Final score margin (team1 - team2).
# net_efficiency_diffs_list    list[float]  team1_net_eff - team2_net_eff at prediction time.
# location_codes_list          list[int]   1 = home, 0 = neutral, -1 = away for team1.
# games_before                 DataFrame  All games from months before the current month.
# blended_efficiency           DataFrame  Bayesian blend used as the rating at this point in time.
# efficiency_lookup            DataFrame (indexed)  Fast team-id → eff lookup.
#
# train():
# calibration_year_list        list[int]  Always a list (even if one year given).
# all_actual_outcomes          ndarray  Concatenated across all calibration years.
# all_actual_margins           ndarray  Concatenated actual margins.
# feature_matrix               ndarray  Shape (n_games, 2): [net_diff, location_code].
# year_actuals / year_margins  ndarray  Outputs for a single calendar year.
# year_net_diffs / year_locs   ndarray  Outputs for a single calendar year.
#
# _match_team_name():
# search_lower                 str   Lowercase stripped version of input name.
# search_words                 set   Words in the search name for subset matching.
# word_subset_candidates       list  Tuples of (extra_words, name_len, name, team_id).
#
# --- TempoPredictor ---
# SPREAD_FEATURES              list[str]  ['tempo_adj', 'home_court', 'form_diff'].
# FORM_GAMES                   int   Rolling window length for recent-form calculation.
# FORM_MODE                    str   'raw', 'residual', 'conf_gate', or 'conf_window'.
# CONF_GATE_GAMES              int   Min conf games before conf-gated form activates.
# HOME_ADV_SHRINK              int   Bayesian k constant for home-court shrinkage.
# current_pace                 dict  {team_id: blended avg possessions per game}.
# current_form                 dict  {team_id: scalar form value under FORM_MODE}.
# home_adv                     dict  {team_id: home-court advantage in points}.
# _league_avg_pace             float  League-wide average possessions per game.
# _league_home_adv             float  League-average home-court advantage.
#
# _compute_form_value():
# game_log_for_team            list[dict]  Each dict: {margin, conf, resid}.
# recent_values                list[float]  The last FORM_GAMES values to average.
#
# _compute_home_adv():
# home_stats                   DataFrame  Mean margin + count for home games per team.
# away_stats                   DataFrame  Mean margin + count for away games per team.
# merged_home_away             DataFrame  Joined home/away stats.
# raw_home_adv                 Series  (home_mean - away_mean) / 2 per team.
# league_avg_home_adv          float  Mean of raw_home_adv across all teams.
# shrink_weight                Series  w = total_games / (total_games + HOME_ADV_SHRINK).
# shrunk_home_adv              Series  Bayesian estimate of home-court advantage.
#
# predict_game():
# team1_id / team2_id          int   Numeric ESPN team IDs.
# team1_display_name / team2_display_name  str  Resolved team names.
# team1_efficiency / team2_efficiency      Series  off_eff, def_eff, net_eff for each team.
# location_code                int   1 home / 0 neutral / -1 away for team1.
# net_efficiency_diff          float  team1_net_eff - team2_net_eff.
# team1_pace / team2_pace      float  Each team's blended avg possessions per game.
# expected_pace                float  Average of both teams' pace.
# pace_factor                  float  expected_pace / league_avg_pace (>1 = faster game).
# tempo_adj                    float  net_efficiency_diff * pace_factor (main spread driver).
# team1_form / team2_form      float  Recent form scalar for each team.
# form_diff                    float  team1_form - team2_form.
# home_court_pts               float  Points from home-court advantage (from team1's perspective).
# predicted_spread             float  Model output: expected margin (team1 - team2).
# team1_win_probability        float  P(team1 wins) after isotonic calibration.
# prediction_result            dict   Full result including spreads, probs, and ratings.