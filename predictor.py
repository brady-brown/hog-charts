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
    Possessions estimated with the standard formula:
        FGA + 0.475 * FTA - ORB + TOV

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
        + 0.475 * combined_stats['free_throws_attempted']
        - combined_stats['offensive_rebounds']
        + combined_stats['turnovers']
    ).replace(0, 1)    # prevent division by zero

    combined_stats['def_poss'] = (
        combined_stats['opp_fga']
        + 0.475 * combined_stats['opp_fta']
        - combined_stats['opp_orb']
        + combined_stats['opp_tov']
    ).replace(0, 1)

    combined_stats['off_eff'] = (combined_stats['team_score']     / combined_stats['off_poss'] * 100).clip(70, 140)
    combined_stats['def_eff'] = (combined_stats['opponent_team_score'] / combined_stats['def_poss'] * 100).clip(70, 140)
    return combined_stats


def calculate_adjusted_efficiency(team_game_df, raw_efficiency_ratings, iterations=10):
    """Iteratively adjust efficiency ratings for opponent quality.

    Each iteration recomputes each game's offensive/defensive efficiency and
    adds the "quality bonus" from facing a tough/weak opponent.  After enough
    iterations (10 is standard) the ratings converge to opponent-adjusted values.

    Parameters
    ----------
    team_game_df         : DataFrame  One row per team per game.
    raw_efficiency_ratings: DataFrame  Starting point from calculate_efficiency().
    iterations            : int        Number of Markov-style adjustment passes.

    Returns
    -------
    DataFrame with columns: season, team_id, off_eff, def_eff, net_eff.
    """
    current_ratings = raw_efficiency_ratings[['season', 'team_id', 'off_eff', 'def_eff']].copy()

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
            team_possessions = (
                game_row['field_goals_attempted']
                + 0.475 * game_row['free_throws_attempted']
                - game_row['offensive_rebounds']
                + game_row['turnovers']
            )
            if team_possessions <= 0:
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


def blend_efficiency_ratings(current_season_efficiency, prior_season_efficiency, game_counts_df):
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
    game_weight_fraction = blended['n'] / (blended['n'] + GAME_WEIGHT_CONSTANT)
    blended['blended_off_eff'] = (blended['off_eff']       * game_weight_fraction
                                   + blended['off_eff_prior'] * (1 - game_weight_fraction))
    blended['blended_def_eff'] = (blended['def_eff']       * game_weight_fraction
                                   + blended['def_eff_prior'] * (1 - game_weight_fraction))
    blended['net_eff'] = blended['blended_off_eff'] - blended['blended_def_eff']
    return blended[['team_id', 'blended_off_eff', 'blended_def_eff', 'net_eff']].rename(
        columns={'blended_off_eff': 'off_eff', 'blended_def_eff': 'def_eff'}
    )


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
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        season_game_df = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
        season_game_df['month'] = pd.to_datetime(season_game_df['game_date']).dt.to_period('M')
        chronological_months = sorted(season_game_df['month'].unique())

        prior_season_game_df   = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
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
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if prior_season is None:
            prior_season = current_season - 1

        current_season_game_df = mbb_loaders.load_mbb_team_boxscore(
            seasons=[current_season]
        ).to_pandas()
        if as_of_date is not None:
            current_season_game_df = current_season_game_df[
                pd.to_datetime(current_season_game_df['game_date']) <= pd.Timestamp(as_of_date)
            ]

        prior_season_game_df    = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
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
    HOME_ADV_MIN_GP   = 20      # minimum games to appear in leaderboard

    _conference_game_id_cache = {}  # season → set(conference game_ids), class-level

    def __init__(self):
        super().__init__()
        self.current_pace       = {}    # {team_id: avg_possessions_per_game}
        self.current_form       = {}    # {team_id: recent_form_value}
        self.home_adv           = {}    # {team_id: estimated_home_court_pts}
        self._league_avg_pace   = 70.0  # fallback if no data
        self._league_home_adv   = 3.0   # fallback home-court advantage

    @classmethod
    def _conference_game_ids(cls, season):
        """Return the set of game_ids flagged as conference competition for a season."""
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if season not in cls._conference_game_id_cache:
            schedule = mbb_loaders.load_mbb_schedule(seasons=[season]).to_pandas()
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
            + 0.475 * team_game_df['free_throws_attempted']
            - team_game_df['offensive_rebounds']
            + team_game_df['turnovers']
        )
        return team_game_df.groupby('team_id')['possessions'].mean()

    @staticmethod
    def _compute_league_avg_pace(team_game_df):
        """League-average possessions per team per game."""
        possessions = (
            team_game_df['field_goals_attempted']
            + 0.475 * team_game_df['free_throws_attempted']
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
    def _compute_home_adv(cls, team_game_df, shrink=None):
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

    def _walk_forward_frame(self, season, prior_season):
        """Produce a training DataFrame for one season via monthly walk-forward.

        Returns a DataFrame with one row per game containing the three model
        features (tempo_adj, home_court, form_diff) plus the actual margin and
        win indicator.
        """
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        season_game_df = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
        season_game_df['month'] = pd.to_datetime(season_game_df['game_date']).dt.to_period('M')
        season_game_df = season_game_df.sort_values('game_date')
        chronological_months = sorted(season_game_df['month'].unique())

        conference_game_id_set = self._conference_game_ids(season)

        prior_season_game_df    = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_season_game_df, calculate_efficiency(prior_season_game_df)
        )
        prior_season_pace       = self._compute_team_pace(prior_season_game_df)
        league_avg_pace         = self._compute_league_avg_pace(prior_season_game_df)
        prior_home_adv_dict, league_home_adv = self._compute_home_adv(prior_season_game_df)

        team_form_logs = {}    # {team_id: list of game log entries}
        training_rows  = []

        for month_index, current_month in enumerate(chronological_months):
            month_games  = season_game_df[season_game_df['month'] == current_month].sort_values('game_date')
            games_before = season_game_df[season_game_df['month'].isin(chronological_months[:month_index])]

            if len(games_before) > 0:
                current_raw_efficiency = calculate_adjusted_efficiency(
                    games_before, calculate_efficiency(games_before)
                )
                games_played_per_team = games_before.groupby('team_id').size()
                blended_efficiency = blend_efficiency_ratings(
                    current_raw_efficiency, prior_season_efficiency,
                    games_played_per_team.reset_index(name='n')
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

            for _, game_row in month_games.iterrows():
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

                training_rows.append({
                    'game_date':   game_row['game_date'],
                    'tempo_adj':   tempo_adj,
                    'net_diff':    net_efficiency_diff,
                    'loc':         location_code,
                    'home_court':  home_court_pts,
                    'pace_factor': pace_factor,
                    'form_diff':   team1_form - team2_form,
                    'margin':      actual_margin,
                    'won':         int(game_row['team_winner']),
                })
                team_form_logs.setdefault(team1_id, []).append({
                    'margin': float(actual_margin),
                    'conf':   bool(game_row['game_id'] in conference_game_id_set),
                    'resid':  float(actual_margin - model_expected_margin),
                })

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
        }, predicted_win_probs, actual_outcomes

    def _build_ratings(self, current_season, prior_season=None, as_of_date=None):
        """Extend BasePredictor._build_ratings to also compute pace, home court, and form."""
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if prior_season is None:
            prior_season = current_season - 1

        current_season_game_df = mbb_loaders.load_mbb_team_boxscore(
            seasons=[current_season]
        ).to_pandas()
        if as_of_date is not None:
            current_season_game_df = current_season_game_df[
                pd.to_datetime(current_season_game_df['game_date']) <= pd.Timestamp(as_of_date)
            ]
        prior_season_game_df    = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()

        prior_season_efficiency = calculate_adjusted_efficiency(
            prior_season_game_df, calculate_efficiency(prior_season_game_df)
        )
        current_raw_efficiency  = calculate_adjusted_efficiency(
            current_season_game_df, calculate_efficiency(current_season_game_df)
        )
        game_counts_per_team = current_season_game_df.groupby('team_id').size()
        blended_efficiency   = blend_efficiency_ratings(
            current_raw_efficiency, prior_season_efficiency,
            game_counts_per_team.reset_index(name='n')
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
        prior_home_adv_dict, prior_league_home_adv = self._compute_home_adv(prior_season_game_df)
        curr_home_adv_dict,  curr_league_home_adv  = self._compute_home_adv(current_season_game_df)
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
# off_poss                     Series  Estimated offensive possessions = FGA + 0.475*FTA - ORB + TOV.
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