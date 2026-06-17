"""
predictor.py — tempo-adjusted college basketball game predictor.

Extracted verbatim from isotonic_predictor.ipynb (the production TempoPredictor,
FORM_MODE='residual') so it can be imported by the web app and the artifact
builder without running a notebook.

Pipeline:
    spread     = reg.predict([tempo_adj, home_court, form_diff])   # predicted margin
    win_prob   = calibrator(spread)                                # monotone margin -> P(win)

Data comes live from ESPN via sportsdataverse. Training + loading current ratings
is slow and network-bound, so build_artifacts.py trains once and pickles a ready
TempoPredictor for the website to load instantly.
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

GAME_WEIGHT_CONSTANT = 7  # tuned: 74.24% held-out vs 74.13% @ 10 (FORM_GAMES=5)


# ==============================================================================
# Efficiency helpers
# ==============================================================================
def calculate_efficiency(df):
    d1_teams = df['team_id'].unique()
    df = df[df['opponent_team_id'].isin(d1_teams)]
    t_stats = df.groupby(['season', 'team_id']).agg({
        'team_score': 'sum',
        'opponent_team_score': 'sum',
        'field_goals_attempted': 'sum',
        'free_throws_attempted': 'sum',
        'offensive_rebounds': 'sum',
        'turnovers': 'sum',
    }).reset_index()
    o_stats = df.groupby(['season', 'opponent_team_id']).agg({
        'field_goals_attempted': 'sum',
        'free_throws_attempted': 'sum',
        'offensive_rebounds': 'sum',
        'turnovers': 'sum'
    }).reset_index()
    o_stats.columns = ['season', 'team_id', 'opp_fga', 'opp_fta', 'opp_orb', 'opp_tov']
    stats = t_stats.merge(o_stats, on=['season', 'team_id'], how='left')
    stats['off_poss'] = (stats['field_goals_attempted'] + 0.475 * stats['free_throws_attempted'] -
                         stats['offensive_rebounds'] + stats['turnovers']).replace(0, 1)
    stats['def_poss'] = (stats['opp_fga'] + 0.475 * stats['opp_fta'] -
                         stats['opp_orb'] + stats['opp_tov']).replace(0, 1)
    stats['off_eff'] = ((stats['team_score'] / stats['off_poss']) * 100).clip(70, 140)
    stats['def_eff'] = ((stats['opponent_team_score'] / stats['def_poss']) * 100).clip(70, 140)
    return stats


def calculate_adjusted_efficiency(df, team_stats, iterations=10):
    ratings = team_stats[['season', 'team_id', 'off_eff', 'def_eff']].copy()
    for _ in range(iterations):
        rating_map = ratings.set_index(['season', 'team_id'])[['off_eff', 'def_eff']].to_dict('index')
        adjustments = []
        for _, game in df.iterrows():
            opp_key = (game['season'], game['opponent_team_id'])
            opp_rating = rating_map.get(opp_key, {'off_eff': 100, 'def_eff': 100})
            p = (game['field_goals_attempted'] + 0.475 * game['free_throws_attempted'] -
                 game['offensive_rebounds'] + game['turnovers'])
            if p <= 0:
                continue
            adj_off = (game['team_score'] / p * 100) + (100 - opp_rating['def_eff'])
            adj_def = (game['opponent_team_score'] / p * 100) + (100 - opp_rating['off_eff'])
            adjustments.append({'season': game['season'], 'team_id': game['team_id'],
                                'a_off': adj_off, 'a_def': adj_def})
        adj_df = pd.DataFrame(adjustments).groupby(['season', 'team_id']).mean().reset_index()
        ratings = ratings[['season', 'team_id']].merge(adj_df, on=['season', 'team_id'])
        ratings.columns = ['season', 'team_id', 'off_eff', 'def_eff']
    ratings['net_eff'] = ratings['off_eff'] - ratings['def_eff']
    return ratings


def blend_efficiency_ratings(current_eff, prior_eff, counts):
    blended = current_eff.merge(counts, on='team_id').merge(
        prior_eff, on='team_id', how='left', suffixes=('', '_p')
    )
    blended[['off_eff_p', 'def_eff_p']] = blended[['off_eff_p', 'def_eff_p']].fillna(100)
    w = blended['n'] / (blended['n'] + GAME_WEIGHT_CONSTANT)
    blended['adj_off'] = blended['off_eff'] * w + blended['off_eff_p'] * (1 - w)
    blended['adj_def'] = blended['def_eff'] * w + blended['def_eff_p'] * (1 - w)
    blended['net_eff'] = blended['adj_off'] - blended['adj_def']
    return blended[['team_id', 'adj_off', 'adj_def', 'net_eff']].rename(
        columns={'adj_off': 'off_eff', 'adj_def': 'def_eff'}
    )


# ==============================================================================
# BasePredictor — one spread model + one monotone win-prob calibrator, both fit
# on a walk-forward pass over the calibration year(s).
# ==============================================================================
class BasePredictor:
    def __init__(self):
        self.reg = None
        self.calibrator = None
        self.calibration_years = None
        self.current_efficiency = None
        self.team_lookup = None
        self.team_id_to_name = None
        self.current_season = None
        self.as_of_date = None

    def _fit_calibrator(self, pred_margins, actuals):
        raise NotImplementedError

    def _calibrate(self, pred_margin):
        raise NotImplementedError

    def _walk_forward_raw(self, season, prior_season):
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        data = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
        data['month'] = pd.to_datetime(data['game_date']).dt.to_period('M')
        months = sorted(data['month'].unique())
        historical = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
        prior_eff = calculate_adjusted_efficiency(historical, calculate_efficiency(historical))

        actuals, margins, net_diffs, locs = [], [], [], []
        for i, month in enumerate(months):
            month_games = data[data['month'] == month]
            prev_games = data[data['month'].isin(months[:i])]
            if len(prev_games) > 0:
                curr_eff = calculate_adjusted_efficiency(prev_games, calculate_efficiency(prev_games))
                eff = blend_efficiency_ratings(
                    curr_eff, prior_eff, prev_games.groupby('team_id').size().reset_index(name='n'))
            else:
                eff = prior_eff[['team_id', 'off_eff', 'def_eff', 'net_eff']].copy()
            eff_map = eff.set_index('team_id')
            for _, game in month_games.iterrows():
                t1, t2 = game['team_id'], game['opponent_team_id']
                if t1 not in eff_map.index or t2 not in eff_map.index:
                    continue
                net_diffs.append(eff_map.loc[t1]['net_eff'] - eff_map.loc[t2]['net_eff'])
                locs.append({'home': 1, 'neutral': 0, 'away': -1}.get(game['team_home_away'], 0))
                actuals.append(int(game['team_winner']))
                margins.append(game['team_score'] - game['opponent_team_score'])
        return (np.array(actuals), np.array(margins), np.array(net_diffs), np.array(locs))

    def train(self, calibration_year):
        start_time = time.time()
        name = type(self).__name__
        cal_years = [calibration_year] if isinstance(calibration_year, int) else list(calibration_year)
        self.calibration_years = cal_years
        print(f"Training {name} (walk-forward) on {cal_years}...")

        act, margin, net, loc = [], [], [], []
        for cy in cal_years:
            ac, mg, nd, lc = self._walk_forward_raw(cy, cy - 1)
            act.append(ac); margin.append(mg); net.append(nd); loc.append(lc)
        act = np.concatenate(act)
        margin = np.concatenate(margin)
        X = np.column_stack([np.concatenate(net), np.concatenate(loc)])

        self.reg = LinearRegression().fit(X, margin)
        print(f"   spread: margin = {self.reg.coef_[0]:.3f}*net_diff "
              f"+ {self.reg.coef_[1]:+.2f}*loc  (home court = {self.reg.coef_[1]:+.2f} pts)")
        self._fit_calibrator(self.reg.predict(X), act)
        print(f"OK {name} trained in {time.time() - start_time:.1f}s ({len(act)} games)")

    @staticmethod
    def _infer_current_season(ts=None):
        """NCAA seasons are labeled by the calendar year they END (spring): a date
        in Jul-Dec belongs to next year's season, Jan-Jun to the current year's."""
        ts = pd.Timestamp.now() if ts is None else pd.Timestamp(ts)
        return ts.year + 1 if ts.month >= 7 else ts.year

    @staticmethod
    def _match_team_name(team_name, lookup):
        """Resolve a team name to its id against {display_name: team_id}:
        exact -> case-insensitive -> best word-subset match. None if no hit."""
        if team_name in lookup:
            return lookup[team_name]
        lower = team_name.lower().strip()
        for name, tid in lookup.items():
            if name.lower() == lower:
                return tid
        search_words = set(lower.split())
        candidates = []
        for name, tid in lookup.items():
            name_words = set(name.lower().split())
            if search_words.issubset(name_words):
                candidates.append((len(name_words) - len(search_words), len(name), name, tid))
        if candidates:
            candidates.sort()
            return candidates[0][3]
        return None

    def _build_ratings(self, current_season, prior_season=None, as_of_date=None):
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if prior_season is None:
            prior_season = current_season - 1
        df_current = mbb_loaders.load_mbb_team_boxscore(seasons=[current_season]).to_pandas()
        if as_of_date is not None:
            df_current = df_current[pd.to_datetime(df_current['game_date']) <= pd.Timestamp(as_of_date)]
        df_prior = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
        prior_eff = calculate_adjusted_efficiency(df_prior, calculate_efficiency(df_prior))
        curr_eff = calculate_adjusted_efficiency(df_current, calculate_efficiency(df_current))
        counts = df_current.groupby('team_id').size().reset_index(name='n')
        blended = blend_efficiency_ratings(curr_eff, prior_eff, counts)
        self.current_efficiency = blended.set_index('team_id')
        self.team_lookup = (df_current[['team_id', 'team_display_name']].drop_duplicates()
                            .set_index('team_display_name')['team_id'].to_dict())
        self.team_id_to_name = {v: k for k, v in self.team_lookup.items()}
        self.current_season = current_season
        self.as_of_date = None if as_of_date is None else pd.Timestamp(as_of_date)
        stamp = f" as of {self.as_of_date.date()}" if self.as_of_date is not None else ""
        print(f"OK Loaded {current_season} ratings for {len(self.current_efficiency)} teams{stamp}")

    def load_current_ratings(self, prior_season=None):
        """Ratings as they stand right now: current season, every game played to date."""
        self._build_ratings(self._infer_current_season(), prior_season=prior_season)

    def get_team_id(self, team_name):
        return self._match_team_name(team_name, self.team_lookup)

    def validate_walk_forward(self, test_season, prior_season=None):
        if prior_season is None:
            prior_season = test_season - 1
        actuals, margins, net_diffs, locs = self._walk_forward_raw(test_season, prior_season)
        spreads = self.reg.predict(np.column_stack([net_diffs, locs]))
        probs = np.array([self._calibrate(s) for s in spreads])
        metrics = {
            'accuracy': np.mean((probs > 0.5) == actuals),
            'logloss': log_loss(actuals, probs),
            'auc': roc_auc_score(actuals, probs),
            'spread_mae': np.mean(np.abs(spreads - margins)),
            'spread_rmse': np.sqrt(np.mean((spreads - margins) ** 2)),
        }
        return metrics, probs, actuals


# ==============================================================================
# IsotonicPredictor — isotonic regression mapping predicted margin -> P(win)
# ==============================================================================
class IsotonicPredictor(BasePredictor):
    def _fit_calibrator(self, pred_margins, actuals):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(pred_margins, actuals)

    def _calibrate(self, pred_margin):
        return float(self.calibrator.predict([pred_margin])[0])


# ==============================================================================
# TempoPredictor — tempo-adjusted spread + recent form + team-specific home court
#
#   tempo_adj  = net_diff * pace_factor          (efficiency x pace interaction)
#   home_court = home_team_adv or -away_team_adv (replaces flat loc coef)
#   form_diff  = team1_form - team2_form         (recent form, last FORM_GAMES)
# ==============================================================================
class TempoPredictor(IsotonicPredictor):
    SPREAD_FEATURES = ['tempo_adj', 'home_court', 'form_diff']
    FORM_GAMES = 5            # rolling window for recent form
    FORM_MODE = 'residual'    # opponent-adjusted form; tuned default
    CONF_GATE_GAMES = 5       # conf games needed before form turns on (gate/window modes)
    HOME_ADV_SHRINK = 30      # Bayesian shrink constant for home court estimates
    HOME_ADV_MIN_GP = 20      # minimum games to appear in leaderboard

    _conf_cache = {}          # season -> set(conference game_ids)

    def __init__(self):
        super().__init__()
        self.current_pace = {}
        self.current_form = {}
        self.home_adv = {}
        self._league_avg_pace = 70.0
        self._league_home_adv = 3.0

    @classmethod
    def _conf_game_ids(cls, season):
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if season not in cls._conf_cache:
            sched = mbb_loaders.load_mbb_schedule(seasons=[season]).to_pandas()
            cls._conf_cache[season] = set(
                sched.loc[sched['conference_competition'] == True, 'game_id'])
        return cls._conf_cache[season]

    def _form_value(self, log):
        """Reduce one team's chronological game log to a single form number under
        the active FORM_MODE. Returns 0.0 when form is unavailable."""
        if not log:
            return 0.0
        mode = self.FORM_MODE
        if mode == 'raw':
            vals = [e['margin'] for e in log][-self.FORM_GAMES:]
        elif mode == 'residual':
            vals = [e['resid'] for e in log][-self.FORM_GAMES:]
        elif mode == 'conf_gate':
            if sum(e['conf'] for e in log) < self.CONF_GATE_GAMES:
                return 0.0
            vals = [e['margin'] for e in log][-self.FORM_GAMES:]
        elif mode == 'conf_window':
            conf = [e['margin'] for e in log if e['conf']]
            if len(conf) < self.CONF_GATE_GAMES:
                return 0.0
            vals = conf[-self.FORM_GAMES:]
        else:
            raise ValueError(f"unknown FORM_MODE: {mode!r}")
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _compute_team_pace(data):
        data = data.copy()
        data['poss'] = (data['field_goals_attempted']
                        + 0.475 * data['free_throws_attempted']
                        - data['offensive_rebounds']
                        + data['turnovers'])
        return data.groupby('team_id')['poss'].mean()

    @staticmethod
    def _league_pace(data):
        poss = (data['field_goals_attempted']
                + 0.475 * data['free_throws_attempted']
                - data['offensive_rebounds']
                + data['turnovers'])
        return float(poss.mean())

    def _compute_current_form(self, data, conf_ids):
        data = data.sort_values('game_date')
        league = self._league_avg_pace
        eff = self.current_efficiency
        logs = {}
        for _, g in data.iterrows():
            t1, t2 = g['team_id'], g['opponent_team_id']
            margin = float(g['team_score'] - g['opponent_team_score'])
            if t1 in eff.index and t2 in eff.index:
                net = eff.loc[t1]['net_eff'] - eff.loc[t2]['net_eff']
                pf = ((self.current_pace.get(t1, league)
                       + self.current_pace.get(t2, league)) / 2) / league
                loc = {'home': 1, 'neutral': 0, 'away': -1}.get(g['team_home_away'], 0)
                hc = self._home_court_feature(t1, t2, loc, self.home_adv, self._league_home_adv)
                exp = net * pf + hc
            else:
                exp = 0.0
            logs.setdefault(t1, []).append({'margin': margin,
                                            'conf': bool(g['game_id'] in conf_ids),
                                            'resid': margin - exp})
        return {tid: self._form_value(log) for tid, log in logs.items()}

    @classmethod
    def _compute_home_adv(cls, data, shrink=None):
        """Per-team home court advantage in pts, Bayes-shrunk toward league mean.
        Returns (adv_dict, league_avg). adv = (mean_home_margin - mean_away_margin)/2."""
        if shrink is None:
            shrink = cls.HOME_ADV_SHRINK
        df = data.copy()
        df['margin'] = df['team_score'] - df['opponent_team_score']
        home = (df[df['team_home_away'] == 'home']
                .groupby('team_id')['margin']
                .agg(home_mean='mean', n_home='count'))
        away = (df[df['team_home_away'] == 'away']
                .groupby('team_id')['margin']
                .agg(away_mean='mean', n_away='count'))
        merged = home.join(away, how='outer').fillna(0)
        merged['raw_adv'] = (merged['home_mean'] - merged['away_mean']) / 2
        league_avg = float(merged['raw_adv'].mean())
        n_total = merged['n_home'] + merged['n_away']
        w = n_total / (n_total + shrink)
        merged['home_adv'] = merged['raw_adv'] * w + league_avg * (1 - w)
        return merged['home_adv'].to_dict(), league_avg

    def _home_court_feature(self, t1, t2, loc, adv_dict, league_avg):
        """Points contribution of home court from team1's perspective."""
        if loc == 1:
            return adv_dict.get(t1, league_avg)
        elif loc == -1:
            return -adv_dict.get(t2, league_avg)
        return 0.0

    def _walk_forward_frame(self, season, prior_season):
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        data = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
        data['month'] = pd.to_datetime(data['game_date']).dt.to_period('M')
        data = data.sort_values('game_date')
        months = sorted(data['month'].unique())
        conf_ids = self._conf_game_ids(season)
        historical = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
        prior_eff = calculate_adjusted_efficiency(historical, calculate_efficiency(historical))
        prior_pace = self._compute_team_pace(historical)
        league_avg = self._league_pace(historical)
        prior_home_adv, league_home_adv = self._compute_home_adv(historical)

        form_log = {}

        rows = []
        for i, month in enumerate(months):
            month_games = data[data['month'] == month].sort_values('game_date')
            prev_games = data[data['month'].isin(months[:i])]

            if len(prev_games) > 0:
                curr_eff = calculate_adjusted_efficiency(prev_games, calculate_efficiency(prev_games))
                counts = prev_games.groupby('team_id').size()
                eff = blend_efficiency_ratings(curr_eff, prior_eff, counts.reset_index(name='n'))
                curr_pace = self._compute_team_pace(prev_games)

                def blended_pace(tid, _cp=curr_pace, _pp=prior_pace, _c=counts, _la=league_avg):
                    n = _c.get(tid, 0)
                    w = n / (n + GAME_WEIGHT_CONSTANT)
                    return float(_cp.get(tid, _la)) * w + float(_pp.get(tid, _la)) * (1 - w)
            else:
                eff = prior_eff[['team_id', 'off_eff', 'def_eff', 'net_eff']].copy()

                def blended_pace(tid, _pp=prior_pace, _la=league_avg):
                    return float(_pp.get(tid, _la))

            eff_map = eff.set_index('team_id')
            for _, g in month_games.iterrows():
                t1, t2 = g['team_id'], g['opponent_team_id']
                if t1 not in eff_map.index or t2 not in eff_map.index:
                    continue

                net_diff = eff_map.loc[t1]['net_eff'] - eff_map.loc[t2]['net_eff']
                exp_pace = (blended_pace(t1) + blended_pace(t2)) / 2
                pace_factor = exp_pace / league_avg
                loc = {'home': 1, 'neutral': 0, 'away': -1}.get(g['team_home_away'], 0)
                home_court = self._home_court_feature(t1, t2, loc, prior_home_adv, league_home_adv)

                t1_form = self._form_value(form_log.get(t1, []))
                t2_form = self._form_value(form_log.get(t2, []))

                margin = g['team_score'] - g['opponent_team_score']
                exp_margin = net_diff * pace_factor + home_court

                rows.append({
                    'game_date': g['game_date'],
                    'tempo_adj': net_diff * pace_factor,
                    'net_diff': net_diff,
                    'loc': loc,
                    'home_court': home_court,
                    'pace_factor': pace_factor,
                    'form_diff': t1_form - t2_form,
                    'margin': margin,
                    'won': int(g['team_winner']),
                })
                form_log.setdefault(t1, []).append({
                    'margin': float(margin),
                    'conf': bool(g['game_id'] in conf_ids),
                    'resid': float(margin - exp_margin),
                })

        return pd.DataFrame(rows)

    def train(self, calibration_year):
        start = time.time()
        cal_years = [calibration_year] if isinstance(calibration_year, int) else list(calibration_year)
        self.calibration_years = cal_years
        print(f"Training TempoPredictor (walk-forward, form={self.FORM_MODE}) on {cal_years}...")
        frame = pd.concat([self._walk_forward_frame(cy, cy - 1) for cy in cal_years],
                          ignore_index=True)
        self.reg = LinearRegression().fit(frame[self.SPREAD_FEATURES], frame['margin'])
        print("   spread: " + "  ".join(f"{k}={v:+.4f}"
              for k, v in zip(self.SPREAD_FEATURES, self.reg.coef_)))
        self._fit_calibrator(self.reg.predict(frame[self.SPREAD_FEATURES]), frame['won'].values)
        print(f"OK TempoPredictor trained in {time.time() - start:.1f}s ({len(frame)} games)")

    def validate_walk_forward(self, test_season, prior_season=None):
        if prior_season is None:
            prior_season = test_season - 1
        f = self._walk_forward_frame(test_season, prior_season)
        spreads = self.reg.predict(f[self.SPREAD_FEATURES])
        probs = np.array([self._calibrate(s) for s in spreads])
        actuals, margins = f['won'].values, f['margin'].values
        return {
            'accuracy': np.mean((probs > 0.5) == actuals),
            'logloss': log_loss(actuals, probs),
            'auc': roc_auc_score(actuals, probs),
            'spread_mae': np.mean(np.abs(spreads - margins)),
            'spread_rmse': np.sqrt(np.mean((spreads - margins) ** 2)),
        }, probs, actuals

    def _build_ratings(self, current_season, prior_season=None, as_of_date=None):
        import sportsdataverse.mbb.mbb_loaders as mbb_loaders
        if prior_season is None:
            prior_season = current_season - 1
        df_current = mbb_loaders.load_mbb_team_boxscore(seasons=[current_season]).to_pandas()
        if as_of_date is not None:
            df_current = df_current[pd.to_datetime(df_current['game_date']) <= pd.Timestamp(as_of_date)]
        df_prior = mbb_loaders.load_mbb_team_boxscore(seasons=[prior_season]).to_pandas()
        prior_eff = calculate_adjusted_efficiency(df_prior, calculate_efficiency(df_prior))
        curr_eff = calculate_adjusted_efficiency(df_current, calculate_efficiency(df_current))
        counts = df_current.groupby('team_id').size()
        blended = blend_efficiency_ratings(curr_eff, prior_eff, counts.reset_index(name='n'))
        self.current_efficiency = blended.set_index('team_id')
        self.team_lookup = (df_current[['team_id', 'team_display_name']].drop_duplicates()
                            .set_index('team_display_name')['team_id'].to_dict())
        self.team_id_to_name = {v: k for k, v in self.team_lookup.items()}
        self.current_season = current_season
        self.as_of_date = None if as_of_date is None else pd.Timestamp(as_of_date)

        self.current_pace = {}
        self.home_adv = {}

        prior_pace = self._compute_team_pace(df_prior)
        curr_pace = self._compute_team_pace(df_current)
        self._league_avg_pace = self._league_pace(df_prior)
        for tid in set(curr_pace.index) | set(prior_pace.index):
            n = counts.get(tid, 0)
            w = n / (n + GAME_WEIGHT_CONSTANT)
            self.current_pace[tid] = (float(curr_pace.get(tid, self._league_avg_pace)) * w
                                      + float(prior_pace.get(tid, self._league_avg_pace)) * (1 - w))

        prior_adv, prior_league = self._compute_home_adv(df_prior)
        curr_adv, curr_league = self._compute_home_adv(df_current)
        self._league_home_adv = curr_league
        for tid in set(prior_adv) | set(curr_adv):
            n = counts.get(tid, 0)
            w = n / (n + self.HOME_ADV_SHRINK)
            self.home_adv[tid] = (curr_adv.get(tid, prior_league) * w
                                  + prior_adv.get(tid, prior_league) * (1 - w))

        conf_ids = self._conf_game_ids(current_season)
        self.current_form = self._compute_current_form(df_current, conf_ids)

        stamp = f" as of {self.as_of_date.date()}" if self.as_of_date is not None else ""
        print(f"OK Loaded {current_season} ratings + pace + form (L{self.FORM_GAMES}, {self.FORM_MODE}) "
              f"+ home court for {len(self.current_efficiency)} teams{stamp}")

    def predict_game(self, team1_name, team2_name, team1_home=True, neutral_site=False,
                     verbose=True, save_output=False):
        t1_id = self.get_team_id(team1_name)
        t2_id = self.get_team_id(team2_name)
        if t1_id is None:
            raise ValueError(f"Team not found: '{team1_name}'")
        if t2_id is None:
            raise ValueError(f"Team not found: '{team2_name}'")
        t1 = self.team_id_to_name[t1_id]
        t2 = self.team_id_to_name[t2_id]
        eff1, eff2 = self.current_efficiency.loc[t1_id], self.current_efficiency.loc[t2_id]
        loc = 0 if neutral_site else (1 if team1_home else -1)
        net_diff = eff1['net_eff'] - eff2['net_eff']
        t1_pace = self.current_pace.get(t1_id, self._league_avg_pace)
        t2_pace = self.current_pace.get(t2_id, self._league_avg_pace)
        exp_pace = (t1_pace + t2_pace) / 2
        pace_factor = exp_pace / self._league_avg_pace
        tempo_adj = net_diff * pace_factor
        t1_form = self.current_form.get(t1_id, 0.0)
        t2_form = self.current_form.get(t2_id, 0.0)
        form_diff = t1_form - t2_form
        home_court = self._home_court_feature(t1_id, t2_id, loc, self.home_adv, self._league_home_adv)

        spread = self.reg.predict(np.array([[tempo_adj, home_court, form_diff]]))[0]
        prob_cal = self._calibrate(spread)

        result = {
            'team1': t1, 'team2': t2,
            'team1_home': team1_home, 'neutral_site': neutral_site,
            'expected_pace': round(exp_pace, 1), 'pace_factor': round(pace_factor, 3),
            'team1_form': round(t1_form, 1), 'team2_form': round(t2_form, 1),
            'home_court_pts': round(home_court, 2),
            'team1_win_prob': prob_cal, 'team2_win_prob': 1 - prob_cal,
            'team1_spread': spread, 'team2_spread': -spread,
            'team1_net_eff': eff1['net_eff'], 'team2_net_eff': eff2['net_eff'],
            'team1_off_eff': eff1['off_eff'], 'team1_def_eff': eff1['def_eff'],
            'team2_off_eff': eff2['off_eff'], 'team2_def_eff': eff2['def_eff'],
        }
        if verbose:
            venue = "Neutral" if neutral_site else (f"{t1} (Home)" if team1_home else f"{t2} (Home)")
            fav = t1 if prob_cal > 0.5 else t2
            fav_prob = max(prob_cal, 1 - prob_cal)
            fav_spread = spread if prob_cal > 0.5 else -spread
            print(f"\n{'='*60}")
            print(f"{t1} vs {t2}  |  {venue}  |  pace {exp_pace:.1f} ({pace_factor:+.2%} vs avg)")
            print(f"   Ratings  - {t1}: {eff1['net_eff']:+.1f}  {t2}: {eff2['net_eff']:+.1f}")
            print(f"   Form L{self.FORM_GAMES} ({self.FORM_MODE}) - {t1}: {t1_form:+.1f}  {t2}: {t2_form:+.1f}")
            print(f"   {t1}: {prob_cal*100:.1f}%   {t2}: {(1-prob_cal)*100:.1f}%")
            print(f"   Favorite: {fav} by {abs(fav_spread):.1f}  ({fav_prob*100:.1f}%)")
            print(f"{'='*60}\n")
        return result
