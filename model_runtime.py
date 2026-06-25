"""
model_runtime.py — Pure numpy/pandas game prediction without sklearn.

WHY THIS FILE EXISTS
--------------------
The Hog Charts predictor is trained in Python with sklearn's LinearRegression
and IsotonicRegression.  A naive deployment would pickle those objects and
load them on the cloud server — but this breaks whenever the numpy or sklearn
version on the server differs from the build machine.

Instead, build_artifacts.py serializes the model to a compact model.json
containing just the coefficient arrays and the isotonic calibration lookup
table.  This module reads those plain-number arrays and reproduces the
prediction in ~20 lines using only numpy.  No sklearn, no pickle, no version
skew.

The prediction formula implemented here:
  1. tempo_adjusted_diff = (team1_net_eff − team2_net_eff) × (avg_pace / league_avg_pace)
  2. home_court_pts      = team's raw home_adv × spread model's home-court coefficient
  3. predicted_spread    = intercept
                           + coef[0] × tempo_adjusted_diff
                           + coef[1] × home_court_pts
                           + coef[2] × form_difference
  4. win_probability     = np.interp(predicted_spread, iso_x, iso_y)
                           (the isotonic calibration curve built during training)

Artifacts are produced by build_artifacts.py.
"""

import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Team name matching
# ---------------------------------------------------------------------------

def _match_team_name(input_team_name, team_name_to_id_lookup):
    """Resolve a user-typed team name to a numeric team ID.

    Resolution order (same logic as predictor.BasePredictor._match_team_name):
      1. Exact match
      2. Case-insensitive exact match
      3. Best word-subset match (user's words ⊆ official name words)

    Returns the team_id integer, or None if no match found.
    """
    if input_team_name in team_name_to_id_lookup:
        return team_name_to_id_lookup[input_team_name]

    normalized_input = input_team_name.lower().strip()
    for official_name, team_id in team_name_to_id_lookup.items():
        if official_name.lower() == normalized_input:
            return team_id

    input_words = set(normalized_input.split())
    word_subset_candidates = []
    for official_name, team_id in team_name_to_id_lookup.items():
        official_words = set(official_name.lower().split())
        if input_words.issubset(official_words):
            extra_words = len(official_words) - len(input_words)
            word_subset_candidates.append((extra_words, len(official_name), official_name, team_id))

    if word_subset_candidates:
        word_subset_candidates.sort()
        return word_subset_candidates[0][3]

    return None


# ---------------------------------------------------------------------------
# RuntimePredictor
# ---------------------------------------------------------------------------

class RuntimePredictor:
    """sklearn-free predictor that loads model.json + teams.parquet.

    Attributes:
        regression_coef         ndarray  [tempo_adj_coef, home_court_coef, form_diff_coef]
        regression_intercept    float    Linear regression intercept (near 0).
        isotonic_x_thresholds   ndarray  Predicted spread breakpoints for the calibration curve.
        isotonic_y_win_probs    ndarray  Win-probability values at each breakpoint.
        league_avg_pace         float    League-average possessions per game (for tempo scaling).
        league_home_advantage   float    League-average home-court advantage (fallback).
        form_games              int/None Number of recent games used to compute form.
        form_mode               str/None "raw" or "residual" form calculation mode.
        all_teams_df            DataFrame  One row per team; all ratings columns.
        team_ratings_by_id      DataFrame  Same, indexed by team_id for fast lookup.
        team_name_to_id         dict     {team_display_name: team_id}.
        team_id_to_name         dict     {team_id: team_display_name}.
        league_avg_defensive_eff float   Used to normalize home-court sign conventions.
    """

    def __init__(self, artifacts_directory):
        """Load model.json and teams.parquet from the artifacts directory.

        Args:
            artifacts_directory: Path to the artifacts/{season}/ folder.
        """
        model_json_path = os.path.join(artifacts_directory, "model.json")
        with open(model_json_path) as model_file:
            model_data = json.load(model_file)

        self.regression_coef         = np.array(model_data["coef"],  dtype=float)
        self.regression_intercept    = float(model_data["intercept"])
        self.isotonic_x_thresholds   = np.array(model_data["iso_x"], dtype=float)
        self.isotonic_y_win_probs    = np.array(model_data["iso_y"], dtype=float)
        self.league_avg_pace         = float(model_data["league_avg_pace"])
        self.league_home_advantage   = float(model_data["league_home_adv"])
        self.form_games              = model_data.get("form_games")
        self.form_mode               = model_data.get("form_mode")

        teams_parquet_path = os.path.join(artifacts_directory, "teams.parquet")
        self.all_teams_df         = pd.read_parquet(teams_parquet_path)
        self.team_ratings_by_id   = self.all_teams_df.set_index("team_id")
        self.team_name_to_id      = dict(zip(self.all_teams_df["team"], self.all_teams_df["team_id"]))
        self.team_id_to_name      = dict(zip(self.all_teams_df["team_id"], self.all_teams_df["team"]))
        self.league_avg_defensive_eff = float(self.all_teams_df["def_eff"].mean())

    # ── Name resolution ──────────────────────────────────────────────────────

    def get_team_id(self, team_name_string):
        """Resolve a team name to a numeric ID using fuzzy matching."""
        return _match_team_name(team_name_string, self.team_name_to_id)

    def team_names(self):
        """Return sorted list of all team names (for dropdowns / autocomplete)."""
        return sorted(self.team_name_to_id.keys())

    # ── Prediction ───────────────────────────────────────────────────────────

    def _calibrate_spread_to_win_prob(self, predicted_spread):
        """Map a predicted point spread to a win probability using the isotonic curve.

        Uses np.interp which clamps values outside the calibration range to the
        boundary win probabilities — matching sklearn's out_of_bounds='clip'.
        """
        return float(np.interp(predicted_spread, self.isotonic_x_thresholds, self.isotonic_y_win_probs))

    def _home_court_points(self, team1_id, team2_id, location_code):
        """Return the home-court contribution in predicted points.

        Args:
            location_code:  1 = team1 is at home, -1 = team2 is at home, 0 = neutral.
        """
        if location_code == 1:
            return self.team_ratings_by_id.loc[team1_id].get("home_adv", self.league_home_advantage)
        if location_code == -1:
            return -self.team_ratings_by_id.loc[team2_id].get("home_adv", self.league_home_advantage)
        return 0.0

    def predict_game(self, team1_name, team2_name,
                     team1_is_home=True, is_neutral_site=False,
                     verbose=False, save_output=False):
        """Predict the spread and win probability for a matchup.

        Args:
            team1_name:       Display name of the first team.
            team2_name:       Display name of the second team.
            team1_is_home:    True if team1 is the home team (ignored if neutral).
            is_neutral_site:  True if the game is at a neutral site.

        Returns:
            dict with spread, win probabilities, and all intermediate values.
        """
        team1_id = self.get_team_id(team1_name)
        team2_id = self.get_team_id(team2_name)
        if team1_id is None:
            raise ValueError(f"Team not found: '{team1_name}'")
        if team2_id is None:
            raise ValueError(f"Team not found: '{team2_name}'")

        team1_ratings = self.team_ratings_by_id.loc[team1_id]
        team2_ratings = self.team_ratings_by_id.loc[team2_id]
        team1_display = self.team_id_to_name[team1_id]
        team2_display = self.team_id_to_name[team2_id]

        # location_code: +1 = team1 home, -1 = team2 home, 0 = neutral
        location_code = 0 if is_neutral_site else (1 if team1_is_home else -1)

        net_efficiency_diff  = team1_ratings["net_eff"] - team2_ratings["net_eff"]
        expected_pace        = (team1_ratings["pace"] + team2_ratings["pace"]) / 2
        pace_factor          = expected_pace / self.league_avg_pace
        tempo_adjusted_diff  = net_efficiency_diff * pace_factor

        # form_diff: positive → team1 is in better recent form than team2
        form_difference = team1_ratings["form"] - team2_ratings["form"]
        home_court_pts  = self._home_court_points(team1_id, team2_id, location_code)

        predicted_spread = (
            self.regression_intercept
            + self.regression_coef[0] * tempo_adjusted_diff
            + self.regression_coef[1] * home_court_pts
            + self.regression_coef[2] * form_difference
        )
        team1_win_probability = self._calibrate_spread_to_win_prob(predicted_spread)

        return {
            "team1":               team1_display,
            "team2":               team2_display,
            "team1_home":          team1_is_home,
            "neutral_site":        is_neutral_site,
            "expected_pace":       round(float(expected_pace), 1),
            "pace_factor":         round(float(pace_factor), 3),
            "team1_form":          round(float(team1_ratings["form"]), 1),
            "team2_form":          round(float(team2_ratings["form"]), 1),
            "home_court_pts":      round(float(home_court_pts), 2),
            "team1_win_prob":      team1_win_probability,
            "team2_win_prob":      1 - team1_win_probability,
            "team1_spread":        float(predicted_spread),
            "team2_spread":        float(-predicted_spread),
            "team1_net_eff":       float(team1_ratings["net_eff"]),
            "team2_net_eff":       float(team2_ratings["net_eff"]),
            "team1_off_eff":       float(team1_ratings["off_eff"]),
            "team1_def_eff":       float(team1_ratings["def_eff"]),
            "team2_off_eff":       float(team2_ratings["off_eff"]),
            "team2_def_eff":       float(team2_ratings["def_eff"]),
        }


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# --- _match_team_name() ---
# input_team_name              str     User-supplied team name to resolve.
# team_name_to_id_lookup       dict    {official_name: team_id} from teams.parquet.
# normalized_input             str     Lowercase, stripped input for case-insensitive check.
# input_words                  set     Words in the user's input for subset matching.
# official_words               set     Words in one official name for subset matching.
# extra_words                  int     Number of words in official name beyond input words (sort key).
# word_subset_candidates       list    [(extra_words, name_len, name, team_id), …] sorted ascending.
#
# --- RuntimePredictor.__init__() ---
# artifacts_directory          str     Path to artifacts/{season}/ (contains model.json + teams.parquet).
# model_data                   dict    Parsed model.json.
# regression_coef              ndarray [tempo_adj_coef, home_court_coef, form_diff_coef].
# regression_intercept         float   Linear regression intercept.
# isotonic_x_thresholds        ndarray Sorted predicted spread values from training.
# isotonic_y_win_probs         ndarray Win probability at each threshold (isotonic curve).
# league_avg_pace              float   League mean possessions per game; scales tempo diff.
# league_home_advantage        float   League mean home-court raw advantage (fallback).
# all_teams_df                 DataFrame  off_eff, def_eff, net_eff, pace, home_adv, form per team.
# team_ratings_by_id           DataFrame  Same, indexed by team_id for O(1) lookups.
# team_name_to_id              dict    {team_display_name: team_id}.
# team_id_to_name              dict    {team_id: team_display_name}.
# league_avg_defensive_eff     float   Mean defensive efficiency (not used in prediction; diagnostic).
#
# --- predict_game() ---
# team1_id / team2_id          int     Numeric ESPN team IDs for the two teams.
# team1_ratings / team2_ratings Row    Slices from team_ratings_by_id; contain eff + form + pace.
# location_code                int     +1 = team1 home, -1 = team2 home, 0 = neutral.
# net_efficiency_diff          float   team1.net_eff − team2.net_eff (positive → team1 better).
# expected_pace                float   Average of both teams' pace estimates.
# pace_factor                  float   expected_pace / league_avg_pace (1.0 = average pace game).
# tempo_adjusted_diff          float   net_efficiency_diff × pace_factor (the core feature).
# form_difference              float   team1.form − team2.form (positive → team1 in better form).
# home_court_pts               float   Home-court adjustment in predicted points.
# predicted_spread             float   Predicted point margin from team1's perspective.
# team1_win_probability        float   Calibrated win probability for team1 (0–1).
