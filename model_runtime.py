"""
model_runtime.py — pure numpy/pandas inference for the trained TempoPredictor.

Loads the data-only artifacts (model.json + teams.parquet) and reproduces
predict_game without unpickling sklearn objects. This keeps the deployed app
free of scikit-learn / sportsdataverse and immune to numpy/pickle version skew
between the machine that builds the artifacts and the server that runs them.

Artifacts are produced by build_artifacts.py.
"""

import json
import os

import numpy as np
import pandas as pd


def _match_team_name(team_name, lookup):
    """exact -> case-insensitive -> best word-subset match. None if no hit.
    (Identical resolution logic to predictor.BasePredictor._match_team_name.)"""
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


class RuntimePredictor:
    def __init__(self, art_dir):
        with open(os.path.join(art_dir, "model.json")) as f:
            m = json.load(f)
        self.coef = np.array(m["coef"], dtype=float)
        self.intercept = float(m["intercept"])
        self.iso_x = np.array(m["iso_x"], dtype=float)
        self.iso_y = np.array(m["iso_y"], dtype=float)
        self.league_avg_pace = float(m["league_avg_pace"])
        self.league_home_adv = float(m["league_home_adv"])
        self.form_games = m.get("form_games")
        self.form_mode = m.get("form_mode")

        t = pd.read_parquet(os.path.join(art_dir, "teams.parquet"))
        self.teams = t
        self.by_id = t.set_index("team_id")
        self.lookup = dict(zip(t["team"], t["team_id"]))          # name -> id
        self.id_to_name = dict(zip(t["team_id"], t["team"]))
        self.lg_def = float(t["def_eff"].mean())

    # -- name resolution -------------------------------------------------------
    def get_team_id(self, name):
        return _match_team_name(name, self.lookup)

    def team_names(self):
        return sorted(self.lookup.keys())

    # -- inference -------------------------------------------------------------
    def _calibrate(self, spread):
        # np.interp clamps to the end values outside [iso_x[0], iso_x[-1]],
        # matching IsotonicRegression(out_of_bounds='clip').
        return float(np.interp(spread, self.iso_x, self.iso_y))

    def _home_court(self, t1_id, t2_id, loc):
        if loc == 1:
            return self.by_id.loc[t1_id].get("home_adv", self.league_home_adv)
        if loc == -1:
            return -self.by_id.loc[t2_id].get("home_adv", self.league_home_adv)
        return 0.0

    def predict_game(self, team1_name, team2_name, team1_home=True, neutral_site=False,
                     verbose=False, save_output=False):
        t1_id = self.get_team_id(team1_name)
        t2_id = self.get_team_id(team2_name)
        if t1_id is None:
            raise ValueError(f"Team not found: '{team1_name}'")
        if t2_id is None:
            raise ValueError(f"Team not found: '{team2_name}'")
        e1 = self.by_id.loc[t1_id]
        e2 = self.by_id.loc[t2_id]
        t1, t2 = self.id_to_name[t1_id], self.id_to_name[t2_id]

        loc = 0 if neutral_site else (1 if team1_home else -1)
        net_diff = e1["net_eff"] - e2["net_eff"]
        exp_pace = (e1["pace"] + e2["pace"]) / 2
        pace_factor = exp_pace / self.league_avg_pace
        tempo_adj = net_diff * pace_factor
        form_diff = e1["form"] - e2["form"]
        home_court = self._home_court(t1_id, t2_id, loc)

        spread = (self.intercept
                  + self.coef[0] * tempo_adj
                  + self.coef[1] * home_court
                  + self.coef[2] * form_diff)
        prob_cal = self._calibrate(spread)

        return {
            "team1": t1, "team2": t2,
            "team1_home": team1_home, "neutral_site": neutral_site,
            "expected_pace": round(float(exp_pace), 1), "pace_factor": round(float(pace_factor), 3),
            "team1_form": round(float(e1["form"]), 1), "team2_form": round(float(e2["form"]), 1),
            "home_court_pts": round(float(home_court), 2),
            "team1_win_prob": prob_cal, "team2_win_prob": 1 - prob_cal,
            "team1_spread": float(spread), "team2_spread": float(-spread),
            "team1_net_eff": float(e1["net_eff"]), "team2_net_eff": float(e2["net_eff"]),
            "team1_off_eff": float(e1["off_eff"]), "team1_def_eff": float(e1["def_eff"]),
            "team2_off_eff": float(e2["off_eff"]), "team2_def_eff": float(e2["def_eff"]),
        }
