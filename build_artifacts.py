"""
build_artifacts.py — train the predictor once and dump everything the website
needs to load instantly.

Outputs (in artifacts/):
    predictor.pkl      pickled, trained TempoPredictor (model + current ratings)
    net_ratings.parquet  one row per D-I team: off/def/net eff, pace, home court,
                         form, record, ranks
    metadata.json      build timestamp, season, calibration years, backtest metrics

Re-run this whenever you want to refresh (e.g. nightly cron) — the Streamlit app
reads only these files and never touches the network.
"""

import json
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

import sportsdataverse.mbb.mbb_loaders as mbb_loaders
from predictor import TempoPredictor

import os as _os
_override = _os.environ.get("OVERRIDE_SEASON")
if _override:
    _season = int(_override)
else:
    _today = _date.today()
    _season = _today.year + 1 if _today.month >= 11 else _today.year

RATINGS_ONLY = bool(_os.environ.get("RATINGS_ONLY"))  # skip model training for historical builds

ART = Path(__file__).parent / "artifacts" / str(_season)
ART.mkdir(parents=True, exist_ok=True)
CALIBRATION_YEARS = list(range(2021, _season))  # all complete seasons before current
BACKTEST_YEAR = _season - 1                       # most recently completed season
MIN_GAMES = 5                 # teams below this are dropped from the ratings table


def build_net_ratings(pred):
    """Assemble the public net-ratings table from a loaded predictor + records."""
    season = pred.current_season
    box = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
    rec = (box.groupby('team_id')
              .agg(wins=('team_winner', 'sum'), games=('team_winner', 'count'))
              .reset_index())
    rec['losses'] = rec['games'] - rec['wins']

    # Strength of schedule: average adjusted net rating of the opponents a team
    # faced. Higher = tougher slate. (The ratings themselves are already opponent
    # adjusted; this surfaces the schedule difficulty explicitly.)
    net_by_id = pred.current_efficiency['net_eff']
    opp = box[box['opponent_team_id'].isin(net_by_id.index)].copy()
    opp['opp_net'] = opp['opponent_team_id'].map(net_by_id)
    sos = opp.groupby('team_id')['opp_net'].mean().rename('sos').reset_index()

    eff = pred.current_efficiency.reset_index()  # team_id, off_eff, def_eff, net_eff
    eff['team'] = eff['team_id'].map(pred.team_id_to_name)
    eff['pace'] = eff['team_id'].map(lambda t: pred.current_pace.get(t, np.nan))
    # Scale raw home-court estimate by the spread coefficient so the displayed
    # value is the actual points it contributes to a prediction (raw is ~3x larger).
    # In ratings-only mode the model isn't trained so fall back to a neutral scale.
    hc_coef = float(pred.reg.coef_[1]) if pred.reg is not None else 1.0
    eff['home_court'] = eff['team_id'].map(lambda t: pred.home_adv.get(t, np.nan)) * hc_coef
    eff['form'] = eff['team_id'].map(lambda t: pred.current_form.get(t, 0.0))

    df = eff.merge(rec[['team_id', 'wins', 'losses', 'games']], on='team_id', how='left')
    df = df.merge(sos, on='team_id', how='left')
    df = df[df['games'].fillna(0) >= MIN_GAMES].copy()
    df = df.dropna(subset=['team'])

    df = df.sort_values('net_eff', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    df['off_rank'] = df['off_eff'].rank(ascending=False, method='min').astype(int)
    df['def_rank'] = df['def_eff'].rank(ascending=True, method='min').astype(int)  # lower is better
    df['sos_rank'] = df['sos'].rank(ascending=False, method='min').astype(int)     # tougher = better rank

    cols = ['rank', 'team', 'wins', 'losses', 'net_eff', 'off_eff', 'def_eff',
            'off_rank', 'def_rank', 'sos', 'sos_rank', 'pace', 'home_court', 'form',
            'games', 'team_id']
    return df[cols].round({'net_eff': 1, 'off_eff': 1, 'def_eff': 1,
                           'sos': 1, 'pace': 1, 'home_court': 1, 'form': 1})


def build_teams_table(pred):
    """Per-team inputs the runtime predictor needs (raw, unscaled)."""
    eff = pred.current_efficiency.reset_index()
    eff["team"] = eff["team_id"].map(pred.team_id_to_name)
    eff["pace"] = eff["team_id"].map(lambda t: pred.current_pace.get(t, pred._league_avg_pace))
    eff["home_adv"] = eff["team_id"].map(lambda t: pred.home_adv.get(t, pred._league_home_adv))
    eff["form"] = eff["team_id"].map(lambda t: pred.current_form.get(t, 0.0))
    eff = eff.dropna(subset=["team"])
    return eff[["team_id", "team", "off_eff", "def_eff", "net_eff", "pace", "home_adv", "form"]]


def build_model_json(pred, metrics):
    """Everything the runtime needs to reproduce predict_game without sklearn."""
    return {
        "built_at": pd.Timestamp.now().isoformat(),
        "season": int(pred.current_season),
        "calibration_years": CALIBRATION_YEARS,
        "form_mode": pred.FORM_MODE,
        "form_games": pred.FORM_GAMES,
        "league_avg_pace": float(pred._league_avg_pace),
        "league_home_adv": float(pred._league_home_adv),
        "coef": [float(c) for c in pred.reg.coef_],            # [tempo_adj, home_court, form_diff]
        "intercept": float(pred.reg.intercept_),
        "iso_x": [float(x) for x in pred.calibrator.X_thresholds_],
        "iso_y": [float(y) for y in pred.calibrator.y_thresholds_],
        "backtest_year": BACKTEST_YEAR,
        "backtest": metrics,
    }


def main():
    ART.mkdir(exist_ok=True)
    mode = "ratings-only" if RATINGS_ONLY else "full"
    print("=" * 60)
    print(f"Building predictor artifacts  [{mode}]")
    print("=" * 60)

    pred = TempoPredictor()

    if not RATINGS_ONLY:
        pred.train(calibration_year=CALIBRATION_YEARS)

    pred._build_ratings(_season)

    metrics = {}
    if not RATINGS_ONLY and BACKTEST_YEAR is not None:
        print(f"\nBacktesting on {BACKTEST_YEAR} (held out)...")
        m, _, _ = pred.validate_walk_forward(BACKTEST_YEAR)
        metrics = {k: float(v) for k, v in m.items()}
        print(f"   accuracy={m['accuracy']:.4f}  logloss={m['logloss']:.4f}  "
              f"auc={m['auc']:.4f}  spread_mae={m['spread_mae']:.2f}")

    teams = build_teams_table(pred)
    teams.to_parquet(ART / "teams.parquet", index=False)
    print(f"\nWrote {ART / 'teams.parquet'}  ({len(teams)} teams)")

    if not RATINGS_ONLY:
        with open(ART / "model.json", "w") as f:
            json.dump(build_model_json(pred, metrics), f, indent=2)
        print(f"Wrote {ART / 'model.json'}")

    ratings = build_net_ratings(pred)
    ratings.to_parquet(ART / "net_ratings.parquet", index=False)
    print(f"Wrote {ART / 'net_ratings.parquet'}  ({len(ratings)} teams)")

    meta = {
        "built_at": pd.Timestamp.now().isoformat(),
        "season": int(pred.current_season),
        "calibration_years": CALIBRATION_YEARS,
        "n_teams": int(len(ratings)),
        "form_mode": pred.FORM_MODE,
        "form_games": pred.FORM_GAMES,
        "league_avg_pace": round(pred._league_avg_pace, 2),
        "league_home_adv": round(pred._league_home_adv, 2),
        "backtest_year": BACKTEST_YEAR,
        "backtest": metrics,
    }
    with open(ART / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {ART / 'metadata.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
