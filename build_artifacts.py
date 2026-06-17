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
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sportsdataverse.mbb.mbb_loaders as mbb_loaders
from predictor import TempoPredictor

ART = Path(__file__).parent / "artifacts"
CALIBRATION_YEARS = [2021, 2022, 2023, 2024, 2025]
BACKTEST_YEAR = 2026          # set to None to skip the (slower) held-out backtest
MIN_GAMES = 5                 # teams below this are dropped from the ratings table


def build_net_ratings(pred):
    """Assemble the public net-ratings table from a loaded predictor + records."""
    season = pred.current_season
    box = mbb_loaders.load_mbb_team_boxscore(seasons=[season]).to_pandas()
    rec = (box.groupby('team_id')
              .agg(wins=('team_winner', 'sum'), games=('team_winner', 'count'))
              .reset_index())
    rec['losses'] = rec['games'] - rec['wins']

    eff = pred.current_efficiency.reset_index()  # team_id, off_eff, def_eff, net_eff
    eff['team'] = eff['team_id'].map(pred.team_id_to_name)
    eff['pace'] = eff['team_id'].map(lambda t: pred.current_pace.get(t, np.nan))
    # Scale raw home-court estimate by the spread coefficient so the displayed
    # value is the actual points it contributes to a prediction (raw is ~3x larger).
    hc_coef = float(pred.reg.coef_[1])
    eff['home_court'] = eff['team_id'].map(lambda t: pred.home_adv.get(t, np.nan)) * hc_coef
    eff['form'] = eff['team_id'].map(lambda t: pred.current_form.get(t, 0.0))

    df = eff.merge(rec[['team_id', 'wins', 'losses', 'games']], on='team_id', how='left')
    df = df[df['games'].fillna(0) >= MIN_GAMES].copy()
    df = df.dropna(subset=['team'])

    df = df.sort_values('net_eff', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    df['off_rank'] = df['off_eff'].rank(ascending=False, method='min').astype(int)
    df['def_rank'] = df['def_eff'].rank(ascending=True, method='min').astype(int)  # lower is better

    cols = ['rank', 'team', 'wins', 'losses', 'net_eff', 'off_eff', 'def_eff',
            'off_rank', 'def_rank', 'pace', 'home_court', 'form', 'games', 'team_id']
    return df[cols].round({'net_eff': 1, 'off_eff': 1, 'def_eff': 1,
                           'pace': 1, 'home_court': 1, 'form': 1})


def main():
    ART.mkdir(exist_ok=True)
    print("=" * 60)
    print("Building predictor artifacts")
    print("=" * 60)

    pred = TempoPredictor()
    pred.train(calibration_year=CALIBRATION_YEARS)
    pred.load_current_ratings()

    metrics = {}
    if BACKTEST_YEAR is not None:
        print(f"\nBacktesting on {BACKTEST_YEAR} (held out)...")
        m, _, _ = pred.validate_walk_forward(BACKTEST_YEAR)
        metrics = {k: float(v) for k, v in m.items()}
        print(f"   accuracy={m['accuracy']:.4f}  logloss={m['logloss']:.4f}  "
              f"auc={m['auc']:.4f}  spread_mae={m['spread_mae']:.2f}")

    with open(ART / "predictor.pkl", "wb") as f:
        pickle.dump(pred, f)
    print(f"\nWrote {ART / 'predictor.pkl'}")

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
