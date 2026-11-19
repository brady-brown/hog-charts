"""
line_study.py — the model against historical betting lines, 2021–2026.

For every past game: what the model would have said that morning, next to the
book's closing (and, from 2024, opening) line and the final.

How each prediction is made (no game sees its own result):
  - ratings  : TempoPredictor walk-forward with refit='day' — efficiencies,
               pace and form rebuilt from every game before that date, blended
               with the prior season, the way the live site rates teams;
  - model    : spread regression, win-prob calibration and total correction
               fit on the OTHER five seasons' (monthly) walk-forward rows, so a
               season is never graded by coefficients that saw it. (Later seasons
               do inform earlier ones' coefficients — three slopes, stable across
               years — which is noted, not hidden.)
  - lines    : raw_hogcharts/odds_history.py (ESPN core odds API). One book per
               game: DraftKings, else ESPN BET (2025 lists only it), else consensus.

Writes raw_hogcharts/data/line_study/games.parquet (one row per game, for digging)
and prints line_report's tables for all seasons, a season-by-season summary,
and DraftKings-only. Daily walk-forward frames are cached per season.

Run:
    python -m raw_hogcharts.odds_history 2021 2022 2023 2024 2025 2026   # once
    python line_study.py            # ~25 min the first time, seconds after
    python line_study.py --refresh  # rebuild the cached frames
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import line_report
from predictor import TempoPredictor
from raw_hogcharts import ingest

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
SEASONS = list(range(2021, 2027))
OUT_DIR = os.path.join(ROOT, "raw_hogcharts", "data", "line_study")
BOOK_ORDER = ["DraftKings", "ESPN BET", "consensus"]


def cached_frame(predictor, season, refit, refresh, tag=""):
    path = os.path.join(OUT_DIR, f"walk_{refit}{tag}_{season}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    print(f"  walk-forward {season} (refit={refit})…", flush=True)
    frame = predictor._walk_forward_frame(season, season - 1, refit=refit)
    frame.to_parquet(path, index=False)
    return frame


def fit_models(train):
    """Exactly what TempoPredictor.train fits: spread LR, isotonic win prob, total LR."""
    feats = TempoPredictor.SPREAD_FEATURES
    spread = LinearRegression().fit(train[feats], train["margin"])
    calib = IsotonicRegression(out_of_bounds="clip").fit(spread.predict(train[feats]), train["won"])
    tot = train[~train["prior_only"]].dropna(subset=["proj_total", "total"])
    total = LinearRegression().fit(tot[["proj_total"]], tot["total"])
    return spread, calib, total


def pick_book(odds):
    rank = {name: i for i, name in enumerate(BOOK_ORDER)}
    odds = odds[odds["provider"].isin(BOOK_ORDER)].copy()
    odds["rank"] = odds["provider"].map(rank)
    return odds.sort_values(["game_id", "rank"]).drop_duplicates("game_id").drop(columns="rank")


def season_conferences(season):
    path = os.path.join(ROOT, "site", "data", str(season), "team-stats-all.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return {t["team_id"]: t.get("conf") for t in json.load(handle)["team_stats"]}


def build(refresh=False, three_point=None):
    """three_point=(k_off, k_def) rates teams on 3P%-regressed scores (experimental)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    predictor = TempoPredictor()
    predictor.THREE_PT_REGRESSION = three_point
    tag = f"_3pt{three_point[0]}-{three_point[1]}" if three_point else ""
    monthly = {s: cached_frame(predictor, s, "month", refresh, tag) for s in SEASONS}
    daily = {s: cached_frame(predictor, s, "day", refresh, tag) for s in SEASONS}

    games = []
    for season in SEASONS:
        odds_path = os.path.join(ingest.season_dir(season), "odds_history.parquet")
        if not os.path.exists(odds_path):
            print(f"  [{season}] no odds_history.parquet — run raw_hogcharts.odds_history first; skipping")
            continue
        spread_m, calib, total_m = fit_models(pd.concat([monthly[s] for s in SEASONS if s != season]))

        sched = pd.read_parquet(ingest.feed_path(season, "schedule"),
                                columns=["game_id", "home_id", "away_id", "neutral_site"])
        sched = sched.dropna(subset=["home_id", "away_id"]).astype({"game_id": "int64", "home_id": "int64",
                                                                    "away_id": "int64"})
        day = daily[season].astype({"game_id": "int64", "team_id": "int64"})
        # One row per game, from ESPN's designated home team's side (the line's side).
        day = day.merge(sched, on="game_id").query("team_id == home_id").drop_duplicates("game_id")
        feats = day[TempoPredictor.SPREAD_FEATURES]
        home_margin = spread_m.predict(feats)
        book = pick_book(pd.read_parquet(odds_path))
        df = pd.DataFrame({
            "season": season, "g": day["game_id"].values, "d": day["game_date"].astype(str).str[:10].values,
            "home": day["home_id"].values, "away": day["away_id"].values,
            "model_spread": -home_margin,                         # betting convention
            "model_prob": calib.predict(home_margin),
            "model_total": day["proj_total"].values,
            "model_total_cal": total_m.predict(day[["proj_total"]]),
            "home_pts": ((day["total"] + day["margin"]) / 2).values,
            "away_pts": ((day["total"] - day["margin"]) / 2).values,
        }).merge(book.rename(columns={"game_id": "g"}), on="g", how="inner")
        conf = season_conferences(season)
        high = df["away"].map(lambda t: conf.get(t) in line_report.HIGH_MAJOR).astype(int) \
             + df["home"].map(lambda t: conf.get(t) in line_report.HIGH_MAJOR).astype(int)
        df["tier"] = high.map({2: "high-major vs high-major", 1: "high-major vs other", 0: "neither high-major"})
        games.append(df)
        print(f"  [{season}] {len(df):,} games with a line  "
              f"({df['provider'].value_counts().to_dict()})", flush=True)

    out = pd.concat(games, ignore_index=True)
    out["capture_spread"] = np.nan          # line_report's at-capture column has no history
    out.to_parquet(os.path.join(OUT_DIR, f"games{tag}.parquet"), index=False)
    return out


def season_table(df):
    line_report.section("By season — model vs. closing line")
    print("  season  book(s)            games   MAE model  MAE close   ATS vs close (N≥0)   ATS vs close (N≥3)"
          "   moved toward model")
    for season, g in df.groupby("season"):
        margin = g["home_pts"] - g["away_pts"]
        w0, l0, _ = line_report.ats(g, "close_spread", 0)
        w3, l3, _ = line_report.ats(g, "close_spread", 3)
        mv = g[g["open_spread"].notna() & (g["open_spread"] != g["close_spread"])
               & (g["open_spread"] != g["model_spread"])]
        toward = (np.sign(mv["close_spread"] - mv["open_spread"])
                  == np.sign(mv["model_spread"] - mv["open_spread"])).mean() if len(mv) else np.nan
        books = "/".join(g["provider"].value_counts().index[:2])
        print(f"  {season}    {books:<18} {len(g):>6,}   {(-g['model_spread'] - margin).abs().mean():8.2f}"
              f"   {(-g['close_spread'] - margin).abs().mean():8.2f}   {w0:>5}-{l0:<5} {line_report.pct(w0, l0)}"
              f"   {w3:>4}-{l3:<4} {line_report.pct(w3, l3)}"
              f"   {'—' if np.isnan(toward) else f'{100 * toward:.1f}% of {len(mv):,}'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="rebuild cached walk-forward frames")
    parser.add_argument("--three-point", nargs=2, type=int, metavar=("K_OFF", "K_DEF"),
                        help="experimental: rate teams on 3P%%-regressed scores (e.g. 600 5000)")
    args = parser.parse_args()
    df = build(refresh=args.refresh, three_point=tuple(args.three_point) if args.three_point else None)
    df = df.dropna(subset=["model_spread", "close_spread", "home_pts", "away_pts"])
    print(f"\nHog Charts model vs. historical lines — {df['season'].min()}–{df['season'].max()}, "
          f"{len(df):,} games")
    season_table(df)
    line_report.report(df, " · all seasons")
    for tier, group in df.groupby("tier"):
        line_report.report(group, f" · {tier}")
    dk = df[df["provider"] == "DraftKings"]
    if len(dk):
        line_report.report(dk, " · DraftKings only")
    line_report.totals(df, "model_total", "raw projection")
    line_report.totals(df, "model_total_cal", "corrected")


if __name__ == "__main__":
    main()
