"""
line_report.py — how the model's pre-game numbers compare with the book's line.

Reads site/data/{season}/games/lines.json (written nightly by hoglib/lines.py:
the model's spread/total frozen before tip, the book's open/close, the final)
and prints:

  accuracy      mean absolute error vs. the final margin: model, opening line, closing line
  agreement     how closely the model tracks the line
  against the spread   when the model disagrees with the line by at least N points,
                how often the model's side covered (52.4% breaks even at -110)
  line movement when the model disagreed with the opener, how often the line
                moved toward the model by the close
  by tier       the same split by conference tier (the low-major markets are thinner)
  totals        the same for over/unders

Run:
    python line_report.py
    OVERRIDE_SEASON=2027 python line_report.py
"""
import json
import os

import numpy as np
import pandas as pd

from hoglib.season import detect_season

SEASON = detect_season()
ROOT = os.path.dirname(os.path.abspath(__file__))
HIGH_MAJOR = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}
BREAKEVEN = 110 / 210          # -110 both ways
THRESHOLDS = (0, 1, 2, 3, 4, 5, 7)


def load():
    path = os.path.join(ROOT, "site", "data", str(SEASON), "games", "lines.json")
    if not os.path.exists(path):
        raise SystemExit(f"No log yet at {os.path.relpath(path, ROOT)} — the nightly build writes it.")
    with open(path) as handle:
        data = json.load(handle)
    df = pd.DataFrame(data["games"], columns=data["cols"])
    for col in ("model_total_cal", "capture_spread", "capture_total"):
        if col not in df:
            df[col] = np.nan
    df = df.astype({c: float for c in ("model_total_cal", "capture_spread", "capture_total")})

    stats_path = os.path.join(ROOT, "site", "data", str(SEASON), "team-stats-all.json")
    conf = {}
    if os.path.exists(stats_path):
        with open(stats_path) as handle:
            conf = {t["team_id"]: t.get("conf") for t in json.load(handle)["team_stats"]}
    high = df["away"].map(lambda t: conf.get(t) in HIGH_MAJOR).astype(int) \
         + df["home"].map(lambda t: conf.get(t) in HIGH_MAJOR).astype(int)
    df["tier"] = high.map({2: "high-major vs high-major", 1: "high-major vs other", 0: "neither high-major"})
    return df


def ats(df, line_col, threshold):
    """Bet the model's side whenever it disagrees with the line by >= threshold."""
    margin = df["home_pts"] - df["away_pts"]
    edge = df[line_col] - df["model_spread"]          # > 0: model likes home more than the line
    picks = df[edge.abs() >= max(threshold, 0.5)]     # 0 = any disagreement at all
    if picks.empty:
        return 0, 0, 0
    home_cover = (margin[picks.index] + picks[line_col])
    model_home = edge[picks.index] > 0
    won = ((home_cover > 0) & model_home) | ((home_cover < 0) & ~model_home)
    push = home_cover == 0
    return int(won.sum()), int((~won & ~push).sum()), int(push.sum())


def pct(w, l):
    return f"{100 * w / (w + l):5.1f}%" if w + l else "    —"


def section(title):
    print(f"\n{title}\n{'─' * len(title)}")


def report(df, label=""):
    margin = df["home_pts"] - df["away_pts"]
    model_err = (-df["model_spread"] - margin).abs()
    close_err = (-df["close_spread"] - margin).abs()
    has_open = df["open_spread"].notna()
    open_err = (-df.loc[has_open, "open_spread"] - margin[has_open]).abs()

    section(f"Accuracy{label} — {len(df):,} games")
    print(f"  MAE vs final margin   model {model_err.mean():5.2f}   opening line {open_err.mean():5.2f}"
          f"   closing line {close_err.mean():5.2f}")
    picked = df[df["close_spread"] != 0]
    model_su = ((picked["model_spread"] < 0) == (margin[picked.index] > 0)).mean()
    line_su = ((picked["close_spread"] < 0) == (margin[picked.index] > 0)).mean()
    print(f"  Picked the winner     model {100 * model_su:5.1f}%   closing favorite {100 * line_su:5.1f}%")
    print(f"  Model vs close        r = {np.corrcoef(df['model_spread'], df['close_spread'])[0, 1]:.3f}"
          f"   mean gap {(df['model_spread'] - df['close_spread']).abs().mean():.2f} pts")

    has_capture = df["capture_spread"].notna()
    section(f"Against the spread{label} — betting the model's side when it disagrees by ≥ N")
    print("    N    vs close W-L-P      win%    vs open W-L-P       win%    at capture W-L-P    win%")
    for t in THRESHOLDS:
        cells = []
        for subset, col in ((df, "close_spread"), (df[has_open], "open_spread"), (df[has_capture], "capture_spread")):
            w, l, p = ats(subset, col, t)
            cells.append(f"{f'{w}-{l}-{p}':<14}  {pct(w, l)}")
        print(f"  {t:>3}    " + "    ".join(cells))
    print(f"  (break-even at -110: {100 * BREAKEVEN:.1f}%. \"at capture\" = the line posted when the model's"
          f" numbers were frozen, the one you could have bet; {int(has_capture.sum()):,} games)")

    moved = df[has_open & (df["open_spread"] != df["model_spread"])]
    if len(moved):
        toward = np.sign(moved["close_spread"] - moved["open_spread"]) \
               == np.sign(moved["model_spread"] - moved["open_spread"])
        changed = moved["close_spread"] != moved["open_spread"]
        section(f"Line movement{label}")
        print(f"  Of {int(changed.sum()):,} games where the line moved, it moved toward the model "
              f"{100 * toward[changed].mean():.1f}% of the time")


def totals(df, col="model_total", label="raw projection"):
    """col: model_total (the raw formula, every row) or model_total_cal (the fitted
    correction the site shows, logged from 2026-11-19 on)."""
    t = df.dropna(subset=[col, "close_total"])
    if t.empty:
        return
    actual = t["home_pts"] + t["away_pts"]
    section(f"Totals, {label} — {len(t):,} games")
    print(f"  MAE vs final total    model {(t[col] - actual).abs().mean():5.2f}"
          f"   closing line {(t['close_total'] - actual).abs().mean():5.2f}")
    print(f"  Bias (model − final)  model {(t[col] - actual).mean():+5.2f}"
          f"   closing line {(t['close_total'] - actual).mean():+5.2f}")
    print("    N    W-L-P            win%   (model's side of the over/under)")
    for n in THRESHOLDS:
        edge = t[col] - t["close_total"]
        picks = t[edge.abs() >= max(n, 0.5)]
        over = edge[picks.index] > 0
        diff = actual[picks.index] - picks["close_total"]
        won = ((diff > 0) & over) | ((diff < 0) & ~over)
        push = diff == 0
        w, l, p = int(won.sum()), int((~won & ~push).sum()), int(push.sum())
        print(f"  {n:>3}    {f'{w}-{l}-{p}':<14}   {pct(w, l)}")


def main():
    df = load()
    logged = len(df)
    graded = df.dropna(subset=["model_spread", "close_spread", "home_pts", "away_pts"]).copy()
    print(f"Hog Charts model vs. the line — {SEASON} season")
    print(f"{logged:,} games logged · {len(graded):,} graded (model + closing line + final)")
    if graded.empty:
        print("\nNothing graded yet: games need a pre-game model capture, a final, and a closing line.")
        return
    report(graded)
    for tier, group in graded.groupby("tier"):
        if len(group) >= 20:
            report(group, f" · {tier}")
    totals(graded)
    totals(graded, "model_total_cal", "corrected (what the site shows)")


if __name__ == "__main__":
    main()
