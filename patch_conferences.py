"""
patch_conferences.py — Backfill per-season conference labels for historical
seasons, which were built with an empty `conf.` column.

Source of truth: ESPN's per-season standings endpoint, which lists every
conference (id + name) and its member teams for that exact season — so it is
realignment-correct (e.g. Pac-12 teams in 2018 are Pac-12, not their 2026 home).

Labels are aligned to the style the current (2026) site already shows by voting
the 2026 `conf.` label onto each conference id; conferences absent in 2026
(defunct, e.g. Pac-12) fall back to the cleaned standings name.

Patches, for each historical season:
  - player_stats_{s}.csv / player_stats_conf_{s}.csv   (source CSVs)
  - site/data/{s}/player-stats.json / player-stats-conf.json
  - site/data/{s}/net-ratings.json

Run:  python patch_conferences.py
"""
import json
import os
import re
from collections import Counter

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CURRENT = 2026
SEASONS = range(2016, CURRENT)  # historical only; current already has conf
STANDINGS = ("https://site.api.espn.com/apis/v2/sports/basketball"
             "/mens-college-basketball/standings?season={season}&level=2")


def clean(name):
    if not name:
        return None
    return re.sub(r"\s+(Conference|Conf\.?)$", "", name).strip()


def standings(season):
    """Return (team_id -> conf_id, conf_id -> name) for a season."""
    j = requests.get(STANDINGS.format(season=season), timeout=30).json()
    tid2cid, cid2name = {}, {}
    for c in j.get("children", []):
        cid = str(c["id"])
        cid2name[cid] = c.get("name")
        for e in c.get("standings", {}).get("entries", []):
            try:
                tid2cid[int(e["team"]["id"])] = cid
            except (KeyError, TypeError, ValueError):
                pass
    return tid2cid, cid2name


def build_cid2label():
    """conf_id -> site-style label, voted from the 2026 conf. labels."""
    ps = (pd.read_csv(os.path.join(BASE, f"player_stats_{CURRENT}.csv"),
                      usecols=["team_id", "conf."]).dropna().drop_duplicates("team_id"))
    tid2label = dict(zip(ps["team_id"].astype(int), ps["conf."]))
    tid2cid, _ = standings(CURRENT)
    votes = {}
    for tid, cid in tid2cid.items():
        if tid in tid2label:
            votes.setdefault(cid, Counter())[tid2label[tid]] += 1
    return {cid: c.most_common(1)[0][0] for cid, c in votes.items()}


def label_map_for(season, cid2label):
    """team_id -> label for a season (site label where known, else clean name)."""
    tid2cid, cid2name = standings(season)
    out = {}
    for tid, cid in tid2cid.items():
        out[tid] = cid2label.get(cid) or clean(cid2name.get(cid))
    return out


def patch_csv(path, tid2label):
    if not os.path.exists(path):
        return f"missing {os.path.basename(path)}"
    df = pd.read_csv(path)
    df["conf."] = df["team_id"].map(tid2label)
    df.to_csv(path, index=False)
    return f"{df['conf.'].notna().sum()}/{len(df)} labeled"


def patch_json(path, id_key, tid2label):
    if not os.path.exists(path):
        return f"missing"
    obj = json.load(open(path))
    rows = obj.get("players") if "players" in obj else obj.get("net_ratings")
    n = 0
    for r in rows:
        tid = r.get(id_key)
        lbl = tid2label.get(int(tid)) if tid is not None else None
        r["conf"] = lbl
        if lbl:
            n += 1
    with open(path, "w") as f:
        f.write(json.dumps(obj, separators=(",", ":")))
    return f"{n}/{len(rows)} labeled"


def main():
    print("Building conf_id -> label map from 2026…")
    cid2label = build_cid2label()
    print(f"  {len(cid2label)} conferences mapped\n")

    for s in SEASONS:
        tid2label = label_map_for(s, cid2label)
        d = os.path.join(BASE, "site", "data", str(s))
        msgs = [
            f"csv={patch_csv(os.path.join(BASE, f'player_stats_{s}.csv'), tid2label)}",
            f"csv-conf={patch_csv(os.path.join(BASE, f'player_stats_conf_{s}.csv'), tid2label)}",
            f"players={patch_json(os.path.join(d, 'player-stats.json'), 'tid', tid2label)}",
            f"players-conf={patch_json(os.path.join(d, 'player-stats-conf.json'), 'tid', tid2label)}",
            f"net={patch_json(os.path.join(d, 'net-ratings.json'), 'team_id', tid2label)}",
        ]
        print(f"{s}: " + " | ".join(msgs))

    print("\nDone.")


if __name__ == "__main__":
    main()
