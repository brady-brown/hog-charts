"""Per-player career tables for the player card.

Joins every season's player-stats-career.json (the all-games stats with no games or
minutes minimum, written by build_site) on athlete id (stable across teams, so a
transfer's seasons line up) and writes them as site/data/careers/{n}.json,
one shard per id % CAREER_SHARDS. The card fetches the single ~85 KB shard that
holds its player rather than every season's 2-3 MB stats file.

Each row keeps the team for that season, which is what shows transfers.
"""
import json
import os
import re

from hoglib.io import sanitize_for_json

CAREER_SHARDS = 64   # MUST match CAREER_SHARDS in site/js/player-card.js

# player-stats keys carried into each career row (box + advanced views on the card).
CAREER_KEYS = ["t", "tid", "conf", "gp", "mpg", "ppg", "rpg", "apg", "spg", "bpg", "tpg",
               "fg", "fg3", "ft", "ts", "efg", "usg", "astp", "tovp", "trbp", "stlp", "blkp"]


def write_career_shards(data_dir):
    """Rebuild data_dir/careers/ from every data_dir/{season}/player-stats-career.json."""
    careers = {}
    seasons = sorted(int(d) for d in os.listdir(data_dir)
                     if re.fullmatch(r"\d{4}", d)
                     and os.path.exists(os.path.join(data_dir, d, "player-stats-career.json")))
    for season in seasons:
        with open(os.path.join(data_dir, str(season), "player-stats-career.json")) as stats_file:
            players = json.load(stats_file)["players"]
        for player in players:
            if player.get("id") is None:
                continue
            row = {"s": season, **{k: player.get(k) for k in CAREER_KEYS}}
            rows = careers.setdefault(int(player["id"]), [])
            if row not in rows:    # a few seasons list the same player row twice
                rows.append(row)

    # Only players with a D-I season ever get a card. Their seasons at non-D-I
    # schools (games against D-I opponents) stay in the table.
    careers = {athlete_id: rows for athlete_id, rows in careers.items()
               if any(row.get("conf") for row in rows)}

    shards = [{} for _ in range(CAREER_SHARDS)]
    for athlete_id, rows in careers.items():
        shards[athlete_id % CAREER_SHARDS][str(athlete_id)] = rows

    out_dir = os.path.join(data_dir, "careers")
    os.makedirs(out_dir, exist_ok=True)
    for shard_index, shard in enumerate(shards):
        with open(os.path.join(out_dir, f"{shard_index}.json"), "w") as shard_file:
            shard_file.write(json.dumps(sanitize_for_json(shard), separators=(",", ":"),
                                        allow_nan=False))
    print(f"  careers/{{0..{CAREER_SHARDS - 1}}}.json  {len(careers):,} players, seasons {seasons}")
