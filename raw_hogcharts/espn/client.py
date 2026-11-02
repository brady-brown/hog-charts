"""HTTP access to ESPN's public college-basketball endpoints.

Everything in raw_hogcharts talks to ESPN through this module. The endpoints are
undocumented and unversioned, so all the defensive behaviour lives in one place:
retries with backoff on 429/5xx, a browser-ish User-Agent, bounded concurrency,
and a hard timeout per request.

Two endpoints supply the whole pipeline:

  scoreboard?dates=YYYYMMDD&groups=50&limit=500
      Every D-I game on one Eastern-time date, with team ids, conference ids,
      scores, status, tournamentId and the notes headline. One call per date.

  summary?event={game_id}
      One game: plays (including substitutions and shot coordinates), the player
      box, the team box, and the header (season, season_type, date, competitors).
      One call per game.
"""
import random
import threading
import time

import requests
from requests.adapters import HTTPAdapter

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# D-I men's basketball group. Without it the scoreboard also returns non-D-I.
DIVISION_I_GROUP = "50"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 30
MAX_ATTEMPTS = 5

_thread_local = threading.local()


class ESPNError(RuntimeError):
    """A request failed after exhausting retries."""


def _session():
    """One pooled Session per thread (Session is not thread-safe to share)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


def get_json(url, params=None, timeout=DEFAULT_TIMEOUT, max_attempts=MAX_ATTEMPTS):
    """GET a JSON payload, retrying on throttling, 5xx and transport errors.

    Backoff is exponential with jitter (1s, 2s, 4s, 8s ±25%). A 404 is returned
    as None rather than raised: ESPN 404s for game ids that exist on the
    scoreboard but have no summary yet, which is normal, not an error.
    """
    last_error = None
    for attempt in range(max_attempts):
        if attempt:
            delay = (2 ** (attempt - 1)) * random.uniform(0.75, 1.25)
            time.sleep(delay)
        try:
            response = _session().get(url, params=params, timeout=timeout)
        except requests.RequestException as error:
            last_error = error
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            last_error = ESPNError(f"HTTP {response.status_code} from {response.url}")
            continue
        if not response.ok:
            raise ESPNError(f"HTTP {response.status_code} from {response.url}")
        try:
            return response.json()
        except ValueError as error:
            last_error = error   # truncated/HTML body — worth another attempt
            continue
    raise ESPNError(f"giving up on {url} params={params}: {last_error}")


def fetch_scoreboard(date_yyyymmdd, groups=DIVISION_I_GROUP):
    """All events on one Eastern-time date. Returns the raw event dicts."""
    payload = get_json(
        f"{SITE_API}/scoreboard",
        params={"dates": date_yyyymmdd, "groups": groups, "limit": "500"},
    )
    return (payload or {}).get("events", []) or []


def fetch_summary(game_id):
    """The full summary payload for one game, or None if ESPN has nothing."""
    return get_json(f"{SITE_API}/summary", params={"event": int(game_id)})


def map_concurrent(function, items, workers=8, on_error="raise", progress=None):
    """Run `function` over `items` with a bounded thread pool, preserving order.

    ESPN tolerates modest concurrency; 8 is well-behaved and gets a full season
    in minutes. on_error="collect" yields (item, None) for failures instead of
    raising, so one dead game can't abort a season-long ingest.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if not items:
        return []
    results = [None] * len(items)
    errors = []

    def run(index_item):
        index, item = index_item
        try:
            return index, function(item), None
        except Exception as error:            # noqa: BLE001 — reported, not swallowed
            if on_error == "raise":
                raise
            return index, None, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, value, error in pool.map(run, enumerate(items)):
            results[index] = value
            if error is not None:
                errors.append((items[index], error))
            if progress is not None:
                progress.update(1)

    if errors and on_error == "collect":
        print(f"  [warn] {len(errors)} of {len(items)} requests failed; first: {errors[0][1]}")
    return results
