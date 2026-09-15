"""Thin client for the public Fantasy Premier League API.

No API key is required. Endpoints used:
  - /bootstrap-static/            all players, teams, gameweeks, scoring rules
  - /fixtures/                    every fixture this season (finished + upcoming)
  - /element-summary/{id}/        one player's match-by-match history + upcoming fixtures
  - /entry/{id}/                  a manager's public profile
  - /entry/{id}/event/{gw}/picks/ a manager's squad for a given gameweek

Responses are cached to disk (data/cache/) so repeated runs don't hammer the
API; pass refresh=True to force a re-fetch.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
DEFAULT_TTL_SECONDS = 6 * 60 * 60  # bootstrap/fixtures/players: refresh every 6h
SHORT_TTL_SECONDS = 30 * 60  # a manager's own team: refresh more eagerly


class FplApiError(RuntimeError):
    """Raised when the FPL API returns an unexpected response."""


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _get(url: str) -> Any:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "fpl-optimizer/0.1"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FplApiError(f"request to {url} failed: {exc}") from exc
    return resp.json()


def _cached_get(name: str, url: str, refresh: bool, ttl: int) -> Any:
    path = _cache_path(name)
    if not refresh and path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        return json.loads(path.read_text())
    data = _get(url)
    path.write_text(json.dumps(data))
    return data


def get_bootstrap(refresh: bool = False) -> dict:
    """All players, teams, gameweeks and the live scoring rules."""
    return _cached_get("bootstrap", f"{BASE_URL}/bootstrap-static/", refresh, DEFAULT_TTL_SECONDS)


def get_fixtures(refresh: bool = False) -> list[dict]:
    """Every fixture for the season, finished and upcoming."""
    return _cached_get("fixtures", f"{BASE_URL}/fixtures/", refresh, DEFAULT_TTL_SECONDS)


def get_element_summary(player_id: int, refresh: bool = False) -> dict:
    """One player's match-by-match history plus their upcoming fixtures."""
    return _cached_get(
        f"element_{player_id}",
        f"{BASE_URL}/element-summary/{player_id}/",
        refresh,
        DEFAULT_TTL_SECONDS,
    )


def get_all_element_summaries(
    player_ids: list[int],
    refresh: bool = False,
    max_workers: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[int, dict]:
    """Fetch element-summary for many players concurrently.

    A failed individual fetch doesn't abort the batch; it's recorded as
    {"error": ...} so callers can decide how to degrade (e.g. fall back to
    bootstrap-only features for that player).
    """
    results: dict[int, dict] = {}
    total = len(player_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(get_element_summary, pid, refresh): pid for pid in player_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                results[pid] = fut.result()
            except FplApiError as exc:
                results[pid] = {"error": str(exc)}
            if on_progress:
                on_progress(i, total)
    return results


def get_entry(entry_id: int, refresh: bool = True) -> dict:
    """A manager's public profile (team name, overall rank, etc.)."""
    return _cached_get(f"entry_{entry_id}", f"{BASE_URL}/entry/{entry_id}/", refresh, SHORT_TTL_SECONDS)


def get_entry_picks(entry_id: int, event: int, refresh: bool = True) -> dict:
    """A manager's 15-player squad and bank balance for a given gameweek."""
    return _cached_get(
        f"entry_{entry_id}_picks_{event}",
        f"{BASE_URL}/entry/{entry_id}/event/{event}/picks/",
        refresh,
        SHORT_TTL_SECONDS,
    )
