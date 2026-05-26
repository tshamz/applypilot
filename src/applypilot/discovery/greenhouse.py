"""Greenhouse ATS direct API scraper: searches employer career boards.

Hits Greenhouse's public Job Board API (no auth) at
  https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true

Unlike Workday's CXS endpoint, Greenhouse returns full HTML descriptions in
the listing response, so there's no separate detail fetch. The trade-off is
that the public API has no server-side search — we fetch the whole board for
each employer (typically 50-500 jobs) and filter client-side against the
configured query strings.

Employer registry is loaded from config/employers.yaml under the
`greenhouse:` key (kept separate from Workday's `employers:` key so the two
scrapers can co-exist in one config file).
"""

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import yaml

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import get_connection, init_db
from applypilot.discovery.workday import (
    _load_location_filter,
    _location_ok,
    strip_html,
)

log = logging.getLogger(__name__)

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"


# -- Employer registry from YAML --------------------------------------------

def load_employers() -> dict:
    """Load Greenhouse employer registry from config/employers.yaml.

    Returns a dict keyed by employer ID with `slug` and `name` fields. The
    `slug` is the Greenhouse board token (e.g. "stripe", "airbnb").
    """
    path = CONFIG_DIR / "employers.yaml"
    if not path.exists():
        log.warning("employers.yaml not found at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("greenhouse", {})


# -- HTTP -------------------------------------------------------------------

def fetch_board(slug: str, timeout: int = 30) -> list[dict]:
    """Fetch all jobs for a Greenhouse board slug.

    Raises urllib.error.HTTPError on non-200 responses (notably 404 for an
    invalid slug). Returns the `jobs` array from the response.
    """
    url = GREENHOUSE_API.format(slug=slug)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (ApplyPilot)",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("jobs", [])


# -- Filtering --------------------------------------------------------------

def _matches_query(title: str, queries: list[str]) -> bool:
    """Case-insensitive substring match of `title` against any configured query."""
    if not title:
        return False
    t = title.lower()
    return any(q.lower() in t for q in queries)


def _job_location(job: dict) -> str:
    """Extract a human-readable location string from a Greenhouse job dict.

    Prefers `offices[].name` if present (often more specific, e.g. "Remote -
    US" vs "Remote") then falls back to `location.name`.
    """
    offices = job.get("offices") or []
    if offices and isinstance(offices, list):
        names = [o.get("name") for o in offices if isinstance(o, dict) and o.get("name")]
        if names:
            return ", ".join(names)

    loc = job.get("location") or {}
    if isinstance(loc, dict):
        return loc.get("name") or ""
    return str(loc) if loc else ""


# -- DB storage -------------------------------------------------------------

def store_results(conn: sqlite3.Connection, jobs: list[dict], employer_name: str) -> tuple[int, int]:
    """Store Greenhouse jobs in DB. Returns (new, existing)."""
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("absolute_url")
        if not url:
            continue

        content = job.get("content") or ""
        full_description = strip_html(content) if content else None
        short_desc = full_description[:500] if full_description else None
        detail_scraped_at = now if full_description else None

        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at, detail_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), None, short_desc, _job_location(job),
                 employer_name, "greenhouse_api", now, full_description, url, detail_scraped_at, None),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


# -- Per-employer worker ----------------------------------------------------

def _process_one(
    employer_key: str,
    employers: dict,
    queries: list[str],
    location_filter: bool,
    accept_locs: list[str],
    reject_locs: list[str],
) -> dict:
    """Fetch one employer's board, filter, store."""
    emp = employers[employer_key]
    slug = emp.get("slug", employer_key)
    name = emp.get("name", employer_key)

    try:
        all_jobs = fetch_board(slug)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.warning("%s: board not found (slug='%s')", name, slug)
        else:
            log.error("%s: HTTP %s fetching board", name, e.code)
        return {"employer": name, "found": 0, "new": 0, "existing": 0, "error": f"HTTP {e.code}"}
    except Exception as e:
        log.error("%s: ERROR fetching board: %s", name, e)
        return {"employer": name, "found": 0, "new": 0, "existing": 0, "error": str(e)}

    # Client-side query filter — Greenhouse public API has no server-side search.
    matched = [j for j in all_jobs if _matches_query(j.get("title", ""), queries)]

    if location_filter and matched:
        matched = [j for j in matched if _location_ok(_job_location(j), accept_locs, reject_locs)]

    if not matched:
        log.info("%s: 0 matches (of %d total jobs)", name, len(all_jobs))
        return {"employer": name, "found": 0, "new": 0, "existing": 0}

    conn = get_connection()
    new, existing = store_results(conn, matched, name)
    log.info("%s: %d matched (of %d), %d new, %d already in DB",
             name, len(matched), len(all_jobs), new, existing)

    return {"employer": name, "found": len(matched), "new": new, "existing": existing}


# -- Main orchestrator ------------------------------------------------------

def scrape_employers(
    queries: list[str],
    employers: dict,
    location_filter: bool = True,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
    workers: int = 1,
) -> dict:
    """Run full Greenhouse crawl across all configured employers.

    Sequential by default. When workers > 1, processes employers in parallel
    using ThreadPoolExecutor.
    """
    init_db()

    if accept_locs is None:
        accept_locs = []
    if reject_locs is None:
        reject_locs = []

    total_new = 0
    total_existing = 0
    total_found = 0
    errors = 0
    t0 = time.time()

    valid_keys = list(employers.keys())

    if workers > 1 and len(valid_keys) > 1:
        completed = 0
        with ThreadPoolExecutor(max_workers=min(workers, len(valid_keys))) as pool:
            futures = {
                pool.submit(
                    _process_one, key, employers, queries,
                    location_filter, accept_locs, reject_locs,
                ): key
                for key in valid_keys
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                total_new += result["new"]
                total_existing += result["existing"]
                total_found += result["found"]
                if "error" in result:
                    errors += 1

                if completed % 10 == 0 or completed == len(valid_keys):
                    elapsed = time.time() - t0
                    log.info("Progress: %d/%d boards (%d new, %d dupes, %d errors) [%.0fs]",
                             completed, len(valid_keys), total_new, total_existing, errors, elapsed)
    else:
        completed = 0
        for key in valid_keys:
            result = _process_one(
                key, employers, queries,
                location_filter, accept_locs, reject_locs,
            )
            completed += 1
            total_new += result["new"]
            total_existing += result["existing"]
            total_found += result["found"]
            if "error" in result:
                errors += 1

            if completed % 10 == 0 or completed == len(valid_keys):
                elapsed = time.time() - t0
                log.info("Progress: %d/%d boards (%d new, %d dupes, %d errors) [%.0fs]",
                         completed, len(valid_keys), total_new, total_existing, errors, elapsed)

    elapsed = time.time() - t0
    log.info("Greenhouse: %d matched, %d new, %d dupes, %d errors in %.0fs",
             total_found, total_new, total_existing, errors, elapsed)

    return {
        "found": total_found,
        "new": total_new,
        "existing": total_existing,
        "errors": errors,
    }


# -- Public entry point -----------------------------------------------------

def run_greenhouse_discovery(employers: dict | None = None, workers: int = 1) -> dict:
    """Main entry point for Greenhouse-based job discovery.

    Loads the Greenhouse employer registry from config/employers.yaml
    (`greenhouse:` section) and the user's search config, then fetches each
    board, filters client-side by query + location, and stores matches.

    Args:
        employers: Override the employer registry. If None, loads from YAML.
        workers: Number of parallel threads for board fetches. Default 1.

    Returns:
        Dict with stats: found, new, existing, queries.
    """
    if employers is None:
        employers = load_employers()

    if not employers:
        log.warning(
            "No Greenhouse employers configured. Add a `greenhouse:` section "
            "to config/employers.yaml.",
        )
        return {"found": 0, "new": 0, "existing": 0, "queries": 0}

    search_cfg = config.load_search_config()
    queries_cfg = search_cfg.get("queries", [])
    accept_locs, reject_locs = _load_location_filter(search_cfg)

    max_tier = search_cfg.get("greenhouse_max_tier", 2)
    queries = [q["query"] for q in queries_cfg if q.get("tier", 99) <= max_tier]

    if not queries:
        queries = [q["query"] for q in queries_cfg]

    if not queries:
        log.warning("No search queries configured in searches.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "queries": 0}

    location_filter = search_cfg.get("greenhouse_location_filter", True)

    log.info("Greenhouse crawl: %d queries x %d boards (workers=%d)",
             len(queries), len(employers), workers)

    result = scrape_employers(
        queries=queries,
        employers=employers,
        location_filter=location_filter,
        accept_locs=accept_locs,
        reject_locs=reject_locs,
        workers=workers,
    )

    return {
        "found": result["found"],
        "new": result["new"],
        "existing": result["existing"],
        "queries": len(queries),
        "errors": result.get("errors", 0),
    }
