"""
Stockyard Baseball scraper (https://app.stockyardbaseball.com).

Stockyard Baseball provides expected pitch movement profiles by arm angle/slot,
pitch comparisons, and grading benchmarks. We pull this data to:
  1. Supplement movement baselines (expected IVB/HB by pitch type + slot).
  2. Cross-validate our Stuff+ grades against their proprietary grades.
  3. Extract any publicly exposed pitch-level tables for training signal.

NOTE: Stockyard is a React SPA. Public endpoints are reverse-engineered from
      its network traffic. Update BASE_API if the endpoints change.
"""

import logging
import time
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://app.stockyardbaseball.com"
BASE_API = "https://app.stockyardbaseball.com"   # update if a separate API host is found

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": BASE_URL,
}

# Known public endpoints (update based on network inspection)
ENDPOINTS = {
    "pitch_types":       "/api/pitch-types",
    "pitcher_profile":   "/api/pitcher/{pitcher_id}",
    "movement_comps":    "/api/movement-comps",
    "arm_angle_baselines": "/api/arm-angle-baselines",
    "leaderboard":       "/api/leaderboard",
}


class StockyardScraper:
    """Thin wrapper around Stockyard Baseball's public/inferred API."""

    def __init__(self, sleep_between: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.sleep = sleep_between

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        url = BASE_API + path
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            time.sleep(self.sleep)
            return resp.json()
        except requests.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} for {url}")
            return None
        except Exception as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Public methods – each returns a DataFrame or None
    # ------------------------------------------------------------------

    def get_arm_angle_baselines(self) -> Optional[pd.DataFrame]:
        """
        Fetch expected movement (IVB, HB) by pitch type and arm angle slot.
        These supplement our own population-based baselines.
        """
        data = self._get(ENDPOINTS["arm_angle_baselines"])
        if data is None:
            logger.info("arm_angle_baselines endpoint not available; falling back to population baselines")
            return None
        try:
            df = pd.DataFrame(data)
            logger.info(f"Got {len(df)} arm-angle baseline rows from Stockyard")
            return df
        except Exception as e:
            logger.warning(f"Could not parse arm_angle_baselines: {e}")
            return None

    def get_leaderboard(self, season: int = 2025) -> Optional[pd.DataFrame]:
        """Fetch pitcher leaderboard (if public)."""
        data = self._get(ENDPOINTS["leaderboard"], params={"season": season})
        if data is None:
            return None
        try:
            rows = data.get("data", data)
            df = pd.DataFrame(rows)
            logger.info(f"Got {len(df)} leaderboard rows from Stockyard ({season})")
            return df
        except Exception as e:
            logger.warning(f"Could not parse leaderboard: {e}")
            return None

    def get_pitcher_profile(self, pitcher_id: int) -> Optional[pd.DataFrame]:
        path = ENDPOINTS["pitcher_profile"].format(pitcher_id=pitcher_id)
        data = self._get(path)
        if data is None:
            return None
        try:
            pitches = data.get("pitches", data)
            df = pd.DataFrame(pitches)
            return df
        except Exception as e:
            logger.warning(f"Could not parse pitcher {pitcher_id}: {e}")
            return None

    def scrape_movement_comps_html(self) -> Optional[pd.DataFrame]:
        """
        Fallback: scrape any publicly rendered HTML tables for movement
        comparison benchmarks (used if JSON API isn't accessible).
        """
        try:
            resp = self.session.get(BASE_URL, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            tables = pd.read_html(str(soup))
            if tables:
                logger.info(f"Scraped {len(tables)} HTML tables from Stockyard home")
                return pd.concat(tables, ignore_index=True)
        except Exception as e:
            logger.warning(f"HTML scrape failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience function used by main pipeline
# ---------------------------------------------------------------------------

def fetch_stockyard_baselines() -> Optional[pd.DataFrame]:
    """
    Try all known methods to pull movement baselines from Stockyard.
    Returns a DataFrame with columns [pitch_type, arm_slot, ivb_mean, hb_mean, ...]
    or None if the site is inaccessible.
    """
    scraper = StockyardScraper()

    df = scraper.get_arm_angle_baselines()
    if df is not None:
        return df

    # If JSON endpoint isn't available, try HTML fallback
    df = scraper.scrape_movement_comps_html()
    if df is not None:
        return df

    logger.info(
        "Stockyard Baseball data not available. "
        "The model will use Statcast population baselines only."
    )
    return None
