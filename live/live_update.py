"""
Live 2026 season updater.

Every N hours (default: 6):
  1. Pull the latest Statcast data since the last stored game date.
  2. Engineer features using pre-computed 2024-25 baselines.
  3. Generate Stuff+ predictions.
  4. Append to pitches_2026 SQLite table.
  5. Regenerate player cards for every pitcher with new pitches.
  6. Also updates spring training (pitches_spring2026) while ST is active.

Usage:
    python main.py live [--interval 6]
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import schedule

from config import DB_PATH, MODEL_DIR, PROFILES_DIR, DATA_DIR
from scraper.statcast_scraper import download_date_range, download_spring, append_to_db, load_from_db, table_exists, save_to_db
from model.predict import StuffPlusPredictor, load_baselines
from profiles.player_cards import generate_all_cards

logger = logging.getLogger(__name__)

LIVE_TABLE   = "pitches_2026"
SPRING_TABLE = "pitches_spring2026"
SPRING_END   = "2026-03-25"   # Spring training ends the day before Opening Day


def _last_stored_date() -> str:
    """Return the most recent game_date in pitches_2026, or season start."""
    season_start = "2026-03-26"
    if not table_exists(LIVE_TABLE):
        return season_start
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(
            f"SELECT MAX(game_date) FROM [{LIVE_TABLE}]"
        ).fetchone()
        last = result[0] if result and result[0] else None
    finally:
        conn.close()
    if last:
        d = datetime.strptime(last[:10], "%Y-%m-%d") + timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    return season_start


def _last_stored_spring_date() -> str:
    """Return the most recent game_date in pitches_spring2026, or ST start."""
    st_start = "2026-02-15"
    if not table_exists(SPRING_TABLE):
        return st_start
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(
            f"SELECT MAX(game_date) FROM [{SPRING_TABLE}]"
        ).fetchone()
        last = result[0] if result and result[0] else None
    finally:
        conn.close()
    if last:
        d = datetime.strptime(last[:10], "%Y-%m-%d") + timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    return st_start


class LiveUpdater:
    """Pull daily Statcast data, score with Stuff+, update cards."""

    def __init__(self):
        self.predictor = StuffPlusPredictor()
        self.baselines = self._load_baselines()
        self._json_dir = os.path.join(PROFILES_DIR, "2026", "json")
        os.makedirs(self._json_dir, exist_ok=True)

    def _load_baselines(self):
        try:
            return load_baselines()
        except FileNotFoundError:
            logger.warning("No baselines found – run 'python main.py train' first")
            return None

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self):
        """Pull new data and regenerate affected cards."""
        today = datetime.today().strftime("%Y-%m-%d")
        start = _last_stored_date()

        if start > today:
            logger.info("Already up to date through today.")
            return

        logger.info(f"Fetching Statcast: {start} → {today}")
        df = download_date_range(start, today)

        if df is None or df.empty:
            logger.info("No new pitches returned.")
            return

        # pybaseball returns nullable Int64/boolean dtypes; cast to plain numpy types
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], "dtype") and hasattr(df[col].dtype, "numpy_dtype"):
                try:
                    df[col] = df[col].astype(df[col].dtype.numpy_dtype)
                except Exception:
                    df[col] = df[col].astype(object)

        logger.info(f"Scoring {len(df):,} new pitches…")
        df_scored = self.predictor.predict(df, baselines=self.baselines)

        # Persist raw data to pitches_2026, scored data to pitches_2026_scored
        append_to_db(df_scored, table=LIVE_TABLE)
        logger.info(f"Appended {len(df_scored):,} rows to '{LIVE_TABLE}'")

        # Rebuild full scored table from all raw rows scored fresh
        df_all_raw = load_from_db(LIVE_TABLE)
        if not df_all_raw.empty:
            df_all_scored = self.predictor.predict(df_all_raw, baselines=self.baselines)
            save_to_db(df_all_scored, table=f"{LIVE_TABLE}_scored", replace=True)

        # Regenerate cards for affected pitchers
        affected = df_scored["player_name"].dropna().unique()
        self._update_cards(affected)

    def _update_cards(self, pitcher_names: list):
        """
        Reload ALL 2026 scored data and regenerate cards for affected pitchers.
        Uses generate_all_cards so cross-pitcher recalibration stays accurate.
        """
        try:
            df_all = load_from_db(f"{LIVE_TABLE}_scored")
            if df_all.empty:
                return
            generate_all_cards(df_all, season="2026")
            logger.info(f"Regenerated 2026 cards after update ({len(pitcher_names)} new pitchers)")
        except Exception as e:
            logger.error(f"Error updating 2026 cards: {e}")

    # ------------------------------------------------------------------
    # Spring training live update
    # ------------------------------------------------------------------

    def update_spring(self):
        """
        Fetch any new spring training pitches since last stored date,
        append to pitches_spring2026, re-score all ST data, and
        regenerate spring2026 player cards.
        Stops automatically once today passes SPRING_END.
        """
        today = datetime.today().strftime("%Y-%m-%d")
        if today > SPRING_END:
            logger.info("Spring training has ended — skipping spring update.")
            return

        start = _last_stored_spring_date()
        if start > today:
            logger.info("Spring training data already up to date.")
            return

        # Use the Baseball Savant CSV endpoint (download_spring) rather than
        # pybaseball.statcast() so that fields like inning_topbot are populated.
        logger.info(f"Fetching spring training data via CSV endpoint")
        df_new = download_spring(2026)

        if df_new is None or df_new.empty:
            logger.info("No spring training pitches.")
            return

        logger.info(f"  {len(df_new):,} spring training pitches")

        # Mark as 'R' so engineering pipeline doesn't filter them out
        df_new["game_type"] = "R"

        # Replace the entire raw spring table so stale/duplicate rows are removed
        save_to_db(df_new, table=SPRING_TABLE, replace=True)

        # Re-score ALL spring training data for consistent population baselines
        try:
            df_all_spring = load_from_db(SPRING_TABLE)
            df_all_spring["game_type"] = "R"   # ensure filter passes
            df_scored = self.predictor.predict(df_all_spring, baselines=self.baselines)
            save_to_db(df_scored, table=f"{SPRING_TABLE}_scored", replace=True)
            generate_all_cards(df_scored, season="spring2026")
            logger.info("Spring training cards regenerated.")
        except Exception as e:
            logger.error(f"Error regenerating spring training cards: {e}")

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def run_scheduled(self, interval_seconds: int = 90):
        """Run update() immediately and then every interval_seconds."""
        logger.info(f"Live updater starting (interval = {interval_seconds}s)")
        self.update()

        schedule.every(interval_seconds).seconds.do(self.update)
        while True:
            schedule.run_pending()
            time.sleep(10)

    # ------------------------------------------------------------------
    # Manual / ad-hoc helpers
    # ------------------------------------------------------------------

    def backfill(self, start: str, end: str):
        """Manually backfill a specific date range into pitches_2026."""
        logger.info(f"Backfilling {start} → {end}")
        df = download_date_range(start, end)
        if df is None or df.empty:
            logger.info("Nothing to backfill.")
            return
        df_scored = self.predictor.predict(df, baselines=self.baselines)
        append_to_db(df_scored, table=LIVE_TABLE)
        logger.info(f"Backfilled {len(df_scored):,} pitches.")


def run_live(interval_seconds: int = 90):
    """Module-level entry point used by main.py cmd_live()."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    updater = LiveUpdater()
    updater.run_scheduled(interval_seconds=interval_seconds)
