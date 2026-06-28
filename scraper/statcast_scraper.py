"""
Baseball Savant / Statcast data collection via pybaseball.

Downloads full season pitch-by-pitch data and stores it in SQLite.
Also supports incremental daily pulls for live 2026 updates.
"""

import io
import os
import logging
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests
import pybaseball

from config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# Regular-season date ranges per year
SEASON_DATES = {
    2020: ("2020-07-23", "2020-09-27"),  # COVID shortened season
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
    2026: ("2026-03-26", None),   # end=None means "up to today"
}


def _ensure_db_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_season(year: int) -> pd.DataFrame:
    """
    Download full season Statcast pitch data via pybaseball.
    Uses local cache to avoid re-downloading identical date ranges.
    """
    pybaseball.cache.enable()

    start, end = SEASON_DATES.get(year, (f"{year}-03-26", f"{year}-09-28"))
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    logger.info(f"Downloading Statcast data: {start} → {end}")
    df = pybaseball.statcast(start_dt=start, end_dt=end)
    logger.info(f"Downloaded {len(df):,} pitches for {year}")
    return df


def download_spring(year: int) -> pd.DataFrame:
    """
    Download spring training Statcast data for a given year directly from
    Baseball Savant. pybaseball skips spring training dates as 'offseason',
    so we scrape the bulk CSV endpoint with hfGT=S (spring training game type).
    Pulls Feb 15 → today in 3-day chunks to stay under Savant's 25k row cap.
    Spring training peaks at ~8-10k pitches/day; 3 days = ~24-30k, safely
    under the cap for the full spring schedule.
    """
    start_date = datetime(year, 2, 15)
    end_date   = datetime.today()

    frames = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=2), end_date)
        s   = chunk_start.strftime("%Y-%m-%d")
        e   = chunk_end.strftime("%Y-%m-%d")
        # game_date_lt is exclusive on Savant, so add 1 day to include chunk_end
        e_lt = (chunk_end + timedelta(days=1)).strftime("%Y-%m-%d")
        url = (
            "https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&hfGT=S%7C&hfSea={year}%7C"
            f"&game_date_gt={s}&game_date_lt={e_lt}"
            "&player_type=pitcher&type=details"
            "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
        )
        logger.info(f"Fetching spring training {s} → {e}")
        try:
            resp = requests.get(url, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            chunk_df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if not chunk_df.empty and "pitch_type" in chunk_df.columns:
                frames.append(chunk_df)
                logger.info(f"  Got {len(chunk_df):,} pitches")
        except Exception as exc:
            logger.warning(f"  Failed {s}→{e}: {exc}")

        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        logger.warning("No spring training data retrieved.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Drop the trailing summary row Baseball Savant sometimes appends
    if "pitch_type" in df.columns:
        df = df[df["pitch_type"].notna() & (df["pitch_type"] != "pitch_type")]
    logger.info(f"Spring training total: {len(df):,} pitches")
    return df


def download_date_range(start: str, end: str) -> pd.DataFrame:
    """Download a specific date range (used for live updates)."""
    pybaseball.cache.enable()
    logger.info(f"Downloading {start} → {end}")
    df = pybaseball.statcast(start_dt=start, end_dt=end)
    return df


def get_recent_pitches(days: int = 1) -> pd.DataFrame:
    """Fetch the last N days of Statcast data for live updates."""
    today = datetime.today()
    start = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    return download_date_range(start, end)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def save_to_db(df: pd.DataFrame, table: str = "pitches_train", replace: bool = True):
    """Persist a DataFrame to SQLite."""
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    mode = "replace" if replace else "append"
    df.to_sql(table, conn, if_exists=mode, index=False)
    conn.close()
    logger.info(f"Saved {len(df):,} rows → table '{table}' (mode={mode})")


def append_to_db(df: pd.DataFrame, table: str):
    """Append rows without replacing existing data.

    If the table already exists, align df columns to match it so new
    feature-engineering columns don't cause a schema mismatch.
    """
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        existing = pd.read_sql(f"SELECT * FROM [{table}] LIMIT 0", conn)
        shared = [c for c in existing.columns if c in df.columns]
        df = df[shared]
    except Exception:
        pass  # table doesn't exist yet — let save_to_db create it
    finally:
        conn.close()
    save_to_db(df, table=table, replace=False)


def load_from_db(table: str = "pitches_train") -> pd.DataFrame:
    """Load a full table from SQLite."""
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
    finally:
        conn.close()
    logger.info(f"Loaded {len(df):,} rows from '{table}'")
    return df


def table_exists(table: str) -> bool:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    result = cursor.fetchone() is not None
    conn.close()
    return result


def list_tables() -> list:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    conn.close()
    return tables
