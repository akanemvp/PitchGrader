"""
Stuff+ CLI entry point.

Commands
--------
  train     — Load training data, engineer features, train all models,
              score training data, and save pitcher-level norms.
  score     — Score each historical season table and write *_scored tables.
  profiles  — Load scored data and regenerate player cards / leaderboards.
  live      — Start the live 2026 update loop (delegates to live/live_update.py).
"""

import logging
import os
import sqlite3
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_train():
    from config import DB_PATH, TRAINING_SEASONS
    from features.engineering import engineer_features
    from model.train import train_all, save_baselines

    # Load raw columns only — pitches_train has 200+ pre-engineered columns
    # from the old architecture; we only need the raw inputs.
    RAW_COLS = """
        pitch_type, game_date, game_pk, player_name, pitcher, batter,
        p_throws, stand, balls, strikes, description, events,
        release_speed, release_spin_rate, release_extension,
        release_pos_x, release_pos_z, release_pos_y,
        pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot,
        vx0, vy0, vz0, ax, ay, az,
        spin_axis, arm_angle,
        delta_run_exp, estimated_woba_using_speedangle,
        bb_type, launch_speed,
        bat_score, home_score, away_score,
        post_bat_score, inning, outs_when_up, on_1b, on_2b, on_3b,
        game_year
    """
    sample_frac = None
    ensemble_movement_rv  = "--ensemble-movement-rv" in sys.argv
    count_rv              = "--count-rv" in sys.argv
    count_neutral         = "--count-neutral" in sys.argv
    swing_quality         = "--swing-quality" in sys.argv
    siera                 = "--siera" in sys.argv
    residual_location     = "--residual-location" in sys.argv
    residual_model        = "--residual-model" in sys.argv or "--linear-weights" in sys.argv or count_rv or swing_quality
    linear_weights        = "--linear-weights" in sys.argv
    for arg in sys.argv:
        if arg.startswith("--sample="):
            sample_frac = float(arg.split("=")[1])

    logger.info("Loading training data …")
    conn = sqlite3.connect(DB_PATH)
    seasons_sql = " UNION ALL ".join(
        f"SELECT {RAW_COLS} FROM pitches_{s}" for s in TRAINING_SEASONS
    )
    df = pd.read_sql(seasons_sql, conn)
    conn.close()
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
        logger.info(f"  Loaded {len(df):,} rows (sample={sample_frac}).")
    else:
        logger.info(f"  Loaded {len(df):,} rows.")

    logger.info("Training models …")
    if ensemble_movement_rv:
        logger.info("  Using ensemble-based movement_rv (2-pass training)")
    if count_neutral:
        logger.info("  Using count-neutral ensemble (swing residual approach)")
    elif swing_quality:
        logger.info("  Using swing quality target (whiff + xwOBAcon, swings only)")
    elif count_rv:
        logger.info("  Using count-stratified RV target (TJ Stats approach)")
    elif linear_weights:
        logger.info("  Using linear weights target (context-neutral per-pitch RV)")
    elif residual_model:
        logger.info("  Using residual xRV model (single regressor per family)")
    elif residual_location:
        logger.info("  Using residual pipeline: Location+ → residual → Stuff+")
    elif siera:
        logger.info("  Using SIERA-calibrated target: Location+(siera_rv) → residual → Stuff+")
    from model.train import train_unified
    train_unified(df, ensemble_movement_rv=ensemble_movement_rv, residual_model=residual_model, linear_weights=linear_weights, count_rv=count_rv, count_neutral=count_neutral, swing_quality=swing_quality, residual_location=residual_location, siera=siera)

    logger.info("Done.")



def cmd_score():
    from config import DB_PATH
    from features.engineering import engineer_features
    from model.predict import StuffPlusPredictor

    predictor = StuffPlusPredictor()

    # 2023/2024/2025: use historical norms (2020-2024 baseline) to avoid contamination
    # 2026 seasons: use current norms (2020-2025 baseline)
    season_norm = {
        "2023": "historical", "2024": "historical", "2025": "historical",
        "spring2026": "current", "breakout2026": "current",
        "2026": "current",
    }
    seasons = list(season_norm.keys())
    for s in seasons:
        src_table   = f"pitches_{s}"
        dst_table   = f"pitches_{s}_scored"
        conn = sqlite3.connect(DB_PATH)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if src_table not in tables:
                logger.info(f"  No table {src_table} — skipping.")
                conn.close()
                continue
            df = pd.read_sql(f"SELECT * FROM [{src_table}]", conn)
        finally:
            conn.close()

        if df.empty:
            logger.info(f"  {src_table} is empty — skipping.")
            continue

        norm_set = season_norm[s]
        logger.info(f"Scoring {src_table} ({len(df):,} rows) [norm={norm_set}] …")
        df_eng, _ = engineer_features(df, baselines=predictor.baselines)
        df_scored = predictor.predict(df_eng, already_engineered=True, norm_set=norm_set)

        df_scored["stuff_plus"] = df_scored["stuff_plus"].clip(50.0, 160.0)

        conn = sqlite3.connect(DB_PATH, timeout=60)
        df_scored.to_sql(dst_table, conn, if_exists="replace", index=False, chunksize=10000)
        conn.close()
        logger.info(f"  Written → {dst_table}")

    logger.info("Done.")


def cmd_profiles():
    from config import DB_PATH
    from profiles.player_cards import generate_all_cards

    seasons = [
        ("2023",         "2023"),
        ("2024",         "2024"),
        ("2025",         "2025"),
        ("spring2026",   "spring2026"),
        ("breakout2026", "breakout2026"),
        ("2026",         "2026"),
    ]
    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()

    for table_suffix, season_int in seasons:
        table = f"pitches_{table_suffix}_scored"
        if table not in tables:
            logger.info(f"  No table {table} — skipping season {season_int}.")
            continue

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
        conn.close()

        if df.empty:
            logger.info(f"  {table} is empty — skipping.")
            continue

        logger.info(f"Building profiles for season {season_int} ({len(df):,} pitches) …")
        generate_all_cards(df, season=season_int, skip_png="--skip-png" in sys.argv)

    logger.info("Done.")


def cmd_live():
    from live.live_update import run_live
    run_live()


if __name__ == "__main__":
    commands = {
        "train":    cmd_train,
        "score":    cmd_score,
        "profiles":     cmd_profiles,
        "live":         cmd_live,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python main.py [{' | '.join(commands)}]")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    commands[sys.argv[1]]()
