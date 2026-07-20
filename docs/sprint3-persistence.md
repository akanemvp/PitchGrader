# Sprint 3 — Persistence: Saving Pitch Re-Classifications

## The problem
Sprint 2 delivered the pitch-type editor: a user re-labels a mislabeled pitch, the
model re-scores it, and the profile shows updated grades. But the correction is
**entirely ephemeral** — it lives in the browser's `sessionStorage`, is re-scored in
memory by `/api/reclassify_profile`, and **nothing is ever written to disk**. Refresh
the page and the correction is gone.

That makes the feature a demo rather than a tool. Pitch-type mislabeling is a real,
recurring data-quality problem (Statcast's classifier regularly confuses
slider/sweeper/cutter, and the minor-league gamefeed is worse), so a correction is
only valuable if it *sticks*.

**Sprint 3 goal:** give corrections durable, queryable storage so they survive
refreshes, restarts, and re-scrapes — and are automatically applied wherever pitches
are read or scored.

## Why a separate table (the key design decision)
The obvious approach — just `UPDATE pitches_<season>_scored SET pitch_type = ...` —
is wrong for this project, and the reason is concrete:

> The live updaters **rewrite whole season tables** (`to_sql(..., if_exists="replace")`
> in `update_aaa2026.py`, `update_futures2026.py`, etc.). An in-place edit would be
> silently erased on the next scrape.

So corrections get their own table. Benefits:
1. **Durable across re-ingestion** — re-scraping never touches the overrides table.
2. **Non-destructive** — the raw scraped label is preserved in `original_type`, so
   every correction is auditable and revertible.
3. **Explicit application** — `apply_overrides()` is a single, testable step in the
   read/score path rather than hidden mutation.

Trade-off accepted: reads now require a join/merge step. At current scale (hundreds of
overrides vs. millions of pitches) that cost is negligible, and it's indexed.

## Storage choice
**SQLite** — the same database already holding every pitch table, accessed with plain
`sqlite3` (no ORM), consistent with the rest of the project.

Alternatives considered:
- **JSON file** — simple, but no atomic upserts, no indexed lookups, and risks
  clobbering on concurrent writes from the web app and the launchd updaters.
- **A new relational DB (Postgres)** — real concurrency and constraints, but
  operationally heavy for a single-user local app and would fracture the datastore.

SQLite wins: it already exists, gives atomic upserts and a real `UNIQUE` constraint,
and keeps everything in one file.

## Data model
One table, `pitch_overrides`, keyed by the same composite identity the app already
uses for a pitch.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement surrogate key |
| `season` | TEXT | season key (`2026`, `aaa2026`, `college2026`, …) |
| `game_pk` | INTEGER | part of the natural key |
| `at_bat_number` | INTEGER | part of the natural key |
| `pitch_number` | INTEGER | part of the natural key |
| `player_name` | TEXT | pitcher, denormalized for fast lookup |
| `original_type` | TEXT | label before the correction (audit trail) |
| `new_type` | TEXT | the corrected pitch type |
| `source` | TEXT | provenance, defaults to `editor` |
| `created_at` / `updated_at` | TEXT | UTC ISO-8601 timestamps |

**Natural key / uniqueness:** `UNIQUE (season, game_pk, at_bat_number, pitch_number)`
— one correction per pitch; re-editing the same pitch **upserts** rather than
duplicating.

**Relationship:** each row references exactly one pitch in `pitches_<season>` /
`pitches_<season>_scored` via that composite key. It's a logical foreign key (not
enforced, since the pitch tables are replaced wholesale by re-scrapes — an override
intentionally survives its pitch row being rewritten).

**Indexes:** `(season, player_name)` for profile lookups, `(season, game_pk)` for the
game view — the two access patterns the UI actually uses.

## Implementation status
`storage/overrides.py` implements the layer:

| Function | Purpose |
|---|---|
| `init_schema()` | idempotent `CREATE TABLE IF NOT EXISTS` + indexes |
| `save_override()` | upsert one correction (`ON CONFLICT … DO UPDATE`) |
| `save_many()` | bulk-save a batch from the editor |
| `get_overrides()` | read, optionally filtered by pitcher / game |
| `delete_override()` | revert a pitch to its scraped label |
| `apply_overrides(df, season)` | apply stored corrections to any pitch DataFrame |
| `count_overrides()` | totals, for verification |

### HTTP API (`app.py`)
A small CRUD surface over the store:

| Endpoint | Purpose |
|---|---|
| `POST /api/overrides` | save corrections; looks up and stores the scraped `original_type` |
| `GET /api/overrides/<season>` | list corrections, filterable by `?player=` / `?game_pk=` |
| `DELETE /api/overrides` | revert one pitch to its scraped label |

### Scoring integration (`main.py`)
`cmd_score` calls `apply_overrides()` on each season's pitches **before** feature
engineering, so saved corrections flow into the grades on every re-score. Because the
raw tables are replaced by re-scrapes, re-applying here on each run is exactly what
makes a correction durable.

**Verified working (end-to-end via the API):**
1. `POST` a correction → saved, with `original_type` captured automatically (`CH`).
2. `GET` → reads the record back.
3. `apply_overrides()` → rewrites that pitch `CH` → `ST`.
4. Row present in the on-disk `pitch_overrides` table (survives process restart).
5. `DELETE` → reverts; count returns to zero.

### Editor UI (`templates/editor.html`)
A **Save Changes** button posts the pending corrections to `POST /api/overrides` and
reports how many were saved. This is the meaningful distinction from Sprint 2's
"View in Profile", which remains a throwaway in-memory preview:

| Action | Behavior |
|---|---|
| *View in Profile* | in-memory preview; discarded on refresh (Sprint 2) |
| **Save Changes** | writes to `pitch_overrides`; survives refresh/restart/re-scrape |

Verified through the running app: saving two pitches returned
`{"saved": 2, "season_total": 2}` with the scraped labels captured as
`original_type` (`CH`→`ST`, `FF`→`ST`).

### Applying corrections in the read path
Saving alone wasn't enough: the correction was durable in the database, but the app
still served the original label because `apply_overrides` only ran at scoring time.
A user would save, see "Saved N corrections," refresh, and see no change.

Two additions closed that:
- **`api_player_pitches_raw`** applies saved overrides, so the editor shows the user's
  saved work after a refresh.
- **Saving rebuilds that pitcher's profile card** so the arsenal table updates
  immediately. This is cheap because **`pitch_type` is not a model feature** — each
  pitch keeps its grade, and only the aggregation (which type a pitch counts toward)
  changes. No re-scoring required.

One trap worth recording: `generate_all_cards` **rewrites the season leaderboard** from
whatever frame it is given, so rebuilding a single pitcher truncated a 721-entry
leaderboard to one row and broke search. The rebuild now snapshots the leaderboard,
regenerates the card, and merges that pitcher's entry back in.

### Revert
| Action | Scope |
|---|---|
| *Clear all changes* | drops **pending** edits that were never saved |
| **Revert saved** | deletes **persisted** corrections and restores scraped labels |

`POST /api/overrides/revert` clears every correction for a pitcher/season;
`DELETE /api/overrides` reverts a single pitch. Both rebuild the card.

**Verified cycle:** save 5 corrections → card and editor both show FF 1042 / SI 5 →
re-request (refresh) → still 1042/5 → revert → back to FF 1047, SI gone.

## Remaining work
1. Decide override precedence when a re-scrape changes a pitch's original label.
2. Apply overrides in the game-detail read path (season and editor views are done).
3. Surface saved corrections visually in the editor (e.g. a marker on corrected
   pitches) so a user can see what is already persisted before editing.

## Biggest risk
**Key stability across re-scrapes.** Overrides are tied to
`(game_pk, at_bat_number, pitch_number)`. If a provider ever renumbers at-bats or
pitches for a game, a stored override would silently attach to the *wrong* pitch.
Mitigation under consideration: store `original_type` (already done) and skip applying
an override when the current label no longer matches what was corrected — turning a
silent mis-apply into a detectable, ignorable no-op.
