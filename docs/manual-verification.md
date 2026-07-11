# Manual Verification

How the Sprint 2 prototype is checked by hand — what we run, what we look at, and
what "correct" means. The goal is to confirm behavior end-to-end in the real app, not
just that code compiles.

## How to run the app locally
```bash
pip install -r requirements.txt
python app.py                     # serves http://127.0.0.1:5001/
```
Open the homepage, pick a season from the dropdown, search a pitcher, and open their
profile.

## Feature slice: re-classify → re-score → preview
**What it should do:** relabeling a pitch in the editor updates that pitch's grade and
the movement-chart dots, as a preview, without changing the database.

**Steps**
1. Open a pitcher profile and click **Editor**.
2. Relabel one or more pitches (e.g. change a group of "SL" to "CU").
3. Return to the profile (preview mode).

**Expected result**
- The pitch-arsenal table shows the new pitch type with a recomputed Stuff+ grade.
- The **movement-profile dots recolor** to the new pitch type.
- A "Preview" banner is shown; refreshing the page reverts to the stored data (no DB
  write).

**How it was verified**
The `/api/reclassify_profile` endpoint was called directly with a `type_map` of
`{"SL": "CU"}` for a test pitcher. The response's per-pitch `movement_dots` label
counts moved all 138 slider dots into the curveball bucket (CU 73 → 211, SL → 0),
confirming the dots carry the re-classified labels rather than the original database
labels.

## Model scoring is stable and correct
**What it should do:** the model must produce the same grades after a code change, and
grades must be on the intended 100 = average / 10-per-SD scale.

**How it was verified**
- After refactors to the training/scoring code, the model was retrained and produced
  the **same version hash**, and re-scoring a data sample matched the stored
  production grades to within floating-point noise (mean absolute difference
  ≈ 0.00004; only a handful of rows differ, and those trace to per-batch median
  imputation, not the code change).
- Spot-checks: known elite pitches grade well above 100 and known weak pitches below
  it; per-pitcher/per-pitch-type means sit near 100 as expected.

## New season ingest (college 2026)
**What it should do:** ingesting a new level should produce populated raw + scored
tables and a working dropdown tab.

**How it was verified**
- The college ingest produced `pitches_college2026` / `pitches_college2026_scored`
  with **129,003 pitches across 407 tracked games and 1,493 pitchers**, plus
  generated profile cards and a leaderboard JSON.
- Endpoint checks: `/`, `/api/leaderboard/college2026`, and a player card all return
  HTTP 200, and the leaderboard returns sensible top grades.

## Date rendering
**What it should do:** every game date in the dropdown renders as a real date.

**How it was verified**
- Some games store `game_date` with a time suffix (`"2026-06-16 00:00:00"`), which
  previously rendered as "Invalid Date." `fmtGameDate` was hardened to strip the time
  and guard bad input; it now returns `"Jun 16"` for both `"2026-06-16"` and
  `"2026-06-16 00:00:00"`, and an empty string for null/malformed input.

## Regression checklist (run before committing UI/model changes)
- [ ] App boots and the homepage returns HTTP 200.
- [ ] Season dropdown loads; searching a pitcher opens their card.
- [ ] Editor re-classify updates both the arsenal table **and** the movement dots.
- [ ] Game dropdown dates all render (no "Invalid Date").
- [ ] Model change → retrain reproduces the same version hash / byte-identical grades.
