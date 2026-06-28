# Requirements

## Functional Requirements
What the first useful version of the project should do:

- The system should **ingest pitch-tracking data** for a season and store each pitch.
- The system should **engineer physical shape features** for each pitch (velocity,
  vertical and horizontal movement, spin rate, release point, and arm angle).
- The system should **score every pitch** with the Stuff+ model on a fixed scale
  where 100 is league average and 10 points equals one standard deviation.
- The system should **summarize a pitcher** into a profile that shows a Stuff+ grade
  for each of their pitch types, along with usage and movement.
- The system should **generate leaderboards** ranking the best pitches of each type.
- The system should let a user **re-label a pitch's type** and **re-score** to see
  the updated grades.
- The system should **serve profiles and leaderboards through a web interface**.

## Data Requirements
What information the project needs to store, track, retrieve, or update:

- The system needs to **store raw pitch records** (velocity, movement, spin, release
  position, pitch type, game, batter handedness) for each tracked pitch.
- The system needs to **store the scored output** (the Stuff+ value) for each pitch.
- The system needs to **track per-pitcher, per-season summaries** (grades by pitch
  type, innings pitched, and basic results).
- The system needs to **store the trained model** and the frozen league norms used to
  put grades on the 100 / 10-per-SD scale.
- The system needs to **retrieve** a pitcher's pitches on demand to rebuild profiles
  or re-score after a manual edit.

## Non-Functional Requirements
What qualities the project should have:

- The project should be **understandable to a new user** — grades on one clear scale
  with a readable profile.
- The project should **give clear feedback** when an action (such as a pitch-type
  edit) changes the results.
- The project should **keep data organized and consistent** across seasons and levels
  (raw data, scored data, and summaries stay in sync).
- The project should **score the same pitch the same way every time** so grades are
  reproducible.
- The project should be **reasonably responsive** when serving a profile or
  leaderboard from already-scored data.

## Out of Scope for the First Version
- This version will not include **user accounts, logins, or saved preferences**.
- This version will not **predict or forecast** future pitcher performance.
- This version will not provide a **mobile-native app**.
- This version will not guarantee **complete coverage of untracked games** (some
  minor-league games have no pitch-tracking data and cannot be scored).
