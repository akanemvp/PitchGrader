# Project Vision

## Problem
Coaches, analysts, and fans want to know **how good a pitcher's stuff is** —
independent of luck, defense, or the hitter. Traditional stats (ERA, strikeouts)
mix in everything that happened after the ball left the hand. What's missing is a
single, trustworthy number that describes the *quality of the pitch itself* from
its physical shape (velocity, movement, spin, release), and that works the same way
across every pitcher and every level of play.

## Intended Users
- **Player-development and pitching coaches** who want to evaluate a pitcher's
  arsenal pitch by pitch.
- **Amateur analysts and scouts** comparing prospects to big-league pitchers.
- **Curious fans** who want an at-a-glance grade for a pitcher's pitches.

## Goals
The project should help users:
- See a **per-pitch-type Stuff+ grade** for any tracked pitcher on one scale
  (100 = average, 10 points = one standard deviation).
- **Compare pitchers and pitches** across teams and levels using a common standard.
- View a **pitcher profile** with movement, velocity, and grades for each pitch.
- Browse **leaderboards** to find the best pitches of a given type.
- **Correct pitch-type labels** when the automatic classification is wrong and see
  how the grades change.

## Why This Project
This project sits at the intersection of my interest in baseball analytics and
software engineering. It uses real, public pitch-tracking data and turns it into a
working tool — covering data ingestion, machine learning, a web backend, and a
user-facing interface. It is meaningful because pitch-quality modeling is an active,
real-world problem in professional baseball, and building one end-to-end is strong
evidence of full-stack and data-engineering skills relevant to my future work.

## Initial Scope

**In the first version:**
- Ingest tracked pitch data and store it.
- Score every pitch with the Stuff+ model on the 100 / 10-per-SD scale.
- Generate per-pitcher profile cards and per-season leaderboards.
- Serve profiles and leaderboards through a web app.
- Allow manual pitch-type re-labeling with on-the-fly re-scoring.

**What waits until later:**
- User accounts, saved views, or social features.
- Automated, scheduled data refresh for every level.
- Predictive/forecasting features (e.g., projecting future performance).
- Mobile-native apps or polished production-grade UI.
