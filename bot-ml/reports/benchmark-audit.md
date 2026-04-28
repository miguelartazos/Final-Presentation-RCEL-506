# Benchmark Audit

## Why The Previous Result Was Not Trustworthy

The previous leaderboard compared one target bot against medium opponents from a fixed target seat. It also treated unrestricted random legal play as `easy`, which made the app difficulty look stronger than the taught bots whenever random exploration extended the game or took extra finance actions.

## What Changed

- `randomDiagnostic` is now an offline-only policy.
- App `easy` is a novice productive policy, not unrestricted random legal play.
- Every policy is evaluated from every seat on the same seeds.
- Reports include average rank, same-seat score lift, safe no-op rate, finance action rate, and game length.

## How To Interpret A Random Diagnostic Win

If `randomDiagnostic` beats medium, the output is a benchmark or game-balance warning. It is not treated as evidence that random play is intelligent, and it is not used as the baseline for the comparison.
