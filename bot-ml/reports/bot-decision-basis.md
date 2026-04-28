# Bot Decision Basis

## Summary

All app bots use the legal action generator. The bot policy never creates arbitrary reducer actions by itself; it receives legal candidate actions, scores or selects among them, and returns the same action shape used by the playable app.

## Random Diagnostic

`randomDiagnostic` is an offline benchmark policy only. It chooses from legal candidates and exists to catch benchmark or game-economy problems. It is **not** the app's easy difficulty and **not** the naive baseline for the comparison.

Latest seat-rotated result: `2.3%` win rate, `8.95` average score, `-18.20` score diff vs same-seat medium, `0` invalid actions.

## Easy

`easy` is the novice app bot. It takes simple productive actions and avoids advanced optimization. It is useful for gameplay experience, but it is not the data science baseline.

Latest seat-rotated result: `1.0%` win rate, `14.63` average score, `-12.52` score diff vs same-seat medium, `0` invalid actions.

## Medium

`medium` is the naive baseline. It is rule-based, legal-action-aware, and reproducible. It avoids presenting unrestricted random play as the comparison target.

Latest seat-rotated result: `25.0%` win rate, `27.15` average score, `0.00` score diff vs same-seat medium, `0` invalid actions.

## Hard

`hard` uses deterministic heuristic scoring over legal candidates. It was selected with Monte Carlo/autoresearch-style tuning under acceptance gates.

Current hard decision features include:

- action cost
- projected income
- valuation points
- board placement affinity
- current brand
- hand pressure
- temporary focus usage
- projected net value
- remaining quarters
- cash after action
- staff fit
- launch affordability
- legal-action availability
- finance exploit risk

Latest seat-rotated result: `61.3%` win rate, `35.66` average score, `+8.51` score diff vs same-seat medium, `0` invalid actions.

Current interpretation: hard is the strongest accepted policy.

## Expert

`expert` uses the same legal candidate generator as every other bot, then ranks candidates with the generated linear ML artifact in `src/utils/beBotModel.generated.ts`.

The model does not output raw moves. It scores legal actions only. This keeps the behavior explainable and prevents illegal moves.

Current expert features include:

- cash
- brand
- net profit
- business count
- hand count
- action cost
- action income
- action valuation points
- placement affinity
- temporary focus usage

Latest seat-rotated result: `39.8%` win rate, `29.52` average score, `+2.37` score diff vs same-seat medium, `0` invalid actions.

Current interpretation: expert ML improves over the medium baseline, but it does not beat the optimized hard heuristic in the current benchmark.

## Final Claim

The project has a reproducible data science loop:

1. define a naive baseline
2. generate legal candidate actions
3. run deterministic self-play simulations
4. tune hard with held-out gates
5. train and export an explainable ML scorer
6. compare policies by seat-rotated metrics
7. push the accepted artifacts into the playable app

The benchmark shows measurable improvement over baseline, zero invalid actions, and a clear reason why the optimized heuristic currently outperforms the linear ML scorer.
