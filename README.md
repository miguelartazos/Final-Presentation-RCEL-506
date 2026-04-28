# Business Empire — Data Science Final Submission

**Course:** RCEL 506 — Data Science  
**Author:** Maiky Artazos  
**Term:** Spring 2026

A data science project that uses card data, seeded simulations, ridge regression, and bot benchmarks to find balance problems before physical playtesting.

## Business problem

Manual game balancing does not scale. A team designing a 93-card strategy game cannot manually playtest every interaction between cards, strategies, draw orders, board positions, and bot opponents. The combinatorial test surface is too large to cover before the next physical playtest.

The business goal is to make balance decisions earlier. Instead of waiting for many physical playtests, the project uses repeatable simulations and simple models to identify cards, board positions, and bot policies that need review.

## Approach

1. **Real card data** — 93 parsed card markdown files become a structured DataFrame with derived features (ROI, payback, complexity).
2. **Simulation metrics** — a seeded Monte Carlo tournament produces `win_bias` labels and starting-zone win rates. Model outputs are compared against these simulated outcomes.
3. **Machine learning** — a ridge-regression model estimates card strength, and an expert linear scorer ranks legal actions for the playable bot. Both models are small enough to explain.

## Key results — naive baseline vs final

| Metric | Naive baseline | Final | Source |
|---|---|---|---|
| Card surrogate MAE on `win_bias` | 0.0180 (mean predictor) | 0.0163 (ridge OOF) | `reports/balance_summary.md` |
| Card surrogate R² (OOF) | 0.0000 | 0.0905 | same |
| Board zone gap (lower better) | 0.3067 | 0.1484 (−52%) | `reports/board_balance_report.json` |
| Weakest-zone win rate (higher better) | 0.3333 | 0.4516 | same |
| Bot win rate — medium (naive baseline) | 25.0% | — | `bot-ml/reports/bot-ml-summary.md` |
| Bot win rate — hard tuned heuristic | — | 61.3% (+8.51 score lift) | same |
| Bot win rate — expert ML scorer | — | 39.8% (+2.37 score lift) | same |

## How to reproduce

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then any of:

```bash
# 1. Main analysis notebook
.venv/bin/jupyter lab notebooks/final_pipeline.ipynb

# 2. Fast simulation check
.venv/bin/python3 src/live_smoke_demo.py

# 3. Focused tests
.venv/bin/python3 -m unittest discover -s src/tests -t src -p test_ml_balance_model.py -v
.venv/bin/python3 -m unittest discover -s src/tests -t src -p test_game_engine_v3.py -v
.venv/bin/python3 -m unittest discover -s src/tests -t src -p test_board_evaluator.py -v

# 3b. Optional full discovery suite
.venv/bin/python3 -m unittest discover -s src/tests -t src

# 4. Open the slide deck
# The final PPTX is committed directly.
open slides/Business_Empire_Final_v3.pptx

# 5. Streamlit app
.venv/bin/python3 -m streamlit run app/streamlit_demo.py

# 6. Regenerate the OWLET PDF
.venv/bin/python3 src/build_owlet_audit.py
```

## Streamlit app

The Streamlit app provides a quick way to run the simulator and inspect the card artwork. It has two tabs:

- **Balance demo:** runs the same seeded simulator used by `src/live_smoke_demo.py`.
- **Card gallery:** shows 61 compressed card thumbnails joined to `data/cards_dataset.csv` metadata.

Run locally:

```bash
.venv/bin/python3 -m streamlit run app/streamlit_demo.py
```

Streamlit Cloud deployment:

- Repository: this GitHub repo.
- Branch: `main`.
- Entrypoint: `app/streamlit_demo.py`.
- Dependency file: `app/requirements.txt`.
- Sharing: private app.

## Cross-reference: every deck claim → source file

| Slide | Claim | Source file |
|---|---|---|
| 5 | 93 cards, 56 businesses, 7 industries | `data/cards_dataset.csv` |
| 8 | Naive MAE 0.0180, Ridge OOF MAE 0.0163, R² 0.0905 | `reports/balance_summary.md` |
| 9 | Zone gap 0.3067 → 0.1484, weakest-zone win 0.333 → 0.452 | `reports/board_balance_report.json` |
| 9 | 0 card edits accepted (145 candidates evaluated) | `reports/search_trace.csv` |
| 10 | Bot ML coefficients (actionIncome +0.88, actionVP +0.83, actionCost −0.63) | `bot-ml/artifacts/optimized-expert-model.json` |
| 11 | Hard 61.3%, Expert 39.8%, Medium 25.0% | `bot-ml/reports/bot-ml-summary.md` |

## Model notes

- **R² = 0.0905 is modest.** Ridge is a diagnostic surrogate on N=56 with high simulation noise, not an acceptance gate. The MAE improvement of 9.4% over the naive mean is the baseline-comparison result.
- **0 card edits accepted is intentional.** `reports/search_trace.csv` shows 145 candidates evaluated and rejected. The optimizer rejected edits that did not hold up during confirmation runs.
- **The hard tuned heuristic (+8.51 score lift) outperforms the expert ML scorer (+2.37 score lift).** Both beat the medium baseline. On small-sample action ranking with well-engineered domain features, search beats parametric models.

## Repository map

```
Final-Presentation-RCEL-506/
├── README.md                          this file
├── requirements.txt
├── slides/
│   ├── Business_Empire_Final_v3.pptx  12 slides, regenerated with coefficient interpretation slide
│   └── card_asset_contact_sheet.png
├── notebooks/
│   └── final_pipeline.ipynb           main analysis notebook (ridge by hand, OOF, coefficient interpretation)
├── src/                               supporting modules + tests
│   ├── card_parser.py
│   ├── game_engine_v3.py
│   ├── ml_balance_model.py
│   ├── balance_optimizer.py
│   ├── board_*.py
│   ├── run_balance_pipeline.py
│   ├── live_smoke_demo.py
│   ├── build_owlet_audit.py
│   └── tests/
├── data/
│   └── cards_dataset.csv
├── app/                               Streamlit app
│   ├── streamlit_demo.py
│   ├── requirements.txt
│   └── streamlit_assets/cards/        61 compressed card thumbnails
├── bot-ml/                            selected bot ML files from the playable app
│   ├── README.md
│   ├── reports/                       benchmark reports and decision notes
│   ├── artifacts/                     optimized policy and ML model JSONs
│   └── source-snapshot/               TypeScript files for the bot decision code
├── reports/                           generated artifacts the deck references
│   ├── balance_summary.md
│   ├── balance_summary.json
│   ├── board_balance_report.json
│   ├── card_strength_model_report.json
│   ├── search_trace.csv
│   └── owlet/                         the OWLET deliverable (PDF + ternary chart)
```

## Where the actual ML is

Two places contain the machine learning work:

1. **`notebooks/final_pipeline.ipynb`**: report metrics are loaded first, then a smaller recompute repeats the ridge-regression method on standardized features (`(X.T X + αI)^{-1} X.T y`), 5-fold out-of-fold cross-validation, and coefficient interpretation table with business meaning.
2. **`bot-ml/artifacts/optimized-expert-model.json`**: a 10-feature standardized linear scorer that ranks legal candidate actions inside the playable Expo app. Coefficients shown on slide 10 of the deck.

Both models are intentionally simple because the dataset is small and the results need to be explainable.
