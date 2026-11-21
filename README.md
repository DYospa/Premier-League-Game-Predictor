# Premier League Match Outcome Prediction

CSC 4740 Data Mining course project by **Anuraag Dasari** and **Doron Yospa**.

We predict the result of English Premier League matches (home win `H`, draw `D`, away win `A`) using classic data mining and machine learning methods, and compare every model against a bookmaker baseline. This is a three-class classification problem that uses only information available before kickoff.

## Why this project

Soccer is low scoring, so a single goal can change the result, and draws (about a quarter of matches) are easy to ignore. Prior work often reports only overall accuracy, uses older seasons, and skips comparison with bookmaker odds. We focus on those gaps by:

- testing on the most recent full season (2025-26),
- looking closely at draw prediction (precision, recall, F1),
- comparing every model with a bookmaker baseline on the same matches.

## Data

Match data comes from [football-data.co.uk](https://www.football-data.co.uk/englandm.php), one CSV per season, 2016-17 to 2025-26 (about 3,800 matches).

| Split | Seasons | Purpose |
|---|---|---|
| Train | 2016-17 to 2024-25 | Fit and tune models (expanding-window CV inside this range) |
| Test (headline) | 2025-26 | Locked until the final evaluation; used once |
| Extra test seasons | 2023-24, 2024-25 | Rolling-origin check: each is re-tuned using only earlier seasons |

The mid-project milestone (`pl_midreport.py`) used 2016-17 to 2023-24 for training and 2024-25 for validation. Splits are always by time, never random, so future games are never used to predict past ones.

Columns used: date, teams, full-time goals, result, shots on target, and Bet365 odds (`B365H`, `B365D`, `B365A`).

## Features

Every feature is computed from matches played **before** the one being predicted (rolling windows are shifted by one match), so there is no data leakage.

- Rolling averages over each team's last 5 matches: goals scored, goals conceded, shots on target, points.
- Venue-specific form: the home team's last 5 home games and the away team's last 5 away games.
- Head-to-head: the home team's average points over the last 3 meetings between the two clubs.
- Difference features: home value minus away value.

Matches where a team has fewer than 3 prior games are dropped. A missing head-to-head history is filled with a neutral value and flagged with an indicator column.

## Models and method

| Model | Tuned parameters |
|---|---|
| Always predict home win | none (baseline) |
| Always predict draw | none (baseline, shows how hard draw prediction is) |
| Bookmaker baseline (lowest odds) | none (baseline) |
| Logistic Regression | regularization strength `C` |
| Decision Tree | max depth, min samples per leaf |
| Random Forest | max depth, min samples per leaf (300 trees) |
| Naive Bayes (Gaussian) | variance smoothing |
| k-Nearest Neighbors | `k`, uniform vs distance weights |

- **Tuning:** expanding-window cross validation over seasons. Train on 2016-17 to 2019-20 and validate on 2020-21, then grow the training window one season at a time up to validating on 2024-25 (5 folds). Hyperparameters are chosen by log loss.
- **Draw handling:** draws are rarely the most likely outcome, so plain argmax almost never predicts them. We try two fixes: (1) a *draw multiplier* (grid 1.0 to 3.0) that scales the predicted draw probability before taking the argmax, tuned on out-of-fold predictions to maximize macro F1; (2) class-weighted (`balanced`) versions of Logistic Regression and Random Forest. The script warns if a tuned multiplier lands at the edge of the grid.
- **Naive Bayes calibration:** Gaussian Naive Bayes is overconfident because the features are correlated, so we also report a sigmoid-calibrated version.
- **Model choice:** the headline model is picked by cross-validated log loss on the training seasons, never by test performance.
- **Rolling origin:** one test season (about 380 matches) is too small to separate models, so the full pipeline (tuning included) is also run for test seasons 2023-24 and 2024-25, and the three seasons are pooled.

## Evaluation

- Accuracy, with a 95% bootstrap confidence interval
- Precision, recall, and F1 per class, with extra focus on draws; `Draw_Lift` is draw precision divided by the draw base rate (1.0 = no better than random)
- Confusion matrices
- Log loss, to judge predicted probabilities and not just the final pick
- Comparison with the bookmaker baseline on the same matches (odds converted to probabilities and normalized to remove the bookmaker margin), with two significance tests: a McNemar test on accuracy and a paired bootstrap interval on the log loss difference

## Repository structure

```
.
├── pl_midreport.py       # mid-project: cleaning, EDA, features, baselines
├── pl_final.py           # final: tuning, all models, rolling-origin + test evaluation
├── pl_search.py          # time-boxed feature/model search + one-shot final test
├── requirements.txt
├── data/                 # season CSVs (E0_1617.csv ... E0_2526.csv)
├── figures/              # generated plots
├── results/              # generated CSV / markdown tables
└── README.md
```

`figures/` and `results/` are created automatically when you run the scripts.

## Getting started

### Requirements

- Python 3.10+
- pandas 2.2+, numpy, matplotlib, scikit-learn

```
pip install -r requirements.txt
```

### Get the data

The scripts download any missing season file from football-data.co.uk. If you are offline, download the files manually and save them in `data/` using this naming pattern:

```
https://www.football-data.co.uk/mmz4281/1617/E0.csv  ->  data/E0_1617.csv
https://www.football-data.co.uk/mmz4281/1718/E0.csv  ->  data/E0_1718.csv
...
https://www.football-data.co.uk/mmz4281/2526/E0.csv  ->  data/E0_2526.csv
```

### Run

```
python pl_midreport.py   # cleaning, EDA, baselines on the 2024-25 validation season
python pl_final.py       # tuning + final evaluation (2025-26 headline, plus rolling origin)
```

`pl_final.py` prints CV and test results, writes the figures and tables below, and refreshes the Results section of this README automatically.

| File | Contents |
|---|---|
| `figures/fig1`–`fig7` | EDA, bookmaker calibration, feature correlations, baseline confusion matrices (from `pl_midreport.py`) |
| `figures/fig8_test_confusion_matrices.png` | Test-season confusion matrices |
| `figures/fig9_test_accuracy_logloss.png` | Test accuracy (with CI) and log loss |
| `figures/fig10_draw_metrics.png` | Draw precision / recall / F1, plain vs draw boost |
| `figures/fig11_rf_feature_importance.png` | Random Forest feature importance |
| `figures/fig12_draw_multiplier_tuning.png` | Out-of-fold draw multiplier tuning curves |
| `figures/fig13_rolling_origin.png` | Accuracy and log loss per test season and pooled |
| `results/baseline_results.csv` | Mid-project baseline metrics (validation season) |
| `results/search_log.csv`, `results/best_config.json` | Every search trial and the best config (from `pl_search.py`) |
| `results/final_best_model.csv` | One-shot 2025-26 test result of the search-best config vs reference and bookmaker |
| `results/cv_results.csv` | Best parameters, CV log loss, draw multiplier per model and test season |
| `results/final_results.csv` | Full 2025-26 test metrics, CIs, significance tests vs bookmaker |
| `results/rolling_results.csv` | Same metrics for 2023-24, 2024-25, 2025-26 and pooled |
| `results/final_results.md` | Same results as a markdown table |

## Feature and model search

`pl_search.py` runs a time-boxed random + mutation search over features (rolling or exponentially weighted form, venue form, head-to-head, Elo rating difference, rest days, draw-oriented closeness features) and models (Logistic Regression, Random Forest, k-NN, Gradient Boosting, plus two draw-aware models: an ordered logit with the order Away < Draw < Home, and a two-stage model that first predicts draw vs not-draw). Draw skill is tracked with draw AUC, which does not depend on a decision threshold. Every trial is scored by expanding-window CV log loss on 2016-17 to 2024-25.

```
caffeinate -i python pl_search.py --hours 2    # search; never reads the 2025-26 season
python pl_search.py --final                    # one evaluation of the best config on 2025-26
```

Search mode does not load the test season at all, and `--final` writes a lock file so the test season is evaluated only once. Once a season has been tested it cannot be used as a clean test again, so `--test-season 2627` rolls the protocol forward: 2025-26 becomes ordinary search data and the 2026-27 matches played so far become the new untouched test set. Because the best of thousands of trials is partly luck, the CV score of the winner is optimistic; the `--final` test number is the one we report. `--with-odds` adds bookmaker probabilities as features (a separate question from beating the bookmaker with form data alone) and logs to separate files.

## Results

<!-- RESULTS:START -->
Run `python pl_final.py` to generate this section.
<!-- RESULTS:END -->

## References

- Dixon, M. and Coles, S. (1997). Modelling association football scores and inefficiencies in the football betting market.
- Constantinou, A. and Fenton, N. (2013). Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries.
- Baboota, R. and Kaur, H. (2019). Predictive analysis and modelling football results using machine learning approach for English Premier League.
- Hubacek, O., Sourek, G., and Zelezny, F. (2019). Exploiting sports-betting market using machine learning.
- Bunker, R. and Thabtah, F. (2019). A machine learning framework for sport result prediction.](https://drive.google.com/file/d/1hjh0YyelTR3F1q9a5o17pYWTsY46amu4/view?usp=sharing)

## Acknowledgments

Match data and betting odds from [football-data.co.uk](https://www.football-data.co.uk/).
