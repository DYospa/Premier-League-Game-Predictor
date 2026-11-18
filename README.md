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
| Test | 2025-26 | Locked until the final evaluation; used once |

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
| Bookmaker baseline (lowest odds) | none (baseline) |
| Logistic Regression | regularization strength `C` |
| Decision Tree | max depth, min samples per leaf |
| Random Forest | max depth, min samples per leaf (300 trees) |
| Naive Bayes (Gaussian) | variance smoothing |
| k-Nearest Neighbors | `k`, uniform vs distance weights |

- **Tuning:** expanding-window cross validation over seasons. Train on 2016-17 to 2019-20 and validate on 2020-21, then grow the training window one season at a time up to validating on 2024-25 (5 folds). Hyperparameters are chosen by log loss.
- **Draw handling:** draws are rarely the most likely outcome, so plain argmax almost never predicts them. We try two fixes: (1) a *draw multiplier* that scales the predicted draw probability before taking the argmax, tuned on out-of-fold predictions to maximize macro F1; (2) class-weighted (`balanced`) versions of Logistic Regression and Random Forest.
- **Model choice:** the headline model is picked by cross-validated log loss on the training seasons, never by test performance.

## Evaluation

- Accuracy, with a 95% bootstrap confidence interval on the test season
- Precision, recall, and F1 per class, with extra focus on draws
- Confusion matrices
- Log loss, to judge predicted probabilities and not just the final pick
- Comparison with the bookmaker baseline on the same matches (odds converted to probabilities and normalized to remove the bookmaker margin), including the share of bootstrap resamples in which each model beats the bookmaker's accuracy

## Repository structure

```
.
├── pl_midreport.py       # mid-project: cleaning, EDA, features, baselines
├── pl_final.py           # final: tuning, all models, test-season evaluation
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
python pl_final.py       # tuning + final evaluation on the 2025-26 test season
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
| `results/baseline_results.csv` | Mid-project baseline metrics (validation season) |
| `results/cv_results.csv` | Best parameters, CV log loss, draw multiplier per model |
| `results/final_results.csv` | Full test-season metrics, CIs, comparison with bookmaker |
| `results/final_results.md` | Same results as a markdown table |

## Results

<!-- RESULTS:START -->
Run `python pl_final.py` to generate this section.
<!-- RESULTS:END -->

## References

- Dixon, M. and Coles, S. (1997). Modelling association football scores and inefficiencies in the football betting market.
- Constantinou, A. and Fenton, N. (2013). Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries.
- Baboota, R. and Kaur, H. (2019). Predictive analysis and modelling football results using machine learning approach for English Premier League.
- Hubacek, O., Sourek, G., and Zelezny, F. (2019). Exploiting sports-betting market using machine learning.
- Bunker, R. and Thabtah, F. (2019). A machine learning framework for sport result prediction.

## Acknowledgments

Match data and betting odds from [football-data.co.uk](https://www.football-data.co.uk/).
