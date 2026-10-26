[README (1).md](https://github.com/user-attachments/files/33970929/README.1.md)
# Premier-League-Game-Predictor# Premier League Match Outcome Prediction

CSC 4740 Data Mining course project by **Anuraag Dasari** and **Doron Yospa**.

We predict the result of English Premier League matches (home win `H`, draw `D`, away win `A`) using classic data mining and machine learning methods, and compare every model against a bookmaker baseline. This is a three-class classification problem that uses only information available before kickoff.

> **Project status: mid-project milestone.** Data cleaning, exploratory data analysis, feature engineering, and baseline models are done. Model tuning and the final test-season evaluation are in progress (see [Roadmap](#roadmap)).

## Why this project

Soccer is low scoring, so a single goal can change the result, and draws (about a quarter of matches) are easy to ignore. Prior work often reports only overall accuracy, uses older seasons, and skips comparison with bookmaker odds. We focus on those gaps by:

- testing on the most recent full season (2025-26),
- looking closely at draw prediction (precision, recall, F1),
- comparing every model with a bookmaker baseline on the same matches.

## Data

Match data comes from [football-data.co.uk](https://www.football-data.co.uk/englandm.php), one CSV per season, 2016-17 to 2025-26 (about 3,800 matches).

| Split | Seasons | Purpose |
|---|---|---|
| Train | 2016-17 to 2023-24 | Fit models (mid-project) |
| Validation | 2024-25 | Compare baselines (mid-project) |
| Test | 2025-26 | **Locked.** Used only for final experiments |

For the final report the training set expands to 2016-17 through 2024-25. Splits are by time, never random, so future games are never used to predict past ones.

Columns used: date, teams, full-time goals, result, shots on target, and Bet365 odds (`B365H`, `B365D`, `B365A`).

## Features

Every feature is computed from matches played **before** the one being predicted (rolling windows are shifted by one match), so there is no data leakage.

- Rolling averages over each team's last 5 matches: goals scored, goals conceded, shots on target, points.
- Venue-specific form: the home team's last 5 home games and the away team's last 5 away games.
- Head-to-head: the home team's average points over the last 3 meetings between the two clubs.
- Difference features: home value minus away value.

Matches where a team has fewer than 3 prior games are dropped. A missing head-to-head history is filled with a neutral value and flagged with an indicator column.

## Models

| Model | Status |
|---|---|
| Always predict home win | Done (baseline) |
| Bookmaker baseline (lowest odds) | Done (baseline) |
| Logistic Regression | Done (baseline) |
| Decision Tree | Planned |
| Random Forest | Planned |
| Naive Bayes | Planned |
| k-Nearest Neighbors | Planned |

Planned tuning: expanding-window cross validation on the training seasons, class weighting, and an adjustable decision threshold for draws.

## Evaluation

- Accuracy
- Precision, recall, and F1 per class, with extra focus on draws
- Confusion matrices
- Log loss, to judge predicted probabilities and not just the final pick
- Comparison with the bookmaker baseline on the same season (bookmaker probabilities are the odds converted to probabilities and normalized to remove the bookmaker margin)

## Repository structure

```
.
├── pl_midreport.py       # cleaning, EDA, features, baselines
├── data/                 # season CSVs (E0_1617.csv ... E0_2526.csv)
├── figures/              # generated EDA and confusion-matrix plots
├── results/              # generated baseline_results.csv
└── README.md
```

`figures/` and `results/` are created automatically when you run the script.

## Getting started

### Requirements

- Python 3.10+
- pandas 2.2+, numpy, matplotlib, scikit-learn

```bash
pip install "pandas>=2.2" numpy matplotlib scikit-learn
```

### Get the data

The script downloads any missing season file from football-data.co.uk. If you are offline, download the files manually and save them in `data/` using this naming pattern:

```
https://www.football-data.co.uk/mmz4281/1617/E0.csv  ->  data/E0_1617.csv
https://www.football-data.co.uk/mmz4281/1718/E0.csv  ->  data/E0_1718.csv
...
https://www.football-data.co.uk/mmz4281/2526/E0.csv  ->  data/E0_2526.csv
```

### Run

```bash
python pl_midreport.py
```

The script prints data-cleaning checks, EDA statistics, and baseline results, and saves:

| File | Contents |
|---|---|
| `figures/fig1_class_distribution.png` | Home / draw / away split |
| `figures/fig2_results_by_season.png` | Result rates by season |
| `figures/fig3_goals_distribution.png` | Goals per team per match |
| `figures/fig4_odds_calibration.png` | Bookmaker probabilities vs observed frequencies |
| `figures/fig5_draw_vs_odds_gap.png` | Draw rate vs how even the match looks |
| `figures/fig6_feature_correlation.png` | Correlations among engineered features |
| `figures/fig7_confusion_matrices.png` | Baseline confusion matrices |
| `results/baseline_results.csv` | Baseline metrics table |

The script never reads the 2025-26 test season for modeling. It is set aside before any analysis.

## Results

Baseline results on the 2024-25 validation season are written to `results/baseline_results.csv` when you run the script. Final test-season results will be added here after the final experiments.

## Roadmap

- [x] Download and clean data
- [x] Exploratory data analysis
- [x] Rolling form and head-to-head features
- [x] Baseline models and bookmaker comparison
- [ ] Remaining models (Decision Tree, Random Forest, Naive Bayes, k-NN)
- [ ] Cross-validated tuning, class weighting, draw thresholds
- [ ] Final experiments on the 2025-26 test season
- [ ] Final report, cleaned code, demo video

## References

- Dixon, M. and Coles, S. (1997). Modelling association football scores and inefficiencies in the football betting market.
- Constantinou, A. and Fenton, N. (2013). Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries.
- Baboota, R. and Kaur, H. (2019). Predictive analysis and modelling football results using machine learning approach for English Premier League.
- Hubacek, O., Sourek, G., and Zelezny, F. (2019). Exploiting sports-betting market using machine learning.
- Bunker, R. and Thabtah, F. (2019). A machine learning framework for sport result prediction.

## Acknowledgments

Match data and betting odds from [football-data.co.uk](https://www.football-data.co.uk/).
