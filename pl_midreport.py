"""
CSC 4740 Mid Project Report: Premier League match outcome prediction
Cleans data, runs EDA, builds pre-match features, trains baselines.

Usage:
    python pl_midreport.py

Data: reads data/E0_1617.csv ... data/E0_2526.csv. If a file is missing it
tries to download it from football-data.co.uk. If you have no internet on
this machine, download the CSVs in a browser and save them in ./data/:
    https://www.football-data.co.uk/mmz4281/1617/E0.csv  -> data/E0_1617.csv
    https://www.football-data.co.uk/mmz4281/1718/E0.csv  -> data/E0_1718.csv
    ... and so on through 2526.

The 2025-26 season is NOT used here. It is held out for final testing.
Outputs: figures/*.png and results/*.csv (and numbers printed to console).
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, log_loss, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TRAIN_SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
VAL_SEASON = "2425"           # validation for the mid report
TEST_SEASON = "2526"          # locked until final experiments
CLASSES = ["H", "D", "A"]
WINDOW = 5
MIN_GAMES = 3

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------- load + clean
def load_season(code):
    path = f"data/E0_{code}.csv"
    if not os.path.exists(path):
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["Season"] = code
    return df


raw = pd.concat([load_season(s) for s in SEASONS], ignore_index=True)

keep = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HST", "AST", "B365H", "B365D", "B365A"]
df = raw[[c for c in keep if c in raw.columns]].copy()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
df = df.sort_values("Date").reset_index(drop=True)
df["match_id"] = df.index

print("=== DATA CLEANING ===")
print("Total matches:", len(df), "| seasons:", df["Season"].nunique())
print("Matches per season:\n", df.groupby("Season").size().to_string())
print("Missing values:\n", df.isna().sum()[lambda s: s > 0].to_string() or "none")
calc = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
print("FTR mismatches with goals:", int((calc != df.FTR).sum()))

# Locked test season: set aside, never used below
test_locked = df[df.Season == TEST_SEASON].copy()
df = df[df.Season != TEST_SEASON].copy()

# Bookmaker implied probabilities (normalized to remove margin)
inv = 1.0 / df[["B365H", "B365D", "B365A"]]
df["overround"] = inv.sum(axis=1) - 1
probs = inv.div(inv.sum(axis=1), axis=0)
df[["pH", "pD", "pA"]] = probs.values

# ------------------------------------------------------------------------ EDA
print("\n=== EDA ===")
dist = df.FTR.value_counts(normalize=True).reindex(CLASSES) * 100
print("Result distribution (%):\n", dist.round(1).to_string())
print("Mean home goals: %.2f | mean away goals: %.2f" % (df.FTHG.mean(), df.FTAG.mean()))
one_goal_or_draw = ((df.FTHG - df.FTAG).abs() <= 1).mean() * 100
print("Matches decided by <=1 goal (incl. draws): %.1f%%" % one_goal_or_draw)
print("Mean bookmaker overround: %.2f%%" % (df.overround.mean() * 100))

# Fig 1: class distribution
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Home win", "Draw", "Away win"], dist.values, color=["#2a9d8f", "#e9c46a", "#e76f51"])
for i, v in enumerate(dist.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.set_ylabel("% of matches")
ax.set_title("Fig 1: Result distribution (2016-17 to 2024-25)")
plt.tight_layout(); plt.savefig("figures/fig1_class_distribution.png", dpi=150); plt.close()

# Fig 2: results by season
by_season = pd.crosstab(df.Season, df.FTR, normalize="index").reindex(columns=CLASSES) * 100
print("Result % by season:\n", by_season.round(1).to_string())
fig, ax = plt.subplots(figsize=(7, 4))
for c, col in zip(CLASSES, ["#2a9d8f", "#e9c46a", "#e76f51"]):
    ax.plot(by_season.index, by_season[c], marker="o", label=c, color=col)
ax.set_ylabel("% of matches"); ax.set_xlabel("Season"); ax.legend()
ax.set_title("Fig 2: Results by season")
plt.tight_layout(); plt.savefig("figures/fig2_results_by_season.png", dpi=150); plt.close()

# Fig 3: goals distribution
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.arange(0, 8) - 0.5
ax.hist([df.FTHG.clip(upper=6), df.FTAG.clip(upper=6)], bins=bins,
        label=["Home goals", "Away goals"], color=["#2a9d8f", "#e76f51"])
ax.set_xlabel("Goals (6 = 6+)"); ax.set_ylabel("Matches"); ax.legend()
ax.set_title("Fig 3: Goals per team per match")
plt.tight_layout(); plt.savefig("figures/fig3_goals_distribution.png", dpi=150); plt.close()

# Fig 4: bookmaker calibration
fig, ax = plt.subplots(figsize=(5, 5))
for c, col, name in [("H", "#2a9d8f", "pH"), ("D", "#e9c46a", "pD"), ("A", "#e76f51", "pA")]:
    b = pd.cut(df[name], bins=np.linspace(0, 1, 11))
    g = df.groupby(b, observed=True).apply(lambda x: pd.Series(
        {"pred": x[name].mean(), "obs": (x.FTR == c).mean(), "n": len(x)}), include_groups=False)
    g = g[g.n >= 15]
    ax.plot(g.pred, g.obs, marker="o", label=c, color=col)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Bookmaker implied probability"); ax.set_ylabel("Observed frequency")
ax.legend(); ax.set_title("Fig 4: Bookmaker calibration")
plt.tight_layout(); plt.savefig("figures/fig4_odds_calibration.png", dpi=150); plt.close()

# Fig 5: draw rate vs gap between home and away implied prob
df["gap"] = (df.pH - df.pA).abs()
gb = pd.qcut(df.gap, 8)
dr = df.groupby(gb, observed=True).apply(
    lambda x: pd.Series({"gap": x.gap.mean(), "draw_rate": (x.FTR == "D").mean() * 100}),
    include_groups=False)
print("Draw rate by odds gap:\n", dr.round(2).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dr.gap, dr.draw_rate, marker="o", color="#e9c46a")
ax.set_xlabel("|P(home) - P(away)| from odds"); ax.set_ylabel("Draw rate (%)")
ax.set_title("Fig 5: Draws are more likely in even matches")
plt.tight_layout(); plt.savefig("figures/fig5_draw_vs_odds_gap.png", dpi=150); plt.close()

# ----------------------------------------------------------- feature building
def build_features(matches):
    """Rolling pre-match features. Every rolling stat is shifted by one match."""
    m = matches.sort_values("Date").reset_index(drop=True)
    pts_h = np.where(m.FTR == "H", 3, np.where(m.FTR == "D", 1, 0))
    pts_a = np.where(m.FTR == "A", 3, np.where(m.FTR == "D", 1, 0))
    home = pd.DataFrame({"match_id": m.match_id, "Date": m.Date, "team": m.HomeTeam, "venue": "H",
                         "gf": m.FTHG, "ga": m.FTAG, "sot": m.HST, "pts": pts_h})
    away = pd.DataFrame({"match_id": m.match_id, "Date": m.Date, "team": m.AwayTeam, "venue": "A",
                         "gf": m.FTAG, "ga": m.FTHG, "sot": m.AST, "pts": pts_a})
    long = pd.concat([home, away]).sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    stats = ["gf", "ga", "sot", "pts"]

    def roll(g):
        return g[stats].shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()

    all_form = long.groupby("team", group_keys=False).apply(roll, include_groups=False)
    all_form.columns = [f"{c}_all" for c in stats]
    ven_form = long.groupby(["team", "venue"], group_keys=False).apply(roll, include_groups=False)
    ven_form.columns = [f"{c}_ven" for c in stats]
    long = pd.concat([long, all_form, ven_form], axis=1)

    h = long[long.venue == "H"].set_index("match_id")
    a = long[long.venue == "A"].set_index("match_id")
    feat = pd.DataFrame(index=h.index)
    for c in stats:
        feat[f"d_{c}_all"] = h[f"{c}_all"] - a[f"{c}_all"]
        feat[f"d_{c}_ven"] = h[f"{c}_ven"] - a[f"{c}_ven"]   # home form at home vs away form away

    # head-to-head: home team's avg points over last 3 meetings (any venue)
    hist, h2h, h2h_missing = {}, {}, {}
    for r in m.itertuples():
        key = frozenset((r.HomeTeam, r.AwayTeam))
        prev = hist.get(key, [])[-3:]
        if prev:
            h2h[r.match_id] = np.mean([p for t, p in prev if t == r.HomeTeam] +
                                      [3 - p if p != 1 else 1 for t, p in prev if t != r.HomeTeam])
            h2h_missing[r.match_id] = 0
        else:
            h2h[r.match_id], h2h_missing[r.match_id] = 1.0, 1
        res_pts = 3 if r.FTR == "H" else (1 if r.FTR == "D" else 0)
        hist.setdefault(key, []).append((r.HomeTeam, res_pts))
    feat["h2h_pts"] = pd.Series(h2h)
    feat["h2h_missing"] = pd.Series(h2h_missing)
    return feat


feat = build_features(df)
data = df.set_index("match_id").join(feat)
n_before = len(data)
data = data.dropna(subset=[c for c in feat.columns if c != "h2h_missing"])
print("\n=== FEATURES ===")
print("Feature columns:", len(feat.columns))
print("Dropped for insufficient history: %d (%.1f%%)" % (n_before - len(data), 100 * (n_before - len(data)) / n_before))
print("Modeling table rows:", len(data))

# Fig 6: correlation heatmap
fcols = list(feat.columns)
corr_df = data[fcols].copy()
corr_df["result_pts_diff"] = data.FTR.map({"H": 1, "D": 0, "A": -1})
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im); ax.set_title("Fig 6: Feature correlations (result: H=1, D=0, A=-1)")
plt.tight_layout(); plt.savefig("figures/fig6_feature_correlation.png", dpi=150); plt.close()
print("Correlation with result (H=1, D=0, A=-1):\n",
      corr["result_pts_diff"].drop("result_pts_diff").sort_values(ascending=False).round(3).to_string())

# ------------------------------------------------------------------ baselines
train = data[data.Season.isin(TRAIN_SEASONS)]
val = data[data.Season == VAL_SEASON]
print("\nTrain rows: %d | Validation rows: %d" % (len(train), len(val)))

# Row order matters for log_loss: columns are [H, D, A]
def evaluate(name, y_true, y_pred, proba=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    row = {"Model": name, "Accuracy": accuracy_score(y_true, y_pred),
           "LogLoss": log_loss(y_true, np.asarray(proba)[:, ::-1], labels=["A", "D", "H"]) if proba is not None else np.nan,
           "Draw_P": p[1], "Draw_R": r[1], "Draw_F1": f[1], "Macro_F1": f.mean(),
           "Home_F1": f[0], "Away_F1": f[2]}
    return row


yv = val.FTR.values
results, preds = [], {}

# 1) Always home win
p_prior = train.FTR.value_counts(normalize=True).reindex(CLASSES).values
pred = np.array(["H"] * len(val))
results.append(evaluate("Always home win", yv, pred, np.tile(p_prior, (len(val), 1))))
preds["Always home win"] = pred

# 2) Bookmaker: lowest odds
odds = val[["B365H", "B365D", "B365A"]].values
pred = np.array(CLASSES)[odds.argmin(axis=1)]
results.append(evaluate("Bookmaker (lowest odds)", yv, pred, val[["pH", "pD", "pA"]].values))
preds["Bookmaker (lowest odds)"] = pred

# 3) Logistic regression on form features
xcols = [c for c in feat.columns]
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(train[xcols], train.FTR)
order = [list(lr.classes_).index(c) for c in CLASSES]
proba = lr.predict_proba(val[xcols])[:, order]
pred = np.array(CLASSES)[proba.argmax(axis=1)]
results.append(evaluate("Logistic Regression", yv, pred, proba))
preds["Logistic Regression"] = pred

res = pd.DataFrame(results)
res.to_csv("results/baseline_results.csv", index=False)
print("\n=== BASELINE RESULTS (validation %s) ===" % VAL_SEASON)
print(res.round(3).to_string(index=False))
print("Predicted class counts:")
for k, v in preds.items():
    print(" ", k, pd.Series(v).value_counts().reindex(CLASSES, fill_value=0).to_dict())

# Fig 7: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, pr) in zip(axes, preds.items()):
    cm = confusion_matrix(yv, pr, labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
plt.suptitle("Fig 7: Confusion matrices, validation season (rows = actual)")
plt.tight_layout(); plt.savefig("figures/fig7_confusion_matrices.png", dpi=150); plt.close()

print("\nDone. Figures in ./figures, table in ./results/baseline_results.csv")
