"""
CSC 4740 Final Project: Premier League match outcome prediction

Runs the final experiments:
  * rebuilds the cleaned dataset and pre-match features (same as pl_midreport.py)
  * tunes Logistic Regression, Decision Tree, Random Forest, Naive Bayes and
    k-NN with expanding-window (season by season) cross validation on 2016-17
    through 2024-25, scored by log loss
  * tunes a draw-probability multiplier on out-of-fold predictions
  * evaluates every model ONCE on the locked 2025-26 test season and compares
    against the "always home" and bookmaker baselines

Usage:
    python pl_final.py [--data-dir data]

Outputs: figures/fig8..fig12, results/final_results.csv, results/cv_results.csv,
results/final_results.md (and the results table inside README.md if the
<!-- RESULTS:START --> / <!-- RESULTS:END --> markers are present).

Model selection uses only the training seasons. The test season is touched
once, at the end, for the numbers that are reported.
"""

import argparse
import os
import re
import urllib.request

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TRAIN_SEASONS = SEASONS[:-1]          # 2016-17 .. 2024-25
TEST_SEASON = SEASONS[-1]             # 2025-26 (locked until the final evaluation)
FIRST_CV_VAL = 4                      # first CV validation season = TRAIN_SEASONS[4] (2020-21)
CLASSES = ["H", "D", "A"]
WINDOW, MIN_GAMES = 5, 3
SEED = 42
N_BOOT = 2000

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
args = parser.parse_args()
DATA_DIR = args.data_dir

for d in (DATA_DIR, "figures", "results"):
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ data
def load_season(code):
    path = os.path.join(DATA_DIR, f"E0_{code}.csv")
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
df = df.sort_values("Date", kind="stable").reset_index(drop=True)
df["match_id"] = df.index

inv = 1.0 / df[["B365H", "B365D", "B365A"]]
df[["pH", "pD", "pA"]] = inv.div(inv.sum(axis=1), axis=0).values

print("=== DATA ===")
print("Matches:", len(df), "| per season:", df.groupby("Season").size().to_dict())


# -------------------------------------------------------------- features
def build_features(matches):
    """Rolling pre-match features. Every rolling stat is shifted by one match,
    so a match never sees its own result or anything after it."""
    m = matches.sort_values("Date", kind="stable").reset_index(drop=True)
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
        feat[f"d_{c}_ven"] = h[f"{c}_ven"] - a[f"{c}_ven"]

    # head-to-head: home team's average points over the last 3 meetings (any venue)
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


# Features are causal, so building them over all seasons is leakage free: the
# first 2025-26 matches just need 2024-25 history to have form windows.
feat = build_features(df)
XCOLS = list(feat.columns)
data = df.set_index("match_id").join(feat)
data = data.dropna(subset=[c for c in XCOLS if c != "h2h_missing"]).sort_values("Date", kind="stable")
data = data.reset_index(drop=True)

train = data[data.Season.isin(TRAIN_SEASONS)].reset_index(drop=True)
test = data[data.Season == TEST_SEASON].reset_index(drop=True)
print("Train rows: %d (seasons %s-%s) | Test rows: %d (season %s)"
      % (len(train), TRAIN_SEASONS[0], TRAIN_SEASONS[-1], len(test), TEST_SEASON))
assert train.Date.max() < test.Date.min(), "train must be strictly before test"

Xtr, ytr = train[XCOLS], train.FTR
Xte, yte = test[XCOLS], test.FTR.values


# ------------------------------------------------- expanding-window CV
def make_folds(frame):
    folds = []
    for i in range(FIRST_CV_VAL, len(TRAIN_SEASONS)):
        tr = np.where(frame.Season.isin(TRAIN_SEASONS[:i]))[0]
        va = np.where(frame.Season == TRAIN_SEASONS[i])[0]
        folds.append((tr, va))
    return folds


folds = make_folds(train)
print("CV folds (validate on):", TRAIN_SEASONS[FIRST_CV_VAL:])


def reorder(proba, classes):
    return proba[:, [list(classes).index(c) for c in CLASSES]]


def oof_proba(est, X, y):
    P = np.full((len(X), 3), np.nan)
    for tr, va in folds:
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        P[va] = reorder(m.predict_proba(X.iloc[va]), m.classes_)
    return P


MODELS = {
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [20, 50, 100]}),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [10, 25, 50]}),
    "Naive Bayes": (
        Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]}),
    "k-NN": (
        Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [15, 25, 50, 75, 100], "clf__weights": ["uniform", "distance"]}),
}
BALANCED_KEY = {"Logistic Regression": "clf__class_weight", "Random Forest": "class_weight"}

W_GRID = np.round(np.arange(1.0, 2.001, 0.05), 2)


def boost_draw(P, w):
    return np.array(CLASSES)[(P * np.array([1.0, w, 1.0])).argmax(axis=1)]


def macro_f1(y, pred):
    return precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)[2].mean()


print("\n=== TUNING (expanding-window CV, log loss) ===")
fitted, cv_rows, draw_w, oof_curves = {}, [], {}, {}
for name, (est, grid) in MODELS.items():
    gs = GridSearchCV(est, grid, scoring="neg_log_loss", cv=folds, n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    fitted[name] = gs.best_estimator_
    P = oof_proba(gs.best_estimator_, Xtr, ytr)
    ok = ~np.isnan(P).any(axis=1)
    curve = [macro_f1(ytr.values[ok], boost_draw(P[ok], w)) for w in W_GRID]
    draw_w[name] = float(W_GRID[int(np.argmax(curve))])   # first max -> smallest w on ties
    oof_curves[name] = curve
    cv_rows.append({"Model": name, "BestParams": str(gs.best_params_),
                    "CV_LogLoss": -gs.best_score_,
                    "CV_LogLoss_std": gs.cv_results_["std_test_score"][gs.best_index_],
                    "DrawMultiplier": draw_w[name],
                    "OOF_MacroF1_argmax": curve[0], "OOF_MacroF1_boosted": max(curve)})
    print("%-20s CV log loss %.4f | best %s | draw multiplier %.2f"
          % (name, -gs.best_score_, gs.best_params_, draw_w[name]))

cv = pd.DataFrame(cv_rows)
cv.to_csv("results/cv_results.csv", index=False)
headline = cv.sort_values("CV_LogLoss").iloc[0]["Model"]     # chosen WITHOUT looking at test
print("Model with lowest CV log loss (headline model):", headline)

# class-weighted variants of the best parameters (a second way to help draws)
for name, key in BALANCED_KEY.items():
    est = clone(fitted[name]).set_params(**{key: "balanced"})
    fitted[f"{name} (balanced)"] = est.fit(Xtr, ytr)


# ------------------------------------------------- final test evaluation
def metrics(y, pred, proba):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {"Accuracy": accuracy_score(y, pred),
            "LogLoss": log_loss(y, np.asarray(proba)[:, ::-1], labels=["A", "D", "H"]),
            "Draw_P": p[1], "Draw_R": r[1], "Draw_F1": f[1], "Macro_F1": f.mean(),
            "Home_F1": f[0], "Away_F1": f[2]}


rows, preds, probas = [], {}, {}


def add(name, pred, proba):
    preds[name], probas[name] = pred, proba
    rows.append({"Model": name, **metrics(yte, pred, proba)})


prior = ytr.value_counts(normalize=True).reindex(CLASSES).values
add("Always home win", np.array(["H"] * len(test)), np.tile(prior, (len(test), 1)))
bk_proba = test[["pH", "pD", "pA"]].values
add("Bookmaker (lowest odds)", np.array(CLASSES)[test[["B365H", "B365D", "B365A"]].values.argmin(axis=1)], bk_proba)

for name in MODELS:
    P = reorder(fitted[name].predict_proba(Xte), fitted[name].classes_)
    add(name, np.array(CLASSES)[P.argmax(axis=1)], P)
    add(f"{name} + draw boost (x{draw_w[name]:.2f})", boost_draw(P, draw_w[name]), P)
    if name in BALANCED_KEY:
        Pb = reorder(fitted[f"{name} (balanced)"].predict_proba(Xte), fitted[f"{name} (balanced)"].classes_)
        add(f"{name} (balanced)", np.array(CLASSES)[Pb.argmax(axis=1)], Pb)

res = pd.DataFrame(rows)

# bootstrap CI for accuracy and paired comparison with the bookmaker
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(test), size=(N_BOOT, len(test)))
correct = {k: (v == yte).astype(float) for k, v in preds.items()}
bk_boot = correct["Bookmaker (lowest odds)"][idx].mean(axis=1)
lo, hi, beat = [], [], []
for m_ in res.Model:
    b = correct[m_][idx].mean(axis=1)
    lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    beat.append((b > bk_boot).mean())
res["Acc_CI_low"], res["Acc_CI_high"], res["P_beats_bookmaker"] = lo, hi, beat
res.to_csv("results/final_results.csv", index=False)

print("\n=== FINAL TEST RESULTS (season %s, n=%d) ===" % (TEST_SEASON, len(test)))
print(res.round(3).to_string(index=False))
print("Predicted class counts:")
for k, v in preds.items():
    print("  %-40s %s" % (k, pd.Series(v).value_counts().reindex(CLASSES, fill_value=0).to_dict()))


# ------------------------------------------------------------- figures
BASE = ["Bookmaker (lowest odds)"] + list(MODELS)

# Fig 8: confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, name in zip(axes.ravel(), BASE):
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[name], labels=CLASSES),
                           display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
plt.suptitle(f"Fig 8: Confusion matrices, test season {TEST_SEASON} (rows = actual)")
plt.tight_layout(); plt.savefig("figures/fig8_test_confusion_matrices.png", dpi=150); plt.close()

# Fig 9: accuracy (with CI) and log loss
sel = res[res.Model.isin(["Always home win"] + BASE)].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
err = [sel.Accuracy - sel.Acc_CI_low, sel.Acc_CI_high - sel.Accuracy]
axes[0].barh(sel.Model, sel.Accuracy, xerr=err, color="#2a9d8f", capsize=3)
axes[0].set_xlabel("Accuracy (95% bootstrap CI)"); axes[0].invert_yaxis()
axes[1].barh(sel.Model, sel.LogLoss, color="#e76f51")
axes[1].set_xlabel("Log loss (lower is better)"); axes[1].invert_yaxis()
axes[1].set_yticklabels([])
axes[1].set_xlim(left=max(0, sel.LogLoss.min() - 0.1))
plt.suptitle(f"Fig 9: Test season {TEST_SEASON}")
plt.tight_layout(); plt.savefig("figures/fig9_test_accuracy_logloss.png", dpi=150); plt.close()

# Fig 10: draw precision / recall / F1, plain vs draw boost
names = [n for n in res.Model if n in BASE or "draw boost" in n]
sub = res.set_index("Model").loc[names, ["Draw_P", "Draw_R", "Draw_F1"]]
fig, ax = plt.subplots(figsize=(11, 4.5))
sub.plot.bar(ax=ax, color=["#e9c46a", "#2a9d8f", "#264653"])
ax.set_ylabel("Score"); ax.set_title(f"Fig 10: Draw prediction quality, test {TEST_SEASON}")
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig10_draw_metrics.png", dpi=150); plt.close()

# Fig 11: random forest feature importance
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=XCOLS).sort_values()
fig, ax = plt.subplots(figsize=(6, 6))
imp.plot.barh(ax=ax, color="#2a9d8f")
ax.set_title("Fig 11: Random Forest feature importance (train seasons)")
plt.tight_layout(); plt.savefig("figures/fig11_rf_feature_importance.png", dpi=150); plt.close()

# Fig 12: draw multiplier tuning curves (out-of-fold, training seasons only)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, curve in oof_curves.items():
    ax.plot(W_GRID, curve, marker=".", label=name)
ax.set_xlabel("Draw probability multiplier"); ax.set_ylabel("Out-of-fold macro F1")
ax.set_title("Fig 12: Tuning the draw threshold"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig12_draw_multiplier_tuning.png", dpi=150); plt.close()


# ------------------------------------------------ results table for README
def md_table(frame, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    body = ""
    for _, r in frame.iterrows():
        body += "| " + " | ".join(r[c] if isinstance(r[c], str) else f"{r[c]:.3f}" for c in cols) + " |\n"
    return head + body


table = md_table(res, ["Model", "Accuracy", "LogLoss", "Draw_P", "Draw_R", "Draw_F1", "Macro_F1"])
cv_table = md_table(cv, ["Model", "CV_LogLoss", "DrawMultiplier"])
md = (f"Test season {TEST_SEASON} (n = {len(test)} matches after dropping matches with too little history).\n\n"
      f"{table}\n"
      f"Cross-validated tuning on 2016-17 to 2024-25 (expanding window, validated on "
      f"{TRAIN_SEASONS[FIRST_CV_VAL]} to {TRAIN_SEASONS[-1]}):\n\n{cv_table}")
with open("results/final_results.md", "w") as fh:
    fh.write(md)

if os.path.exists("README.md"):
    txt = open("README.md", encoding="utf-8").read()
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
    if pat.search(txt):
        open("README.md", "w", encoding="utf-8").write(pat.sub(lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", txt))
        print("README.md results section updated.")

print("\nDone. Figures in ./figures, tables in ./results")
