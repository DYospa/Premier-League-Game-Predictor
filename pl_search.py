"""
CSC 4740 Final Project: continuous feature + model search

Runs for a fixed time (default 2 hours), trying random and mutated
configurations of features and models, and keeps the best one.

  python pl_search.py --hours 2             # search (safe: never loads 2025-26)
  python pl_search.py --final               # ONE evaluation of the best config on 2025-26
  python pl_search.py --test-season 2627 --hours 1     # roll forward: 2025-26 becomes search data
  python pl_search.py --test-season 2627 --final       # one evaluation on the 2026-27 matches played so far

Why it is built this way
  * Search mode loads only 2016-17 .. 2024-25. The 2025-26 test season is not
    even read, so it cannot influence any choice.
  * Every trial is scored by expanding-window CV (validate on 2020-21 ... 2024-25),
    using log loss by default (--metric accuracy to optimize accuracy instead).
  * Trying thousands of configs means the best CV score is optimistic (it is the
    luckiest of many). The honest number is the --final test-season result.
  * --final writes results/.test_season_used and refuses to run again unless you
    pass --force, so the test season is only spent once.

Search space
  features: rolling window or exponential weighting, venue form on/off, head-to-head,
            Elo rating difference (K, home advantage, season carry-over), rest days
  draws:    optional draw features (|rating gap|, squared gap, scoring environment) and two
            draw-aware models: an ordered logit (Away < Draw < Home) and a two-stage model
            (P(draw), then P(home | no draw)). Draw skill is also logged as draw AUC, which
            does not depend on any decision threshold.
  models:   Logistic Regression, Ordered Logit, Two-stage Logit, Random Forest, k-NN, Gradient Boosting
  --with-odds adds bookmaker probabilities as input features. That is a different
  question ("can a model add to the market?"), so it logs to separate files.

Stop any time with Ctrl-C; progress is saved after every trial and the next run
resumes from results/search_log.csv.
"""

import argparse
import copy
import json
import os
import random
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALL_SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
FIRST_CV_VAL = 4
CLASSES = ["H", "D", "A"]
MIN_GAMES = 3
STATS = ["gf", "ga", "sot", "pts"]
WARMUP = 40

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--hours", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--metric", choices=["logloss", "accuracy"], default="logloss")
parser.add_argument("--with-odds", action="store_true")
parser.add_argument("--test-season", default="2526", choices=ALL_SEASONS[5:],
                    help="season held out for the one-shot final test (default 2526). Use 2627 to roll "
                         "forward once 2025-26 has been spent: seasons before it are then search data.")
parser.add_argument("--final", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

TEST_SEASON = args.test_season
SEASONS = ALL_SEASONS[:ALL_SEASONS.index(TEST_SEASON) + 1]   # everything up to and including the test season
SEARCH_SEASONS = SEASONS[:-1]
TAG = ("_odds" if args.with_odds else "") + ("" if TEST_SEASON == "2526" else f"_t{TEST_SEASON}")
LOG_PATH = f"results/search_log{TAG}.csv"
BEST_PATH = f"results/best_config{TAG}.json"
LOCK_PATH = f"results/.test_season_used{TAG}"
os.makedirs("results", exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)

if args.final and os.path.exists(LOCK_PATH) and not args.force:
    raise SystemExit(f"{LOCK_PATH} exists: the test season was already evaluated. "
                     "Use --force only if you accept that the result is no longer a clean test.")


# ------------------------------------------------- draw-aware estimators
class OrderedLogit(BaseEstimator, ClassifierMixin):
    """Ordinal logistic regression with the order Away < Draw < Home.

    One latent 'home strength' score z = x.beta and two thresholds t1 < t2:
        P(A) = sigmoid(t1 - z),  P(A or D) = sigmoid(t2 - z)
    so the draw probability is largest when z falls between the thresholds,
    i.e. when the teams look evenly matched. A plain multinomial model has to
    learn that shape from the data; this one has it built in.
    """

    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.array(["A", "D", "H"])
        yi = np.searchsorted(self.classes_, np.asarray(y))
        d = X.shape[1]
        freq = np.bincount(yi, minlength=3) / len(yi)
        t1_0 = np.log(freq[0] / (1 - freq[0]))
        t2_0 = np.log((freq[0] + freq[1]) / freq[2])

        def nll(w):
            beta, t1, t2 = w[:d], w[d], w[d] + np.exp(w[d + 1])
            z = X @ beta
            p = self._probs(z, t1, t2)[np.arange(len(yi)), yi]
            return -np.log(np.clip(p, 1e-12, 1)).sum() + 0.5 / self.C * beta @ beta

        w0 = np.concatenate([np.zeros(d), [t1_0, np.log(max(t2_0 - t1_0, 0.1))]])
        res = minimize(nll, w0, method="L-BFGS-B")
        self.beta_, self.t1_, self.t2_ = res.x[:d], res.x[d], res.x[d] + np.exp(res.x[d + 1])
        return self

    @staticmethod
    def _probs(z, t1, t2):
        s1, s2 = expit(t1 - z), expit(t2 - z)
        return np.column_stack([s1, s2 - s1, 1 - s2])

    def predict_proba(self, X):
        P = self._probs(np.asarray(X, float) @ self.beta_, self.t1_, self.t2_)
        P = np.clip(P, 1e-9, 1)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class HierLR(BaseEstimator, ClassifierMixin):
    """Two-stage model: P(draw) from one classifier, then P(home | not a draw) from another.
    P(D) = p_d,  P(H) = (1 - p_d) p_h,  P(A) = (1 - p_d)(1 - p_h).
    Each stage gets its own regularization, so the draw stage can be kept simple."""

    def __init__(self, C_draw=0.1, C_side=0.1):
        self.C_draw = C_draw
        self.C_side = C_side

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.array(["A", "D", "H"])
        is_d = y == "D"
        self.draw_ = LogisticRegression(C=self.C_draw, max_iter=2000).fit(X, is_d)
        self.side_ = LogisticRegression(C=self.C_side, max_iter=2000).fit(X[~is_d], y[~is_d] == "H")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        pd_ = self.draw_.predict_proba(X)[:, 1]
        ph = self.side_.predict_proba(X)[:, 1]
        return np.column_stack([(1 - pd_) * (1 - ph), pd_, (1 - pd_) * ph])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ------------------------------------------------------------------ data
def load_season(code):
    path = os.path.join(args.data_dir, f"E0_{code}.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", path)
    d = pd.read_csv(path, encoding="latin-1").dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    d["Season"] = code
    return d


LOAD = SEASONS if args.final else SEARCH_SEASONS          # search mode never loads the test season
raw = pd.concat([load_season(s) for s in LOAD], ignore_index=True)
keep = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HST", "AST",
        "B365H", "B365D", "B365A"]
DF = raw[[c for c in keep if c in raw.columns]].copy()
DF["Date"] = pd.to_datetime(DF["Date"], dayfirst=True, format="mixed")
DF = DF.sort_values("Date", kind="stable").reset_index(drop=True)
DF["match_id"] = DF.index
inv = 1.0 / DF[["B365H", "B365D", "B365A"]]
DF[["pH", "pD", "pA"]] = inv.div(inv.sum(axis=1), axis=0).values
print("Loaded seasons:", LOAD, "| matches:", len(DF), "| test season:", TEST_SEASON)

pts_h = np.where(DF.FTR == "H", 3, np.where(DF.FTR == "D", 1, 0))
pts_a = np.where(DF.FTR == "A", 3, np.where(DF.FTR == "D", 1, 0))
LONG = pd.concat([
    pd.DataFrame({"match_id": DF.match_id, "Date": DF.Date, "team": DF.HomeTeam, "venue": "H",
                  "gf": DF.FTHG, "ga": DF.FTAG, "sot": DF.HST, "pts": pts_h}),
    pd.DataFrame({"match_id": DF.match_id, "Date": DF.Date, "team": DF.AwayTeam, "venue": "A",
                  "gf": DF.FTAG, "ga": DF.FTHG, "sot": DF.AST, "pts": pts_a}),
]).sort_values(["team", "Date", "match_id"]).reset_index(drop=True)

CACHE = {}


# ---------------------------------------------------------- feature parts
def form_feats(window, span):
    """Home-minus-away form differences. window=0 with span>0 means exponential weighting."""
    key = ("form", window, span)
    if key not in CACHE:
        def roll(g):
            s = g[STATS].shift(1)
            return (s.ewm(span=span, min_periods=MIN_GAMES).mean() if span
                    else s.rolling(window, min_periods=MIN_GAMES).mean())
        a_form = LONG.groupby("team", group_keys=False).apply(roll, include_groups=False)
        a_form.columns = [f"{c}_all" for c in STATS]
        v_form = LONG.groupby(["team", "venue"], group_keys=False).apply(roll, include_groups=False)
        v_form.columns = [f"{c}_ven" for c in STATS]
        lg = pd.concat([LONG, a_form, v_form], axis=1)
        h = lg[lg.venue == "H"].set_index("match_id")
        a = lg[lg.venue == "A"].set_index("match_id")
        out = pd.DataFrame(index=h.index)
        for c in STATS:
            out[f"d_{c}_all"] = h[f"{c}_all"] - a[f"{c}_all"]
            out[f"d_{c}_ven"] = h[f"{c}_ven"] - a[f"{c}_ven"]
        for c in ("gf", "ga"):                                   # total scoring environment (for draws)
            out[f"s_{c}_all"] = h[f"{c}_all"] + a[f"{c}_all"]
        CACHE[key] = out
    return CACHE[key]


def h2h_feats():
    if "h2h" not in CACHE:
        hist, h2h, miss = {}, {}, {}
        for r in DF.itertuples():
            k = frozenset((r.HomeTeam, r.AwayTeam))
            prev = hist.get(k, [])[-3:]
            if prev:
                h2h[r.match_id] = np.mean([p for t, p in prev if t == r.HomeTeam] +
                                          [3 - p if p != 1 else 1 for t, p in prev if t != r.HomeTeam])
                miss[r.match_id] = 0
            else:
                h2h[r.match_id], miss[r.match_id] = 1.0, 1
            hist.setdefault(k, []).append((r.HomeTeam, 3 if r.FTR == "H" else (1 if r.FTR == "D" else 0)))
        CACHE["h2h"] = pd.DataFrame({"h2h_pts": pd.Series(h2h), "h2h_missing": pd.Series(miss)})
    return CACHE["h2h"]


def elo_feats(k, home_adv, regress):
    key = ("elo", k, home_adv, regress)
    if key not in CACHE:
        ratings, out, last = {}, {}, None
        for r in DF.itertuples():
            if r.Season != last:
                if last is not None and regress > 0:
                    for t in ratings:
                        ratings[t] = 1500 + (1 - regress) * (ratings[t] - 1500)
                last = r.Season
            rh, ra = ratings.get(r.HomeTeam, 1450.0), ratings.get(r.AwayTeam, 1450.0)
            d = rh + home_adv - ra
            out[r.match_id] = d                                  # rating BEFORE the match
            e = 1.0 / (1.0 + 10 ** (-d / 400.0))
            s = 1.0 if r.FTR == "H" else (0.5 if r.FTR == "D" else 0.0)
            ratings[r.HomeTeam], ratings[r.AwayTeam] = rh + k * (s - e), ra - k * (s - e)
        CACHE[key] = pd.DataFrame({"elo_diff": pd.Series(out)})
    return CACHE[key]


def rest_feats():
    if "rest" not in CACHE:
        rest = LONG.groupby("team")["Date"].diff().dt.days.clip(upper=14).fillna(7)
        lg = LONG.assign(rest=rest.values)
        h = lg[lg.venue == "H"].set_index("match_id")["rest"]
        a = lg[lg.venue == "A"].set_index("match_id")["rest"]
        CACHE["rest"] = pd.DataFrame({"d_rest": h - a})
    return CACHE["rest"]


# Fixed row set for every config, so scores are comparable across trials.
_ref = form_feats(5, 0).dropna()
VALID_IDS = _ref.index.sort_values()
BASE = DF.set_index("match_id").loc[VALID_IDS]
SEASON_ARR, Y = BASE.Season.values, BASE.FTR.values
print("Rows used for modeling:", len(BASE))


def assemble(cfg):
    f = cfg["features"]
    span = f["ewm_span"]
    form = form_feats(0 if span else f["window"], span)
    parts = [form[[c for c in form.columns
                   if c.startswith("d_") and (f["venue"] or not c.endswith("_ven"))]]]
    if f["h2h"]:
        parts.append(h2h_feats())
    elo = elo_feats(f["elo_k"], f["elo_home"], f["elo_regress"]) if f["elo"] else None
    if elo is not None:
        parts.append(elo)
    if f.get("draw_feats", False):
        # Draws happen between evenly matched, low-scoring teams: give the model
        # closeness (absolute / squared differences) and scoring-environment features.
        dfx = pd.DataFrame(index=form.index)
        dfx["abs_pts"] = form["d_pts_all"].abs()
        dfx["abs_gd"] = (form["d_gf_all"] - form["d_ga_all"]).abs()
        dfx["tot_gf"], dfx["tot_ga"] = form["s_gf_all"], form["s_ga_all"]
        if elo is not None:
            dfx["abs_elo"] = elo["elo_diff"].abs()
            dfx["sq_elo"] = (elo["elo_diff"] / 100.0) ** 2
        parts.append(dfx)
    if f["rest"]:
        parts.append(rest_feats())
    if args.with_odds:
        parts.append(DF.set_index("match_id")[["pH", "pD", "pA"]])
    return pd.concat(parts, axis=1).loc[VALID_IDS].fillna(0.0)


# ----------------------------------------------------------- search space
FEATURE_SPACE = {
    "window": [3, 5, 8, 10], "ewm_span": [0, 0, 4, 6, 10], "venue": [True, False],
    "h2h": [True, False], "elo": [False, True, True], "elo_k": [10, 20, 30, 40],
    "elo_home": [40, 70, 100], "elo_regress": [0.0, 0.25, 0.5], "rest": [True, False],
    "draw_feats": [True, True, False],
}
PARAM_SPACE = {
    "lr": {"C": lambda r: 10 ** r.uniform(-3, 1)},
    "ord": {"C": lambda r: 10 ** r.uniform(-3, 1)},
    "hier": {"C_draw": lambda r: 10 ** r.uniform(-3, 1), "C_side": lambda r: 10 ** r.uniform(-3, 1)},
    "rf": {"n_estimators": lambda r: r.choice([200, 400]),
           "max_depth": lambda r: r.choice([3, 4, 5, 6, 8, None]),
           "min_samples_leaf": lambda r: r.choice([5, 10, 20, 40, 80]),
           "max_features": lambda r: r.choice(["sqrt", 0.3, 0.6, 1.0])},
    "knn": {"n_neighbors": lambda r: r.randint(15, 150), "weights": lambda r: r.choice(["uniform", "distance"])},
    "hgb": {"learning_rate": lambda r: 10 ** r.uniform(-2, -0.8), "max_depth": lambda r: r.choice([2, 3, 4]),
            "max_iter": lambda r: r.randint(40, 300), "min_samples_leaf": lambda r: r.choice([20, 40, 80, 160]),
            "l2_regularization": lambda r: 10 ** r.uniform(-3, 1)},
}


def sample_params(model, r):
    return {k: fn(r) for k, fn in PARAM_SPACE[model].items()}


def sample_config(r, base=None, p_mut=0.25):
    if base is None:
        model = r.choice(list(PARAM_SPACE))
        return {"features": {k: r.choice(v) for k, v in FEATURE_SPACE.items()},
                "model": model, "params": sample_params(model, r)}
    cfg = copy.deepcopy(base)
    for k, v in FEATURE_SPACE.items():
        if r.random() < p_mut:
            cfg["features"][k] = r.choice(v)
    if r.random() < p_mut:                                        # switch model family
        cfg["model"] = r.choice(list(PARAM_SPACE))
        cfg["params"] = sample_params(cfg["model"], r)
    else:                                                         # tweak one hyperparameter
        k = r.choice(list(PARAM_SPACE[cfg["model"]]))
        cfg["params"][k] = PARAM_SPACE[cfg["model"]][k](r)
    return cfg


def make_model(cfg):
    m, p = cfg["model"], cfg["params"]
    if m == "lr":
        return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=p["C"], max_iter=2000))])
    if m == "ord":
        return Pipeline([("s", StandardScaler()), ("c", OrderedLogit(C=p["C"]))])
    if m == "hier":
        return Pipeline([("s", StandardScaler()), ("c", HierLR(**p))])
    if m == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **p)
    if m == "knn":
        return Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(**p))])
    return HistGradientBoostingClassifier(random_state=42, **p)


def cv_score(cfg):
    X = assemble(cfg).values
    losses, accs, aucs = [], [], []
    for i in range(FIRST_CV_VAL, len(SEARCH_SEASONS)):
        tr = np.isin(SEASON_ARR, SEARCH_SEASONS[:i])
        va = SEASON_ARR == SEARCH_SEASONS[i]
        m = make_model(cfg).fit(X[tr], Y[tr])
        P = m.predict_proba(X[va])
        losses.append(log_loss(Y[va], P, labels=m.classes_))
        accs.append(accuracy_score(Y[va], m.classes_[P.argmax(axis=1)]))
        aucs.append(roc_auc_score(Y[va] == "D", P[:, list(m.classes_).index("D")]))
    return float(np.mean(losses)), float(np.std(losses)), float(np.mean(accs)), float(np.mean(aucs))


def score_of(ll, acc):
    return ll if args.metric == "logloss" else 1 - acc


def ckey(cfg):
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------ search mode
def search():
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    r = random.Random(seed)
    print(f"Seed {seed} | metric {args.metric} | with odds: {args.with_odds} | {args.hours} h")
    cols = ["trial", "elapsed_min", "cv_logloss", "cv_logloss_std", "cv_acc", "cv_draw_auc", "score", "config"]
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        if "cv_draw_auc" not in log.columns:                       # log from an older version
            log["cv_draw_auc"] = np.nan
        log = log[cols]
        log.to_csv(LOG_PATH, index=False)
    else:
        log = pd.DataFrame(columns=cols)
    seen = set(log.config)
    trials = log.to_dict("records")
    best = log.sort_values("score").iloc[0] if len(log) else None
    print(f"Resuming with {len(log)} earlier trials." if len(log) else "Starting fresh.")
    start, end = time.time(), time.time() + args.hours * 3600
    n = len(log)
    try:
        while time.time() < end:
            top = sorted(trials, key=lambda t: t["score"])[:5]
            if n < WARMUP or not top or r.random() < 0.3:
                cfg = sample_config(r)
            else:
                cfg = sample_config(r, json.loads(r.choice(top)["config"]))
            if ckey(cfg) in seen:
                continue
            seen.add(ckey(cfg))
            try:
                ll, sd, acc, auc = cv_score(cfg)
            except Exception as e:                                 # a bad combo should not stop the run
                print("  trial failed:", repr(e)[:100])
                continue
            n += 1
            row = {"trial": n, "elapsed_min": round((time.time() - start) / 60, 2), "cv_logloss": ll,
                   "cv_logloss_std": sd, "cv_acc": acc, "cv_draw_auc": auc, "score": score_of(ll, acc),
                   "config": ckey(cfg)}
            pd.DataFrame([row])[cols].to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
            trials.append(row)
            if best is None or row["score"] < best["score"]:
                best = pd.Series(row)
                json.dump({"config": cfg, "cv_logloss": ll, "cv_acc": acc, "cv_draw_auc": auc, "trial": n,
                           "with_odds": args.with_odds}, open(BEST_PATH, "w"), indent=2)
                print(f"[trial {n:5d} | {row['elapsed_min']:6.1f} min] NEW BEST  cv log loss {ll:.4f} "
                      f"(std {sd:.4f})  cv acc {acc:.4f}  draw AUC {auc:.3f}  {cfg['model']}")
            elif n % 25 == 0:
                print(f"[trial {n:5d} | {row['elapsed_min']:6.1f} min] best so far: "
                      f"log loss {best['cv_logloss']:.4f}, acc {best['cv_acc']:.4f}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    final_log = pd.read_csv(LOG_PATH)
    print("\n=== TOP 10 CONFIGS (CV on seasons before 2025-26) ===")
    for _, t in final_log.sort_values("score").head(10).iterrows():
        c = json.loads(t.config)
        print(f"log loss {t.cv_logloss:.4f} | acc {t.cv_acc:.4f} | draw AUC {t.cv_draw_auc:.3f} | {c['model']} {c['params']} | {c['features']}")
    print(f"\nBest config saved to {BEST_PATH}. When you are done searching, run: python pl_search.py --final"
          + (" --with-odds" if args.with_odds else ""))


# ------------------------------------------------------------- final mode
W_GRID = np.round(np.arange(1.0, 3.001, 0.05), 2)


def oof_proba(cfg):
    """Out-of-fold probabilities on the search seasons (never touches the test season)."""
    X = assemble(cfg).values
    P = np.full((len(Y), 3), np.nan)
    for i in range(FIRST_CV_VAL, len(SEARCH_SEASONS)):
        tr = np.isin(SEASON_ARR, SEARCH_SEASONS[:i])
        va = SEASON_ARR == SEARCH_SEASONS[i]
        m = make_model(cfg).fit(X[tr], Y[tr])
        P[va] = m.predict_proba(X[va])[:, [list(m.classes_).index(c) for c in CLASSES]]
    return P


def boost(P, w):
    return np.array(CLASSES)[(P * np.array([1.0, w, 1.0])).argmax(axis=1)]


def final():
    best = json.load(open(BEST_PATH))
    ref = {"features": {"window": 5, "ewm_span": 0, "venue": True, "h2h": True, "elo": False, "elo_k": 20,
                        "elo_home": 70, "elo_regress": 0.25, "rest": False},
           "model": "lr", "params": {"C": 0.01}}
    tr = np.isin(SEASON_ARR, SEARCH_SEASONS)
    te = SEASON_ARR == TEST_SEASON
    yte = Y[te]
    n = len(yte)
    print(f"Final evaluation: train rows {tr.sum()}, test rows {n} (season {TEST_SEASON})")
    preds, probas = {}, {}
    prior = pd.Series(Y[tr]).value_counts(normalize=True).reindex(CLASSES).values

    def add(name, pred, P):
        preds[name], probas[name] = pred, np.asarray(P)

    add("Always home win", np.array(["H"] * n), np.tile(prior, (n, 1)))
    bk = BASE[te][["B365H", "B365D", "B365A"]].values
    add("Bookmaker (lowest odds)", np.array(CLASSES)[bk.argmin(axis=1)], BASE[te][["pH", "pD", "pA"]].values)
    for name, cfg in [("Reference: Logistic Regression", ref), ("Search-best config", best["config"])]:
        X = assemble(cfg).values
        m = make_model(cfg).fit(X[tr], Y[tr])
        P = m.predict_proba(X[te])[:, [list(m.classes_).index(c) for c in CLASSES]]
        add(name, np.array(CLASSES)[P.argmax(axis=1)], P)
        if name == "Search-best config":
            oof = oof_proba(cfg)
            ok = ~np.isnan(oof).any(axis=1)
            curve = [precision_recall_fscore_support(Y[ok], boost(oof[ok], w), labels=CLASSES,
                                                     zero_division=0)[2].mean() for w in W_GRID]
            w_best = float(W_GRID[int(np.argmax(curve))])          # tuned on CV seasons only
            add(f"Search-best + draw boost (x{w_best:.2f})", boost(P, w_best), P)

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(2000, n))
    yi = np.array([CLASSES.index(c) for c in yte])
    loss = {k: -np.log(np.clip(p[np.arange(n), yi], 1e-15, 1)) for k, p in probas.items()}
    correct = {k: (v == yte).astype(float) for k, v in preds.items()}
    bkn = "Bookmaker (lowest odds)"
    rows = []
    for k in preds:
        p_, r_, f_, _ = precision_recall_fscore_support(yte, preds[k], labels=CLASSES, zero_division=0)
        b = int(((correct[bkn] == 1) & (correct[k] == 0)).sum())
        c = int(((correct[k] == 1) & (correct[bkn] == 0)).sum())
        d = loss[k][idx].mean(axis=1) - loss[bkn][idx].mean(axis=1)
        acc_b = correct[k][idx].mean(axis=1)
        rows.append({"Model": k, "Accuracy": correct[k].mean(), "Acc_CI_low": np.percentile(acc_b, 2.5),
                     "Acc_CI_high": np.percentile(acc_b, 97.5), "LogLoss": loss[k].mean(),
                     "Macro_F1": f_.mean(), "Draw_P": p_[1], "Draw_R": r_[1], "Draw_F1": f_[1],
                     "Draw_Lift": p_[1] / (yte == "D").mean(),
                     "Draw_AUC": roc_auc_score(yte == "D", probas[k][:, 1]),
                     "McNemar_p_vs_bookmaker": 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue,
                     "LogLoss_diff_vs_bookmaker": float(loss[k].mean() - loss[bkn].mean()),
                     "LL_diff_CI_low": np.percentile(d, 2.5), "LL_diff_CI_high": np.percentile(d, 97.5)})
    res = pd.DataFrame(rows)
    res.to_csv(f"results/final_best_model{TAG}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("\nSearch-best config:", json.dumps(best["config"]))
    print(f"Its CV log loss was {best['cv_logloss']:.4f}. The test log loss above can differ from it because "
          "the test season may be easier or harder than the CV seasons (compare with the bookmaker row) and "
          "because the best of many trials is slightly optimistic. Report the test number.")
    open(LOCK_PATH, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


final() if args.final else search()
