"""Product features built on top of the ML practicals.

Each practical from the Algorithm Lab powers one user-facing feature:

  Find-S              -> Crop suitability profiles ("can I grow X here?")
  ID3 (hand-coded)    -> Explainable "why this crop?" decision rules
  KNN + GridSearchCV  -> Top-3 picks and "similar fields" lookup
  Linear Regression   -> Estimate an input the farmer does not know
  K-Means             -> Agro-climatic zones

All models are fitted on the training split only. Nothing here reads the test set
except to *report* held-out quality.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import lab_algorithms as lab

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

_LABELS = {
    1: ["Any"],
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very low", "Low", "High", "Very high"],
    5: ["Very low", "Low", "Medium", "High", "Very high"],
}


# --------------------------------------------------------------------------- binning
class QuantileBinner:
    """Turns numeric columns into ordered categories (Low/Medium/High ...).

    Find-S and ID3 work on categorical attributes, so numeric inputs are discretised
    using quantile edges learned from the training split only.
    """

    def __init__(self, n_bins: int = 3):
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame):
        self.edges_, self.labels_ = {}, {}
        for c in X.columns:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(X[c].to_numpy(float), qs))
            self.edges_[c] = edges
            self.labels_[c] = _LABELS[len(edges) + 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            idx = np.searchsorted(self.edges_[c], X[c].to_numpy(float), side="right")
            out[c] = [self.labels_[c][i] for i in idx]
        return pd.DataFrame(out, index=X.index)

    def describe(self, feature: str, label: str) -> str:
        """Human-readable numeric range for a bin label, e.g. 'High (> 120.0)'."""
        labels, edges = self.labels_[feature], self.edges_[feature]
        i = labels.index(label)
        lo = f"{edges[i - 1]:.1f}" if i > 0 else None
        hi = f"{edges[i]:.1f}" if i < len(edges) else None
        if lo and hi:
            return f"{label} ({lo}–{hi})"
        if hi:
            return f"{label} (< {hi})"
        if lo:
            return f"{label} (≥ {lo})"
        return label


# --------------------------------------------------------------------------- ID3
class ID3Classifier:
    """Hand-coded ID3: categorical attributes, entropy and information gain.

    Unlike the sklearn entropy tree in the benchmark this is a literal ID3: each
    attribute is used at most once per path and every attribute value gets a branch.
    """

    def __init__(self, max_depth: int = 6, min_samples: int = 5):
        self.max_depth, self.min_samples = max_depth, min_samples

    @staticmethod
    def entropy(y) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        return -sum((c / n) * math.log2(c / n) for c in Counter(y).values())

    def information_gain(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        base, n = self.entropy(y), len(y)
        rem = sum(len(g) / n * self.entropy(y.loc[g.index]) for _, g in X.groupby(feature))
        return base - rem

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features_ = list(X.columns)
        X, y = X.reset_index(drop=True), pd.Series(list(y)).reset_index(drop=True)
        self.tree_ = self._build(X, y, list(X.columns), 0)
        return self

    def _leaf(self, y):
        counts = Counter(y)
        label, _ = counts.most_common(1)[0]
        return {"leaf": True, "label": label, "n": len(y), "dist": dict(counts)}

    def _build(self, X, y, feats, depth):
        if y.nunique() == 1 or not feats or depth >= self.max_depth or len(y) < self.min_samples:
            return self._leaf(y)
        gains = {f: self.information_gain(X, y, f) for f in feats}
        best = max(gains, key=gains.get)
        if gains[best] <= 1e-12:
            return self._leaf(y)
        node = {"leaf": False, "feature": best, "gain": gains[best], "n": len(y),
                "majority": Counter(y).most_common(1)[0][0], "children": {}}
        rest = [f for f in feats if f != best]
        for val, g in X.groupby(best):
            node["children"][val] = self._build(g, y.loc[g.index], rest, depth + 1)
        return node

    def _walk(self, row: dict):
        node, path = self.tree_, []
        while not node["leaf"]:
            val = row[node["feature"]]
            path.append((node["feature"], val, node["gain"]))
            if val not in node["children"]:
                return {"leaf": True, "label": node["majority"], "n": node["n"],
                        "dist": {node["majority"]: node["n"]}}, path
            node = node["children"][val]
        return node, path

    def predict(self, X: pd.DataFrame):
        return np.array([self._walk(r)[0]["label"] for r in X.to_dict("records")])

    def explain(self, row: dict) -> dict:
        leaf, path = self._walk(row)
        n = max(sum(leaf["dist"].values()), 1)
        return {"prediction": leaf["label"], "path": path, "support": leaf["n"],
                "purity": leaf["dist"].get(leaf["label"], 0) / n}

    def to_text(self, binner: QuantileBinner | None = None) -> str:
        lines = []

        def rec(node, indent):
            pad = "│   " * indent
            if node["leaf"]:
                lines.append(f"{pad}└─ ➜ {node['label']}  (n={node['n']})")
                return
            for val, child in node["children"].items():
                shown = binner.describe(node["feature"], val) if binner else val
                lines.append(f"{pad}├─ {node['feature']} = {shown}")
                rec(child, indent + 1)

        rec(self.tree_, 0)
        return "\n".join(lines)


# --------------------------------------------------------------------------- Find-S
def find_s_profiles(binner: QuantileBinner, X_train: pd.DataFrame, y_train: pd.Series,
                    tolerance: float = 0.9) -> dict:
    """Per-crop concept learning with Find-S (crop = positive, all others = negative).

    Strict Find-S (the practical's algorithm) is very brittle on noisy agricultural
    data: a single unusual sample turns an attribute into '?'. So we also report a
    noise-tolerant variant that keeps a value if >= `tolerance` of positives share it.
    The tolerant variant is an extension, labelled as such in the UI.
    """
    Xb = binner.transform(X_train)
    y = pd.Series(list(y_train), index=Xb.index)
    out = {}
    for crop in sorted(y.unique()):
        ex = Xb.copy()
        ex["target"] = np.where(y == crop, "Yes", "No")
        strict = lab.find_s(ex)  # the practical's implementation, reused as-is
        pos = Xb[y == crop]
        tolerant = []
        for c in Xb.columns:
            val, cnt = Counter(pos[c]).most_common(1)[0]
            tolerant.append(val if cnt / len(pos) >= tolerance else "?")
        out[crop] = {"strict": dict(zip(Xb.columns, strict)),
                     "tolerant": dict(zip(Xb.columns, tolerant)), "n_positive": int(len(pos))}
    return out


def suitability(binner: QuantileBinner, profiles: dict, crop: str, values: dict,
                variant: str = "tolerant") -> dict:
    """Compare a field's conditions to a crop's Find-S hypothesis."""
    hyp = profiles[crop][variant]
    row = binner.transform(pd.DataFrame([values], columns=FEATURES)).iloc[0].to_dict()
    checks = []
    for f in FEATURES:
        required = hyp[f]
        if required == "?":
            continue
        checks.append({"feature": f, "needed": binner.describe(f, required),
                       "yours": row[f], "match": row[f] == required})
    matched = sum(c["match"] for c in checks)
    return {"crop": crop, "variant": variant, "score": matched / len(checks) if checks else None,
            "constrained": len(checks), "matched": matched, "checks": checks}


# --------------------------------------------------------------------------- KNN
def build_knn_finder(X_train, y_train, X_test, y_test) -> dict:
    """KNN with GridSearchCV (training-set CV only); powers top picks + similar fields."""
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = {"knn__n_neighbors": [3, 5, 7, 9, 11], "knn__weights": ["uniform", "distance"]}
    folds = min(5, int(pd.Series(list(y_train)).value_counts().min()))
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(X_train, y_train)
        model, params, cv_score = search.best_estimator_, search.best_params_, float(search.best_score_)
    else:  # too few samples per class to cross-validate
        model, params, cv_score = pipe.fit(X_train, y_train), {"note": "default; CV skipped"}, None
    pred = model.predict(X_test)
    return {"model": model, "best_params": params, "cv_f1_macro": cv_score,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "X_train": X_train.reset_index(drop=True),
            "y_train": pd.Series(list(y_train)).reset_index(drop=True)}


def knn_top_picks(knn: dict, values: dict, top: int = 3) -> pd.DataFrame:
    x = pd.DataFrame([values], columns=FEATURES)
    proba = knn["model"].predict_proba(x)[0]
    classes = knn["model"].classes_
    order = np.argsort(proba)[::-1][:top]
    return pd.DataFrame({"Crop": classes[order], "Neighbour vote share": proba[order]})


def similar_fields(knn: dict, values: dict, n: int = 5) -> pd.DataFrame:
    scaler, nn = knn["model"].named_steps["scaler"], knn["model"].named_steps["knn"]
    z = scaler.transform(pd.DataFrame([values], columns=FEATURES))
    dist, idx = nn.kneighbors(z, n_neighbors=n)
    out = knn["X_train"].iloc[idx[0]].copy()
    out.insert(0, "Crop", knn["y_train"].iloc[idx[0]].to_numpy())
    out.insert(1, "Distance (scaled)", dist[0])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- Linear Regression
def regression_report(X_train, X_test) -> pd.DataFrame:
    """How well can each input be estimated from the other six? (held-out metrics)"""
    rows = []
    for t in FEATURES:
        others = [f for f in FEATURES if f != t]
        m = LinearRegression().fit(X_train[others], X_train[t])
        p = m.predict(X_test[others])
        rows.append({"Estimated input": t, "R²": r2_score(X_test[t], p),
                     "MAE": mean_absolute_error(X_test[t], p),
                     "RMSE": float(np.sqrt(mean_squared_error(X_test[t], p)))})
    return pd.DataFrame(rows)


def estimate_missing(X_train, X_test, known: dict, missing: list[str]) -> dict:
    """Fill unknown inputs with Linear Regression fitted on the inputs the user *does* know."""
    predictors = [f for f in FEATURES if f in known and f not in missing]
    if not predictors:
        raise ValueError("Provide at least one known input to estimate the others.")
    row = pd.DataFrame([{f: known[f] for f in predictors}])
    out = {}
    for t in missing:
        m = LinearRegression().fit(X_train[predictors], X_train[t])
        p = m.predict(X_test[predictors])
        val = float(np.clip(m.predict(row)[0], X_train[t].min(), X_train[t].max()))
        out[t] = {"value": val, "R2": float(r2_score(X_test[t], p)),
                  "MAE": float(mean_absolute_error(X_test[t], p)), "predictors": predictors}
    return out


# --------------------------------------------------------------------------- K-Means
def build_zones(X_train, y_train, k: int | None = None, k_range=range(2, 9)) -> dict:
    """Agro-climatic zones: K-Means on standardised inputs (labels are NOT used to fit)."""
    scaler = StandardScaler().fit(X_train)
    Z = scaler.transform(X_train)
    sweep = []
    for kk in k_range:
        if kk >= len(Z):
            break
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(Z)
        sweep.append({"k": kk, "inertia": float(km.inertia_),
                      "silhouette": float(silhouette_score(Z, km.labels_))})
    sweep = pd.DataFrame(sweep)
    if k is None:
        k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    labels = model.labels_
    y = pd.Series(list(y_train))
    prof_rows = []
    for c in range(k):
        mask = labels == c
        crops = y[mask].value_counts()
        row = {"Zone": f"Zone {c + 1}", "Fields": int(mask.sum())}
        row.update({f: float(X_train[f].to_numpy()[mask].mean()) for f in FEATURES})
        row["Dominant crops"] = ", ".join(f"{n} ({v / mask.sum():.0%})" for n, v in crops.head(3).items())
        prof_rows.append(row)
    pca = PCA(n_components=2, random_state=42).fit(Z)
    return {"scaler": scaler, "model": model, "k": k, "sweep": sweep,
            "profiles": pd.DataFrame(prof_rows), "pca": pca, "points": pca.transform(Z),
            "labels": labels, "silhouette": float(silhouette_score(Z, labels))}


def assign_zone(zones: dict, values: dict) -> dict:
    z = zones["scaler"].transform(pd.DataFrame([values], columns=FEATURES))
    c = int(zones["model"].predict(z)[0])
    return {"zone": c, "name": f"Zone {c + 1}", "profile": zones["profiles"].iloc[c],
            "point": zones["pca"].transform(z)[0]}


# --------------------------------------------------------------------------- bundle
def build_insights(X_train, y_train, X_test, y_test) -> dict:
    """Build every practical-powered feature once, after training. Failures are isolated."""
    ins, errors = {"X_train": X_train.reset_index(drop=True), "X_test": X_test.reset_index(drop=True),
                   "y_train": pd.Series(list(y_train))}, {}

    def step(name, fn):
        try:
            ins[name] = fn()
        except Exception as e:  # keep training usable even if one feature cannot be built
            errors[name] = str(e)

    step("binner3", lambda: QuantileBinner(3).fit(X_train))
    step("profiles", lambda: find_s_profiles(ins["binner3"], X_train, y_train))
    step("binner5", lambda: QuantileBinner(5).fit(X_train))

    def _id3():
        b = ins["binner5"]
        tree = ID3Classifier(max_depth=6, min_samples=5).fit(b.transform(X_train), pd.Series(list(y_train)))
        pred = tree.predict(b.transform(X_test))
        return {"tree": tree, "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0))}

    step("id3", _id3)
    step("knn", lambda: build_knn_finder(X_train, y_train, X_test, y_test))
    step("regression", lambda: regression_report(X_train, X_test))
    step("zones", lambda: build_zones(X_train, y_train))
    ins["errors"] = errors
    return ins
