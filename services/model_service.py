from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

def clean_dataset(df: pd.DataFrame):
    original_rows = len(df)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].astype("string").str.strip()
    before_missing = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    removed_missing = before_missing - len(df)
    before_duplicates = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_duplicates - len(df)
    df = df[np.isfinite(df[FEATURES]).all(axis=1)]
    df = df[df[TARGET].astype(str).str.len() > 0]
    report = {
        "original_rows": int(original_rows),
        "rows_after_cleaning": int(len(df)),
        "rows_removed_missing_or_invalid": int(removed_missing + (before_duplicates - removed_duplicates - len(df))),
        "duplicates_removed": int(removed_duplicates),
        "columns": list(df.columns),
        "class_count": int(df[TARGET].nunique())
    }
    return df.reset_index(drop=True), report

def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, criterion="entropy"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, class_weight=None),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=10, gamma="scale"))]),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "Extra Trees": ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    }

def train_and_compare(df, test_size=0.2, tune=False):
    df, cleaning = clean_dataset(df)
    X, y = df[FEATURES], df[TARGET].astype(str)
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    models = get_models()
    rows, fitted = [], {}
    cv_used = False
    for name, model in models.items():
        candidate = clone(model)
        if tune and name in ("Random Forest", "SVM (RBF)", "KNN"):
            cv_splits = min(3, int(y_train.value_counts().min()))
            if cv_splits >= 2:
                cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                if name == "Random Forest":
                    grid = {"n_estimators": [150, 250], "max_depth": [None, 20]}
                    candidate = GridSearchCV(candidate, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
                elif name == "SVM (RBF)":
                    grid = {"model__C": [1, 10], "model__gamma": ["scale", 0.1]}
                    candidate = GridSearchCV(candidate, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
                else:
                    grid = {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]}
                    candidate = GridSearchCV(candidate, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
                cv_used = True
        start = time.perf_counter()
        candidate.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        estimator = candidate.best_estimator_ if hasattr(candidate, "best_estimator_") else candidate
        start = time.perf_counter()
        pred = estimator.predict(X_test)
        predict_seconds = time.perf_counter() - start
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
            "Train time (s)": train_seconds,
            "Predict time (s)": predict_seconds,
            "Best parameters": json.dumps(candidate.best_params_, default=str) if hasattr(candidate, "best_params_") else "Default configuration"
        })
        fitted[name] = estimator
    metrics = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    from .features import build_insights  # practical-powered features (Find-S, ID3, KNN, LR, K-Means)
    insights = build_insights(X_train, y_train, X_test, y_test)
    return {
        "insights": insights, "X_train": X_train, "y_train": y_train,
        "cleaning": cleaning, "metrics": metrics, "models": fitted,
        "best_name": best_name, "best_model": fitted[best_name],
        "X_test": X_test, "y_test": y_test,
        "cv_used": cv_used, "classes": sorted(y.unique().tolist())
    }

def save_training_result(result, out_dir="artifacts"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["best_model"], out / "best_model.joblib")
    joblib.dump(result["models"], out / "all_models.joblib")
    result["metrics"].to_csv(out / "model_comparison.csv", index=False)
    meta = {
        "best_model": result["best_name"],
        "features": FEATURES,
        "classes": result["classes"],
        "cleaning": result["cleaning"],
        "tuning_used": result["cv_used"]
    }
    (out / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(out / "best_model.joblib")
