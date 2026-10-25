"""Teaching implementations for the requested ML practicals.
These are educational modules and are kept separate from the crop recommendation benchmark.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.datasets import load_iris

def find_s(examples):
    """Find-S over a dataframe of positive/negative examples; last column is target."""
    if examples.shape[1] < 2:
        raise ValueError("Provide at least one feature and a target column.")
    X = examples.iloc[:, :-1].astype(str).to_numpy()
    y = examples.iloc[:, -1].astype(str).str.lower().to_numpy()
    positives = X[np.isin(y, ["yes", "positive", "1", "true"])]
    if len(positives) == 0:
        return ["No positive examples found"] * X.shape[1]
    hypothesis = positives[0].tolist()
    for row in positives[1:]:
        for i, val in enumerate(row):
            if hypothesis[i] != val:
                hypothesis[i] = "?"
    return hypothesis

def linear_regression_demo(df=None, target="rainfall"):
    """Fits a transparent regression example; with no CSV uses a deterministic synthetic demo."""
    if df is not None and target in df.columns:
        numeric = df.select_dtypes(include="number").dropna()
        if target in numeric and len(numeric) >= 10:
            X = numeric.drop(columns=[target])
            y = numeric[target]
        else:
            X, y = synthetic_regression_data()
    else:
        X, y = synthetic_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"model": model, "actual": y_test, "predicted": pred, "MAE": mean_absolute_error(y_test,pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test,pred))), "R2": r2_score(y_test,pred),
            "feature_names": list(X.columns)}

def synthetic_regression_data():
    rng = np.random.default_rng(42)
    x = np.linspace(1, 100, 150)
    df = pd.DataFrame({"input_x": x, "input_x_squared": x*x/100})
    y = 2.8*x + rng.normal(0, 12, len(x))
    return df, pd.Series(y, name="target_y")

def knn_tuning_demo():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=.2, random_state=42, stratify=data.target)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    search = GridSearchCV(pipe, {"knn__n_neighbors":[3,5,7,9], "knn__weights":["uniform","distance"]}, cv=5, scoring="accuracy")
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    return {"best_params": search.best_params_, "cv_score": search.best_score_, "test_accuracy": accuracy_score(y_test,pred),
            "test_f1_macro": f1_score(y_test,pred,average="macro"), "search": search}

def id3_fit(X, y):
    """ID3-style tree using entropy/information gain; sklearn's entropy criterion."""
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X, y)
    return model

def kmeans_demo(df=None):
    if df is not None:
        numeric = df.select_dtypes(include="number").dropna()
        if numeric.shape[0] >= 10 and numeric.shape[1] >= 2:
            X = numeric.iloc[:, :min(4, numeric.shape[1])]
            names = list(X.columns)
        else:
            X, names = synthetic_cluster_data()
    else:
        X, names = synthetic_cluster_data()
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
    out = pd.DataFrame({"x": Z[:,0], "y": Z[:,1], "cluster": model.labels_.astype(str)})
    return {"model":model, "scaled":Z, "plot_data":out, "feature_names":names}

def synthetic_cluster_data():
    rng = np.random.default_rng(42)
    centers = np.array([[0,0],[3,3],[-3,3]])
    X = np.vstack([c + rng.normal(0,.55,(60,2)) for c in centers])
    return pd.DataFrame(X, columns=["feature_1","feature_2"]), ["feature_1","feature_2"]
