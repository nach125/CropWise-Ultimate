import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from services.model_service import clean_dataset, FEATURES, TARGET
from services.lab_algorithms import find_s

def test_clean_dataset_removes_duplicate():
    row = {"N":1,"P":2,"K":3,"temperature":20,"humidity":50,"ph":6,"rainfall":100,"label":"rice"}
    df = pd.DataFrame([row,row])
    clean, report = clean_dataset(df)
    assert len(clean) == 1
    assert report["duplicates_removed"] == 1

def test_find_s_generalizes_positive_examples():
    df = pd.DataFrame({"Sky":["Sunny","Sunny","Rainy"],"Air":["Warm","Warm","Cold"],"Enjoy":["Yes","Yes","No"]})
    assert find_s(df) == ["Sunny","Warm"]


# ---- practical-powered features -------------------------------------------------
from synthetic import make_crop_df
from services.model_service import train_and_compare
from services import features as fx

def _result():
    return train_and_compare(make_crop_df(40))

def test_all_features_built_without_errors():
    ins = _result()["insights"]
    assert ins["errors"] == {}
    for key in ["profiles", "id3", "knn", "regression", "zones"]:
        assert key in ins

def test_id3_is_literal_entropy_gain():
    import pandas as pd
    X = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]}); y = pd.Series(["1", "1", "0", "0"])
    t = fx.ID3Classifier(min_samples=1).fit(X, y)
    assert t.information_gain(X, y, "a") == 1.0 and t.tree_["feature"] == "a"
    assert list(t.predict(X)) == list(y)

def test_find_s_profile_and_suitability():
    r = _result(); ins = r["insights"]
    row = {"N": 80, "P": 48, "K": 40, "temperature": 24, "humidity": 82, "ph": 6.4, "rainfall": 230}
    ok = fx.suitability(ins["binner3"], ins["profiles"], "rice", row)
    bad = fx.suitability(ins["binner3"], ins["profiles"], "chickpea", row)
    assert ok["score"] == 1.0 and bad["score"] < ok["score"]

def test_estimate_missing_uses_only_known_inputs_and_stays_in_range():
    ins = _result()["insights"]
    known = {"N": 80, "P": 48, "K": 40, "temperature": 24, "humidity": 82, "ph": 6.4}
    est = fx.estimate_missing(ins["X_train"], ins["X_test"], known, ["rainfall"])
    assert "rainfall" not in est["rainfall"]["predictors"]
    assert ins["X_train"]["rainfall"].min() <= est["rainfall"]["value"] <= ins["X_train"]["rainfall"].max()

def test_knn_and_zone_lookups():
    ins = _result()["insights"]; row = {"N": 20, "P": 68, "K": 19, "temperature": 24, "humidity": 64, "ph": 6.9, "rainfall": 46}
    assert fx.knn_top_picks(ins["knn"], row).iloc[0]["Crop"] == "lentil"
    assert len(fx.similar_fields(ins["knn"], row, 5)) == 5
    assert 0 <= fx.assign_zone(ins["zones"], row)["zone"] < ins["zones"]["k"]
