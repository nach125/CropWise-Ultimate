from pathlib import Path
import json, sys, time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import database
from services.model_service import FEATURES, TARGET, clean_dataset, train_and_compare, save_training_result
from services import lab_algorithms as lab
from services import features as fx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CropWise Ultimate", page_icon="🌱", layout="wide")
database.init_db()
if "training" not in st.session_state: st.session_state.training = None
if "source_df" not in st.session_state: st.session_state.source_df = None

st.sidebar.title("🌱 CropWise Ultimate")
page = st.sidebar.radio("Navigate", ["Dashboard", "Recommend Crop", "Field Insights", "Model Training & SOTA", "Algorithm Lab", "Prediction History", "Report & Viva", "About"])
st.sidebar.caption("Educational decision-support prototype")

def get_trained():
    return st.session_state.training

def dataset_uploader():
    up = st.file_uploader("Upload Crop_recommendation.csv", type=["csv"], key="crop_dataset_upload")
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.session_state.source_df = df
            st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        except Exception as e: st.error(f"Could not read CSV: {e}")
    if st.session_state.source_df is None:
        local = HERE / "data" / "Crop_recommendation.csv"
        if local.exists():
            st.session_state.source_df = pd.read_csv(local)
    return st.session_state.source_df

if page == "Dashboard":
    st.title("CropWise Ultimate")
    st.subheader("Crop recommendation, model benchmarking and ML practicals in one app")
    st.write("Upload a crop dataset, compare classification models, tune selected hyperparameters, test recommendations and explore core ML algorithms.")
    r = get_trained()
    if r:
        a,b,c = st.columns(3)
        a.metric("Rows after cleaning", f'{r["cleaning"]["rows_after_cleaning"]:,}')
        b.metric("Crop classes", r["cleaning"]["class_count"])
        c.metric("Best model", r["best_name"])
        st.dataframe(r["metrics"], width="stretch")
    else:
        st.info("Start at **Model Training & SOTA** to upload the CSV and train the benchmark.")
    st.markdown("**Workflow:** CSV → cleaning → stratified split → model training → held-out evaluation → recommendation UI → SQLite history.")
    st.markdown("**Practicals as features:** Find-S → crop suitability profiles · ID3 → *why this crop?* rules · KNN + GridSearchCV → top picks & similar fields · Linear Regression → estimate an unknown input · K-Means → agro-climatic zones.")
elif page == "Recommend Crop":
    st.title("Recommend a crop")
    r = get_trained()
    if r is None:
        st.warning("Train the models on the Model Training & SOTA page first.")
    else:
        ins = r["insights"]
        st.caption("Enter soil nutrient levels and weather conditions. Tick **Don't know** to have Linear Regression estimate that input from the others.")
        defaults = {"N":90.0,"P":42.0,"K":43.0,"temperature":20.9,"humidity":82.0,"ph":6.5,"rainfall":202.9}
        with st.form("prediction_form"):
            cols = st.columns(4)
            vals, unknown = {}, []
            for i, f in enumerate(FEATURES):
                with cols[i%4]:
                    vals[f] = st.number_input(f, value=defaults[f], format="%.2f")
                    if st.checkbox("Don't know", key=f"unk_{f}"): unknown.append(f)
            submitted = st.form_submit_button("Predict crop", type="primary")
        if submitted:
            try:
                notes, estimates = None, {}
                if unknown:
                    known = {f: vals[f] for f in FEATURES if f not in unknown}
                    estimates = fx.estimate_missing(ins["X_train"], ins["X_test"], known, unknown)
                    for f, e in estimates.items(): vals[f] = e["value"]
                    notes = "Estimated by Linear Regression: " + ", ".join(f"{f}={e['value']:.2f} (R²={e['R2']:.2f})" for f, e in estimates.items())
                pred = r["best_model"].predict(pd.DataFrame([vals], columns=FEATURES))[0]
                database.save_prediction(vals, str(pred), r["best_name"], notes)
                st.session_state.last_rec = {"vals": vals, "pred": str(pred), "estimates": estimates}
            except Exception as e:
                st.session_state.last_rec = None; st.error(f"Prediction failed: {e}")
        rec = st.session_state.get("last_rec")
        if rec:
            vals, pred = rec["vals"], rec["pred"]
            st.success(f"Predicted crop: **{pred}**  (model: {r['best_name']})")
            st.caption("Dataset-based prediction, not a guarantee of agricultural success.")
            t1,t2,t3,t4,t5,t6 = st.tabs(["🥇 Top picks (KNN)","🌳 Why? (ID3 rules)","🔎 Similar fields (KNN)","🗺️ Climate zone (K-Means)","✅ Suitability (Find-S)","📐 Estimated inputs (Linear Regression)"])
            with t1:
                if "knn" in ins:
                    k = ins["knn"]; top = fx.knn_top_picks(k, vals)
                    st.dataframe(top.style.format({"Neighbour vote share":"{:.0%}"}), width="stretch", hide_index=True)
                    st.caption(f"Tuned KNN (GridSearchCV, training-set CV): {k['best_params']} · held-out accuracy {k['test_accuracy']:.3f}. Vote share is the fraction of neighbours voting for a crop, not a calibrated probability.")
                    if top.iloc[0]["Crop"] != pred: st.info("The best benchmark model and tuned KNN disagree — treat this input as borderline and check the alternatives.")
                else: st.warning("KNN feature unavailable: " + ins["errors"].get("knn","unknown error"))
            with t2:
                if "id3" in ins:
                    d = ins["id3"]; row = ins["binner5"].transform(pd.DataFrame([vals], columns=FEATURES)).iloc[0].to_dict()
                    ex = d["tree"].explain(row)
                    st.markdown(f"**ID3 says: {ex['prediction']}** · leaf support {ex['support']} samples · leaf purity {ex['purity']:.0%}")
                    for f, v, g in ex["path"]: st.write(f"• **{f}** is **{ins['binner5'].describe(f, v)}**  _(information gain {g:.3f} at this split)_")
                    st.caption(f"Hand-coded ID3 on 5-level binned inputs · held-out accuracy {d['test_accuracy']:.3f}. It may differ from the benchmark model; use it as an explanation, not a second opinion.")
                else: st.warning("ID3 feature unavailable: " + ins["errors"].get("id3","unknown error"))
            with t3:
                if "knn" in ins:
                    st.dataframe(fx.similar_fields(ins["knn"], vals, 5), width="stretch", hide_index=True)
                    st.caption("The closest records in the training data (standardised distance). Large distances mean your field is unlike anything in the dataset.")
            with t4:
                if "zones" in ins:
                    z = fx.assign_zone(ins["zones"], vals); st.markdown(f"**Your field falls in {z['name']}** (of {ins['zones']['k']} zones)")
                    st.dataframe(z["profile"].map(lambda v: f"{v:.1f}" if isinstance(v,(float,np.floating)) else str(v)).to_frame("Zone profile (training mean)"), width="stretch")
                    fig, ax = plt.subplots(figsize=(6,4)); P = ins["zones"]["points"]
                    ax.scatter(P[:,0], P[:,1], c=ins["zones"]["labels"], cmap="tab10", alpha=.35, s=12)
                    ax.scatter(*z["point"], marker="*", s=300, c="red", edgecolors="black", label="Your field"); ax.legend()
                    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    st.caption("Zones come from unsupervised K-Means on standardised inputs; they describe similar growing conditions and are not crop labels.")
                else: st.warning("Zone feature unavailable: " + ins["errors"].get("zones","unknown error"))
            with t5:
                if "profiles" in ins:
                    crops = sorted(ins["profiles"]); choice = st.selectbox("Check suitability for crop", crops, index=crops.index(pred) if pred in crops else 0)
                    variant = st.radio("Find-S variant", ["tolerant","strict"], horizontal=True, help="Strict is textbook Find-S; tolerant keeps a value if ≥90% of positive examples share it (extension for noisy data).")
                    res = fx.suitability(ins["binner3"], ins["profiles"], choice, vals, variant)
                    if res["score"] is None: st.info("This crop's hypothesis has no constraints (all '?') so every field matches. Try the tolerant variant.")
                    else:
                        st.metric("Matching conditions", f"{res['matched']}/{res['constrained']}")
                        st.dataframe(pd.DataFrame(res["checks"]).replace({True:"✅",False:"❌"}), width="stretch", hide_index=True)
                    st.caption("Find-S learns the most specific profile covering all (or ≥90%) of a crop's examples. A mismatch flags a condition worth adjusting; a match is not proof of success.")
                else: st.warning("Find-S feature unavailable: " + ins["errors"].get("profiles","unknown error"))
            with t6:
                if rec["estimates"]:
                    st.dataframe(pd.DataFrame([{"Input": f, "Estimated value": e["value"], "Held-out R²": e["R2"], "Held-out MAE": e["MAE"], "Predictors": ", ".join(e["predictors"])} for f, e in rec["estimates"].items()]), width="stretch", hide_index=True)
                    if any(e["R2"] < 0.5 for e in rec["estimates"].values()): st.warning("Some estimates have R² < 0.5 — they are weak guesses. Prefer measuring those inputs.")
                else: st.info("You supplied every input. Tick **Don't know** under an input to have it estimated.")
elif page == "Field Insights":
    st.title("Field Insights — dataset-wide views")
    r = get_trained()
    if r is None: st.warning("Train the models on the Model Training & SOTA page first.")
    else:
        ins = r["insights"]
        for k_, msg in ins["errors"].items(): st.warning(f"{k_} unavailable: {msg}")
        a,b,c,d,e = st.tabs(["Crop profiles (Find-S)","Decision rules (ID3)","KNN tuning","Input estimation (Linear Regression)","Zones (K-Means)"])
        with a:
            if "profiles" in ins:
                var = st.radio("Variant", ["tolerant","strict"], horizontal=True, key="prof_var")
                st.dataframe(pd.DataFrame({c_: v[var] for c_, v in ins["profiles"].items()}).T, width="stretch")
                st.caption("Rows are crops; columns are Low/Medium/High bins of each input (quantiles of the training data). '?' = no consistent requirement.")
        with b:
            if "id3" in ins:
                m1,m2 = st.columns(2); m1.metric("Held-out accuracy", f"{ins['id3']['test_accuracy']:.3f}"); m2.metric("Held-out macro F1", f"{ins['id3']['test_f1_macro']:.3f}")
                st.code(ins["id3"]["tree"].to_text(ins["binner5"]), language="text")
                st.download_button("Download rules (txt)", ins["id3"]["tree"].to_text(ins["binner5"]), "id3_rules.txt")
        with c:
            if "knn" in ins:
                k = ins["knn"]; m1,m2,m3 = st.columns(3)
                m1.metric("CV macro F1 (train)", "n/a" if k["cv_f1_macro"] is None else f"{k['cv_f1_macro']:.3f}"); m2.metric("Test accuracy", f"{k['test_accuracy']:.3f}"); m3.metric("Test macro F1", f"{k['test_f1_macro']:.3f}")
                st.json(k["best_params"])
        with d:
            if "regression" in ins:
                st.dataframe(ins["regression"].style.format({"R²":"{:.3f}","MAE":"{:.2f}","RMSE":"{:.2f}"}), width="stretch", hide_index=True)
                st.caption("How well each input can be estimated from the other six on held-out data. Low R² means that input should be measured, not guessed.")
        with e:
            if "zones" in ins:
                kk = st.slider("Number of zones (default = best silhouette)", 2, 8, ins["zones"]["k"])
                z = ins["zones"] if kk == ins["zones"]["k"] else fx.build_zones(ins["X_train"], ins["y_train"], k=kk)
                st.metric("Silhouette", f"{z['silhouette']:.3f}")
                st.dataframe(z["profiles"].style.format({f_:"{:.1f}" for f_ in FEATURES}), width="stretch", hide_index=True)
                fig, axs = plt.subplots(1,2,figsize=(11,4))
                axs[0].scatter(z["points"][:,0], z["points"][:,1], c=z["labels"], cmap="tab10", s=12, alpha=.6); axs[0].set_title("Zones (PCA view)")
                axs[1].plot(z["sweep"]["k"], z["sweep"]["inertia"], marker="o"); axs[1].set_title("Elbow (inertia)"); axs[1].set_xlabel("k")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
elif page == "Model Training & SOTA":
    st.title("Data cleaning, splitting and SOTA model comparison")
    df = dataset_uploader()
    if df is not None:
        st.subheader("Dataset preview")
        st.dataframe(df.head(10), width="stretch")
        st.write({"rows":len(df), "columns":len(df.columns), "missing_cells":int(df.isna().sum().sum()), "duplicate_rows":int(df.duplicated().sum())})
        test_size = st.slider("Test-set proportion", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        tune = st.checkbox("Tune selected models with cross-validated GridSearchCV (slower)", value=False)
        if st.button("Clean, train and compare models", type="primary"):
            with st.spinner("Training models and evaluating on the held-out test set..."):
                try:
                    result = train_and_compare(df, test_size=test_size, tune=tune)
                    st.session_state.training = result
                    save_training_result(result)
                    database.save_training(result["best_name"], result["cleaning"]["rows_after_cleaning"], len(result["models"]), result["metrics"].to_json(orient="records"))
                    st.success("Training complete. Results below are measured on this run.")
                except Exception as e: st.error(f"Training failed: {e}")
    else:
        st.info("Upload the CSV above or place it at `data/Crop_recommendation.csv`.")
        st.markdown("Expected columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`.")
    r = get_trained()
    if r:
        st.subheader("Cleaning report")
        st.json(r["cleaning"])
        st.subheader("Held-out test-set comparison")
        st.dataframe(r["metrics"].style.format({c:"{:.4f}" for c in r["metrics"].select_dtypes(include="number").columns}), width="stretch")
        st.download_button("Download model comparison CSV", r["metrics"].to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
        fig, ax = plt.subplots(figsize=(9,4))
        sns.barplot(data=r["metrics"], x="F1 (macro)", y="Model", ax=ax)
        ax.set_title("Macro F1 on held-out test set"); ax.set_xlim(0,1); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        if st.button("Show confusion matrix for best model"):
            pred = r["best_model"].predict(r["X_test"])
            from sklearn.metrics import confusion_matrix
            labels = sorted(r["y_test"].unique())
            cm = confusion_matrix(r["y_test"], pred, labels=labels)
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(cm, cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Best model confusion matrix")
            plt.xticks(rotation=90); plt.yticks(rotation=0); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
elif page == "Algorithm Lab":
    st.title("Core ML practicals lab")
    st.info("Each practical below is also wired into the main app as a feature — see **Recommend Crop** and **Field Insights**. This lab keeps the standalone teaching versions.")
    st.warning("Find-S requires a concept-learning dataset with a clearly defined positive target. Regression and K-Means examples use numeric data or a clearly labeled synthetic demo when the crop CSV is unsuitable.")
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Find-S","Linear Regression","Hyperparameter Tuning + KNN","Decision Tree / ID3","K-Means"])
    with tab1:
        st.markdown("### Find-S concept learning")
        st.write("Upload a small CSV where the final column is the target and positive values are Yes/Positive/1/True. Find-S ignores negative examples and generalizes feature values to `?` when positive examples disagree.")
        up = st.file_uploader("Find-S examples CSV", type=["csv"], key="finds")
        if up:
            try:
                fdf = pd.read_csv(up)
                st.dataframe(fdf.head())
                if st.button("Run Find-S"):
                    st.write("Final hypothesis:", lab.find_s(fdf))
            except Exception as e: st.error(str(e))
        else:
            st.caption("No Find-S dataset supplied. You can use a classic EnjoySport-style dataset as a separate practical.")
    with tab2:
        st.markdown("### Linear Regression")
        st.write("This is a regression exercise, not the crop-label classifier. It uses numeric columns from the uploaded crop CSV if suitable; otherwise it clearly labels a deterministic synthetic demo.")
        if st.button("Run Linear Regression demo"):
            df = st.session_state.source_df
            res = lab.linear_regression_demo(df)
            a,b,c = st.columns(3); a.metric("MAE",f'{res["MAE"]:.3f}'); b.metric("RMSE",f'{res["RMSE"]:.3f}'); c.metric("R²",f'{res["R2"]:.3f}')
            fig,ax=plt.subplots(figsize=(6,4))
            ax.scatter(res["actual"],res["predicted"],alpha=.75)
            low=min(float(np.min(res["actual"])),float(np.min(res["predicted"])))
            high=max(float(np.max(res["actual"])),float(np.max(res["predicted"])))
            ax.plot([low,high],[low,high],linestyle="--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Linear Regression: actual vs predicted"); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption("The app displays metrics from this run; they are not classification accuracy.")
    with tab3:
        st.markdown("### KNN + hyperparameter tuning")
        st.write("GridSearchCV uses five-fold cross-validation on the Iris teaching dataset to choose KNN neighbor count and weighting. This is separate from crop benchmark tuning.")
        if st.button("Run KNN GridSearchCV"):
            with st.spinner("Running small cross-validated search..."):
                res=lab.knn_tuning_demo()
            a,b,c=st.columns(3); a.metric("Best CV accuracy",f'{res["cv_score"]:.3f}'); b.metric("Test accuracy",f'{res["test_accuracy"]:.3f}'); c.metric("Test macro F1",f'{res["test_f1_macro"]:.3f}')
            st.json(res["best_params"])
    with tab4:
        st.markdown("### Decision Tree — ID3-style")
        st.write("The crop benchmark includes a decision tree with entropy criterion. This is ID3-style information-gain splitting, not a hand-coded reproduction of every historical ID3 detail.")
        df=st.session_state.source_df
        if df is not None and all(c in df.columns for c in FEATURES+[TARGET]):
            if st.button("Fit ID3-style tree on crop data"):
                clean,_=clean_dataset(df)
                Xtr,Xte,ytr,yte=train_test_split(clean[FEATURES],clean[TARGET].astype(str),test_size=.2,random_state=42,stratify=clean[TARGET].astype(str))
                model=lab.id3_fit(Xtr,ytr); pred=model.predict(Xte)
                st.metric("Held-out accuracy",f"{accuracy_score(yte,pred):.4f}")
                st.caption("Evaluation split uses random_state=42 and stratification.")
                from sklearn.tree import plot_tree
                fig,ax=plt.subplots(figsize=(16,8))
                plot_tree(model,feature_names=FEATURES,class_names=[str(c) for c in model.classes_],max_depth=3,filled=True,fontsize=7,ax=ax)
                st.pyplot(fig); plt.close(fig)
        else:
            st.info("Upload the crop dataset on Model Training & SOTA first.")
    with tab5:
        st.markdown("### K-Means clustering")
        st.write("K-Means is unsupervised and does not use crop labels to fit clusters. It standardizes numeric features, forms three clusters, and plots the first two standardized dimensions.")
        if st.button("Run K-Means clustering"):
            res=lab.kmeans_demo(st.session_state.source_df)
            fig,ax=plt.subplots(figsize=(7,5))
            for cl in sorted(res["plot_data"]["cluster"].unique()):
                sub=res["plot_data"][res["plot_data"]["cluster"]==cl]
                ax.scatter(sub["x"],sub["y"],label=f"Cluster {cl}",alpha=.65)
            ax.set_xlabel(f"Scaled {res['feature_names'][0]}"); ax.set_ylabel(f"Scaled {res['feature_names'][1]}")
            ax.set_title("K-Means clusters (first two scaled features)"); ax.legend(); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption("Clusters are groups, not verified crop recommendations. Interpret them cautiously.")
elif page == "Prediction History":
    st.title("Prediction history")
    rows = database.get_predictions()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), width="stretch")
    else: st.info("No predictions saved yet.")
    runs = database.get_training_runs()
    st.subheader("Training runs")
    if runs: st.dataframe(pd.DataFrame([dict(r) for r in runs]), width="stretch")
    else: st.info("No training runs saved yet.")
elif page == "Report & Viva":
    st.title("10-page report guide and viva notes")
    st.info("A report template is included in `reports/REPORT_TEMPLATE.md`. Replace all placeholders with your measured results and verified papers.")
    st.markdown("### Suggested 10-page allocation")
    st.table(pd.DataFrame([
        ["1","Introduction and problem statement"],["2–3","Literature review: at least 5 recent papers, compared critically"],
        ["4","Methodology and system architecture"],["5","Data cleaning, preprocessing and split"],
        ["6–7","SOTA model comparison, metrics, figures, confusion matrix"],["8","UI development and database"],
        ["9","Deployment, limitations, reproducibility"],["10","Conclusion and references"]
    ],columns=["Pages","Content"]))
    st.markdown("### Viva essentials")
    with st.expander("Why use a held-out test set?"): st.write("It estimates performance on unseen examples. The test set must not be used to tune model choices repeatedly.")
    with st.expander("Why macro F1?"): st.write("It calculates F1 per class and averages classes equally, making each crop class count equally even if class sizes differ.")
    with st.expander("What is hyperparameter tuning?"): st.write("It searches candidate settings using cross-validation on training data. Final performance is reported once on the held-out test set.")
    with st.expander("Why does K-Means not predict labels?"): st.write("K-Means is unsupervised and groups numeric observations by distance; clusters need interpretation and are not ground-truth crop labels.")
    with st.expander("Is this ID3?"): st.write("The implementation uses entropy criterion in scikit-learn and is described as ID3-style. It is not claimed to be a complete hand-coded original ID3 implementation.")
elif page == "About":
    st.title("About CropWise Ultimate")
    st.write("Educational prototype combining supervised crop classification, algorithm demonstrations, SQLite logging and model evaluation.")
    st.warning("Do not use the prediction as the sole basis for farming decisions. Real recommendations need region-specific agronomy, validated soil tests and expert review.")
    st.markdown("Source code is organized into `app.py`, `database.py`, `services/`, `reports/`, and `tests/`.")
