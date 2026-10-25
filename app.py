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

st.sidebar.title("🌱 CropWise")
PAGES = ["Home", "Get a Recommendation", "Explore the Data", "Train & Compare Models", "Learn the Algorithms", "History"]
if "page" not in st.session_state: st.session_state.page = "Home"
page = st.sidebar.radio("Go to", PAGES, key="page")
st.sidebar.caption("Decision-support prototype built on a public dataset. Not agronomic advice.")
def go(p): st.session_state.page = p

def get_trained():
    return st.session_state.training

def explain(what, read, trust, algo=None):
    """Same three questions under every feature: what is it, how to read it, when to trust it."""
    with st.expander("ℹ️ What am I looking at?"):
        st.markdown(f"**What it is.** {what}")
        st.markdown(f"**How to read it.** {read}")
        st.markdown(f"**When to trust it.** {trust}")
        if algo: st.caption(f"Built with: {algo}")

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

INFO = {"N": ("Nitrogen (N)", "Nitrogen content ratio in the soil"), "P": ("Phosphorus (P)", "Phosphorus content ratio in the soil"),
        "K": ("Potassium (K)", "Potassium content ratio in the soil"), "temperature": ("Temperature (°C)", "Average temperature"),
        "humidity": ("Humidity (%)", "Relative humidity"), "ph": ("Soil pH", "Soil acidity (below 7) or alkalinity (above 7)"),
        "rainfall": ("Rainfall (mm)", "Rainfall")}

if page == "Home":
    st.title("🌱 CropWise")
    st.subheader("Describe your soil and weather. Get a crop suggestion and see exactly why.")
    r = get_trained()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1 · Train"); st.write("Load the crop dataset and train the models (about a minute).")
        st.write("✅ Done" if r else "⬜ Not done yet")
        st.button("Go to training →", on_click=go, args=("Train & Compare Models",), key="h1")
    with c2:
        st.markdown("### 2 · Ask"); st.write("Enter your field's conditions and get a recommendation with reasons.")
        st.write("Ready" if r else "Needs step 1")
        st.button("Get a recommendation →", on_click=go, args=("Get a Recommendation",), key="h2", disabled=not r)
    with c3:
        st.markdown("### 3 · Explore"); st.write("See what each crop needs, the decision rules and the climate zones.")
        st.write("Ready" if r else "Needs step 1")
        st.button("Explore the data →", on_click=go, args=("Explore the Data",), key="h3", disabled=not r)
    if r:
        a, b, c = st.columns(3)
        a.metric("Records used", f'{r["cleaning"]["rows_after_cleaning"]:,}'); b.metric("Crops it knows", r["cleaning"]["class_count"]); c.metric("Best model", r["best_name"])
    with st.expander("ℹ️ How does it work?"):
        st.markdown("The app learns from a public dataset of soil and weather readings labelled with the crop that suits them. "
                    "Several models are trained and scored on data they have never seen; the best one makes the recommendation. "
                    "Other methods then explain it: which conditions matter, which real records look similar, and what climate group your field belongs to.")
        st.caption("Built with: model comparison, Find-S, ID3, KNN, Linear Regression and K-Means — each is explained where it appears.")
    st.caption("Treat results as decision support only. Real planting decisions need local soil tests and expert advice.")

elif page == "Get a Recommendation":
    st.title("Get a crop recommendation")
    r = get_trained()
    if r is None:
        st.warning("The models need to be trained first.")
        st.button("Train them now →", on_click=go, args=("Train & Compare Models",))
    else:
        ins = r["insights"]; Xtr = ins["X_train"]
        st.write("Enter your field's conditions. If you don't have a measurement, tick **Estimate it** and the app works it out from the others.")
        defaults = {"N":90.0,"P":42.0,"K":43.0,"temperature":20.9,"humidity":82.0,"ph":6.5,"rainfall":202.9}
        with st.form("prediction_form"):
            cols = st.columns(4); vals, unknown = {}, []
            for i, f in enumerate(FEATURES):
                label, hint = INFO[f]
                with cols[i % 4]:
                    vals[f] = st.number_input(label, value=defaults[f], format="%.2f", help=f"{hint}. Range in the training data: {Xtr[f].min():.1f} to {Xtr[f].max():.1f}.")
                    if st.checkbox("Estimate it", key=f"unk_{f}", help="Tick if you don't know this value. It will be predicted from the other inputs."): unknown.append(f)
            submitted = st.form_submit_button("Recommend a crop", type="primary")
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
                st.session_state.last_rec = None; st.error(f"Could not make a recommendation: {e}")
        rec = st.session_state.get("last_rec")
        if rec:
            vals, pred = rec["vals"], rec["pred"]
            st.success(f"### Suggested crop: {pred}")
            outside = [INFO[f][0] for f in FEATURES if f not in rec["estimates"] and not (Xtr[f].min() <= vals[f] <= Xtr[f].max())]
            if outside: st.warning("These values are outside the range the models learned from, so treat the result with extra caution: " + ", ".join(outside) + ".")
            votes = {f"Best model ({r['best_name']})": pred}
            if "knn" in ins: votes["Nearest real records (KNN)"] = str(fx.knn_top_picks(ins["knn"], vals).iloc[0]["Crop"])
            if "id3" in ins: votes["Decision rules (ID3)"] = str(ins["id3"]["tree"].explain(ins["binner5"].transform(pd.DataFrame([vals], columns=FEATURES)).iloc[0].to_dict())["prediction"])
            agree = sum(v == pred for v in votes.values())
            (st.info if agree == len(votes) else st.warning)(f"{agree} of {len(votes)} methods agree on **{pred}**." + ("" if agree == len(votes) else " This input sits between crops, so compare the alternatives below."))
            with st.expander("See each method's answer"): st.table(pd.DataFrame({"Method": list(votes), "Suggests": list(votes.values())}))
            t1,t2,t3,t4,t5,t6 = st.tabs(["Best matches","Why this crop","Similar fields","Climate zone",f"Can I grow…?","Filled-in values"])
            with t1:
                st.caption("The three crops most common among the real records closest to your field.")
                if "knn" in ins:
                    k = ins["knn"]; st.dataframe(fx.knn_top_picks(k, vals).style.format({"Neighbour vote share":"{:.0%}"}), width="stretch", hide_index=True)
                    explain("The app finds the real records most similar to your field and counts which crop they grew.", "Vote share is the fraction of those neighbours that chose each crop. A high top share means your field looks very like known fields for that crop.",
                            "Trust it more when one crop dominates; if shares are close, your field is borderline. It is a vote count, not a calibrated probability.", f"KNN tuned with GridSearchCV ({k['best_params']}; held-out accuracy {k['test_accuracy']:.3f})")
                else: st.warning("Unavailable: " + ins["errors"].get("knn", "unknown error"))
            with t2:
                if "id3" in ins:
                    d = ins["id3"]; ex = d["tree"].explain(ins["binner5"].transform(pd.DataFrame([vals], columns=FEATURES)).iloc[0].to_dict())
                    st.caption("The chain of questions about your field that leads to the answer, most informative first.")
                    for f, v, g in ex["path"]: st.write(f"• **{INFO[f][0]}** is **{ins['binner5'].describe(f, v)}**")
                    st.markdown(f"➜ **{ex['prediction']}**  ·  based on {ex['support']} similar records, {ex['purity']:.0%} of which agree")
                    explain("A set of if-then rules learned from the data. Each step asks about the input that best separates the crops.", "Read the bullets top to bottom. The last line shows how many training records reached that answer and how many agreed.",
                            "Use it to understand the reasoning. It can differ from the main model, and a small support count means a shaky rule.", f"ID3 decision tree, hand-coded (held-out accuracy {d['test_accuracy']:.3f})")
                else: st.warning("Unavailable: " + ins["errors"].get("id3", "unknown error"))
            with t3:
                st.caption("The real records in the dataset that look most like your field.")
                if "knn" in ins:
                    st.dataframe(fx.similar_fields(ins["knn"], vals, 5), width="stretch", hide_index=True)
                    explain("The five closest records in the training data, measured after putting all inputs on the same scale.", "Check the Crop column for agreement. A small distance means a close match.",
                            "If every distance is large, your field is unlike anything in the dataset, so the recommendation is a guess.", "KNN nearest-neighbour search")
            with t4:
                if "zones" in ins:
                    z = fx.assign_zone(ins["zones"], vals); st.caption(f"Your field belongs to **{z['name']}** out of {ins['zones']['k']} climate and soil groups.")
                    st.dataframe(z["profile"].map(lambda v: f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v)).to_frame("Typical values in this zone"), width="stretch")
                    fig, ax = plt.subplots(figsize=(6,4)); P = ins["zones"]["points"]
                    ax.scatter(P[:,0], P[:,1], c=ins["zones"]["labels"], cmap="tab10", alpha=.35, s=12)
                    ax.scatter(*z["point"], marker="*", s=300, c="red", edgecolors="black", label="Your field"); ax.legend(); ax.set_xlabel("Combined axis 1"); ax.set_ylabel("Combined axis 2"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    explain("Fields were grouped automatically by how similar their soil and weather are, without looking at crop names.", "The table shows the average conditions of your zone and which crops are common there. The red star is your field among all others.",
                            "Zones describe similar growing conditions. They are not recommendations by themselves.", "K-Means clustering")
                else: st.warning("Unavailable: " + ins["errors"].get("zones", "unknown error"))
            with t5:
                if "profiles" in ins:
                    crops = sorted(ins["profiles"]); c_ = st.selectbox("Pick a crop to check", crops, index=crops.index(pred) if pred in crops else 0)
                    var = st.radio("Strictness", ["tolerant", "strict"], horizontal=True, format_func=lambda x: "Relaxed (recommended)" if x == "tolerant" else "Strict (textbook Find-S)")
                    hyp = ins["profiles"][c_][var]; need = [f"{INFO[f][0]}: **{ins['binner3'].describe(f, v)}**" for f, v in hyp.items() if v != "?"]
                    st.markdown(f"**What {c_} usually needs:** " + ("; ".join(need) if need else "no consistent requirement found in this mode."))
                    res = fx.suitability(ins["binner3"], ins["profiles"], c_, vals, var)
                    if res["score"] is not None:
                        st.metric("Conditions your field meets", f"{res['matched']} of {res['constrained']}")
                        st.dataframe(pd.DataFrame(res["checks"]).replace({True:"✅",False:"❌"}), width="stretch", hide_index=True)
                    explain("A profile of the conditions shared by the crop's records, compared against your field.", "✅ means your field falls in the range the crop usually needs; ❌ shows what to adjust or reconsider.",
                            "Good as a checklist. Strict mode often finds almost no requirements because real data is noisy, so relaxed mode keeps a condition if at least 90% of records share it (an extension of Find-S).", "Find-S concept learning")
                else: st.warning("Unavailable: " + ins["errors"].get("profiles", "unknown error"))
            with t6:
                if rec["estimates"]:
                    st.dataframe(pd.DataFrame([{"Input": INFO[f][0], "Estimated value": e["value"], "Accuracy (R²)": e["R2"], "Typical error": e["MAE"], "Based on": ", ".join(INFO[p][0] for p in e["predictors"])} for f, e in rec["estimates"].items()]), width="stretch", hide_index=True)
                    if any(e["R2"] < 0.5 for e in rec["estimates"].values()): st.warning("Some estimates are weak guesses (accuracy below 0.5). Measure those inputs if you can.")
                else: st.info("You entered every value, so nothing was estimated. Tick **Estimate it** under an input to try this.")
                explain("Missing inputs are predicted from the ones you did know, using patterns in the data.", "Accuracy (R²) runs from 0 (no better than guessing the average) to 1 (perfect). Typical error is in the input's own units.",
                        "Rely on estimates only when accuracy is high. They are also saved with the prediction in History.", "Linear Regression")

elif page == "Explore the Data":
    st.title("Explore what the models learned")
    r = get_trained()
    if r is None:
        st.warning("The models need to be trained first."); st.button("Train them now →", on_click=go, args=("Train & Compare Models",), key="e0")
    else:
        ins = r["insights"]
        for k_, msg in ins["errors"].items(): st.warning(f"{k_} is unavailable: {msg}")
        a,b,c,d,e = st.tabs(["What each crop needs","Decision rules","Neighbour search","Missing-value accuracy","Climate zones"])
        with a:
            if "profiles" in ins:
                var = st.radio("Strictness", ["tolerant","strict"], horizontal=True, key="prof_var", format_func=lambda x: "Relaxed (recommended)" if x == "tolerant" else "Strict (textbook Find-S)")
                st.dataframe(pd.DataFrame({c_: v[var] for c_, v in ins["profiles"].items()}).T, width="stretch")
                explain("For every crop, the Low/Medium/High level of each input that its records have in common.", "Rows are crops. '?' means the crop grows across different levels of that input.",
                        "Levels are relative to this dataset (split into thirds), not agronomic standards.", "Find-S concept learning")
        with b:
            if "id3" in ins:
                m1,m2 = st.columns(2); m1.metric("Accuracy on unseen data", f"{ins['id3']['test_accuracy']:.3f}"); m2.metric("Macro F1", f"{ins['id3']['test_f1_macro']:.3f}")
                txt = ins["id3"]["tree"].to_text(ins["binner5"]); st.code(txt, language="text"); st.download_button("Download rules (txt)", txt, "id3_rules.txt")
                explain("The full set of questions the rule-based model asks, from the most informative input down to a crop.", "Follow a branch from the top; each line adds a condition. '➜' marks the crop and (n) the records behind it.",
                        "Long branches with small n are weak. Use the rules to understand patterns, not as the final answer.", "ID3, hand-coded with entropy and information gain")
        with c:
            if "knn" in ins:
                k = ins["knn"]; m1,m2,m3 = st.columns(3)
                m1.metric("Cross-validation score", "n/a" if k["cv_f1_macro"] is None else f"{k['cv_f1_macro']:.3f}"); m2.metric("Accuracy on unseen data", f"{k['test_accuracy']:.3f}"); m3.metric("Macro F1", f"{k['test_f1_macro']:.3f}")
                st.json(k["best_params"])
                explain("The settings chosen for the neighbour search (how many neighbours, and whether closer ones count more).", "Higher scores mean the search recommends the right crop more often on data it has not seen.",
                        "Settings were chosen using training data only, so the held-out score is an honest estimate.", "KNN with GridSearchCV")
        with d:
            if "regression" in ins:
                st.dataframe(ins["regression"].style.format({"R²":"{:.3f}","MAE":"{:.2f}","RMSE":"{:.2f}"}), width="stretch", hide_index=True)
                explain("How well each input can be predicted from the other six.", "R² near 1 means it can be estimated reliably; near 0 means the estimate is little better than the average. MAE and RMSE are typical errors in the input's own units.",
                        "If R² is low for an input, measure it instead of using 'Estimate it'.", "Linear Regression")
        with e:
            if "zones" in ins:
                kk = st.slider("Number of zones (default is the best-separated grouping)", 2, 8, ins["zones"]["k"])
                z = ins["zones"] if kk == ins["zones"]["k"] else fx.build_zones(ins["X_train"], ins["y_train"], k=kk)
                st.metric("Separation score (silhouette)", f"{z['silhouette']:.3f}")
                st.dataframe(z["profiles"].style.format({f_:"{:.1f}" for f_ in FEATURES}), width="stretch", hide_index=True)
                fig, axs = plt.subplots(1,2,figsize=(11,4))
                axs[0].scatter(z["points"][:,0], z["points"][:,1], c=z["labels"], cmap="tab10", s=12, alpha=.6); axs[0].set_title("Zones (2-D view)")
                axs[1].plot(z["sweep"]["k"], z["sweep"]["inertia"], marker="o"); axs[1].set_title("Elbow: tightness vs number of zones"); axs[1].set_xlabel("Number of zones")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                explain("Groups of fields with similar soil and weather, found without using crop names.", "Each table row is a zone with its average conditions and most common crops. Separation score runs from -1 to 1; higher means cleaner groups. On the elbow chart, look for where the curve bends.",
                        "Different numbers of zones give different groupings, so treat them as a way to explore, not as fixed categories.", "K-Means clustering")

elif page == "Train & Compare Models":
    st.title("Train and compare models")
    st.markdown("**1.** Load the dataset  **2.** The app cleans it and splits it 80/20 (the 20% is kept hidden)  **3.** Six models learn from the 80%  **4.** They are scored on the hidden 20%, and the best one is used for recommendations.")
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
        m0 = r["metrics"].iloc[0]
        st.success(f"Best model on unseen data: **{r['best_name']}** (accuracy {m0['Accuracy']:.3f}, macro F1 {m0['F1 (macro)']:.3f}).")
        st.button("Next: get a recommendation →", on_click=go, args=("Get a Recommendation",), key="t_next")
        st.subheader("Cleaning report")
        st.caption("What was checked and removed before training (missing, invalid and duplicate rows).")
        st.json(r["cleaning"])
        st.subheader("Held-out test-set comparison")
        st.dataframe(r["metrics"].style.format({c:"{:.4f}" for c in r["metrics"].select_dtypes(include="number").columns}), width="stretch")
        with st.expander("ℹ️ What do these columns mean?"):
            st.markdown("**Accuracy**: share of test fields where the crop was right.  \n**Precision**: when a model names a crop, how often it is right.  \n**Recall**: how many of a crop's real fields the model finds.  \n**F1 (macro)**: balances precision and recall, averaged equally over all crops. The best model is ranked by this.  \n**Train / Predict time**: seconds to learn and to answer.")
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
elif page == "Learn the Algorithms":
    st.title("Learn the algorithms")
    st.info("Each practical below is also wired into the main app as a feature — see **Get a Recommendation** and **Explore the Data**. This lab keeps the standalone teaching versions.")
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
elif page == "History":
    st.title("History")
    st.caption("Every recommendation made in this app, with any values that were estimated, and past training runs.")
    rows = database.get_predictions()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), width="stretch")
    else: st.info("No predictions saved yet.")
    runs = database.get_training_runs()
    st.subheader("Training runs")
    if runs: st.dataframe(pd.DataFrame([dict(r) for r in runs]), width="stretch")
    else: st.info("No training runs saved yet.")
