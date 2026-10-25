# CropWise Ultimate

An educational B.Tech mini-project for crop classification, model benchmarking, data preprocessing, a Streamlit UI, SQLite logging, and a separate ML practicals lab.

## Features
- Data validation/cleaning: required columns, numeric conversion, invalid/missing rows, duplicate removal, row counts.
- Reproducible hold-out split (`random_state=42`, default 80/20; stratified when possible).
- Six crop classifiers compared on one split: Decision Tree (entropy), Random Forest, SVM with scaling, Gradient Boosting, KNN with scaling, and Extra Trees.
- Metrics: accuracy, macro precision, macro recall, macro F1, weighted F1, train time, prediction time.
- Optional GridSearchCV for selected crop models.
- Confusion matrix and Macro F1 comparison plot.
- Crop recommendation page and SQLite prediction/training history.
- **Practicals as product features** (built after each training run, fitted on the training split only):

| Practical | Feature in the app | Where |
|---|---|---|
| Find-S | Crop suitability profiles: "does my field match what crop X needs?" (strict Find-S + a labelled noise-tolerant extension) | Get a Recommendation → Suitability; Explore the Data |
| ID3 (hand-coded, entropy / information gain) | "Why this crop?" decision path and downloadable rule tree | Get a Recommendation → Why?; Explore the Data |
| KNN + GridSearchCV | Top-3 picks (neighbour vote share) and "similar fields" lookup | Get a Recommendation → Top picks / Similar fields; Explore the Data |
| Linear Regression | Tick "Don't know" on any input and it is estimated from the others, with held-out R²/MAE shown; flagged in history | Get a Recommendation → Estimated inputs |
| K-Means | Agro-climatic zones: your field is assigned a zone with its profile and dominant crops; k chosen by silhouette, elbow plot | Get a Recommendation → Climate zone; Explore the Data |

- Algorithm Lab: standalone teaching versions of the same practicals (Iris KNN demo, synthetic fallbacks, etc.).
- Report template.

## Requirements
Python 3.10+ (3.11 recommended). Windows/macOS/Linux.

## Quick start — Windows
1. Extract the ZIP.
2. Open `CropWise_Ultimate` folder.
3. Double-click `run_windows.bat`, or open CMD in this folder and run:
   ```cmd
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
4. Browse to `http://localhost:8501`.

If `python` is unavailable, install from https://www.python.org/downloads/ and tick **Add Python to PATH**. Some systems use `py` on Windows or `python3` on macOS/Linux.

## Dataset
The project intentionally does not redistribute the CSV. Obtain `Crop_recommendation.csv` from:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Check dataset license/attribution and upload it in **Train & Compare Models**, or place it at `data/Crop_recommendation.csv`. Expected columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`.

## Algorithm Lab notes
- **Find-S:** upload a concept-learning CSV with features plus a final target column whose positive examples are named Yes/Positive/1/True. This algorithm is not trained on the crop labels because those are multiclass classes, not positive/negative concept labels.
- **Linear Regression:** uses suitable numeric columns from the uploaded CSV where possible; otherwise uses a clearly identified deterministic synthetic example. Reports MAE, RMSE, R². It is a teaching exercise, not the crop classifier.
- **KNN + tuning:** demonstrates `GridSearchCV` on the Iris dataset with 5-fold CV. Crop benchmark can optionally tune KNN, SVM and Random Forest using training-set CV.
- **Decision Tree / ID3:** uses entropy/information-gain-style splitting with scikit-learn. This is accurately described as ID3-style, not a hand-coded reproduction of every historical ID3 implementation detail.
- **K-Means:** clusters standardized numeric features without using crop labels. Clusters are exploratory groups, not crop recommendations.
- **Hyperparameter tuning:** cross-validation runs only on the training split; held-out test set is for final evaluation.

## Reproducibility and responsible reporting
Never hard-code or invent metrics. Run the app with your actual dataset and use the measured table and plots. Note versions, split ratio, random seed, preprocessing choices and any tuned parameters in the report. A public demo may use Streamlit Community Cloud; see `DEPLOYMENT.md`. Do not commit secrets or large datasets unless the dataset license permits.

## Structure
- `app.py`: Streamlit UI
- `database.py`: SQLite tables for predictions and training runs
- `services/model_service.py`: preprocessing and benchmark training
- `services/lab_algorithms.py`: requested practical algorithms (standalone teaching versions)
- `services/features.py`: the same practicals wired into the crop workflow as features
- `reports/REPORT_TEMPLATE.md`: report outline
- `tests/test_core.py`: tests (run `cd tests && python -m pytest`); `tests/synthetic.py` is fake data for tests only
