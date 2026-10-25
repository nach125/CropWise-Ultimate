# CropWise Ultimate: Crop Recommendation and Machine-Learning Benchmark

> **Submission note:** Keep the final report to 10 pages. Replace every bracketed placeholder with results from your own run and references you have read and verified. Do not present illustrative or synthetic Algorithm Lab metrics as crop-classification results.

## 1. Introduction
Describe crop selection as a classification/decision-support problem. State the input variables (N, P, K, temperature, humidity, pH, rainfall) and target crop label. Explain the intended users and scope. Clearly state that predictions are limited by dataset representativeness and are not agronomic guarantees.

## 2. Literature Review (at least five recent papers)
Build a comparison table with: citation/year, problem and dataset, method(s), evaluation metrics, key findings, limitations, and relevance to CropWise. Search recent peer-reviewed work on crop recommendation, precision agriculture, ML model comparisons, explainability, and deployment. Verify bibliographic metadata and DOI from publisher/Google Scholar/Crossref before citing. Discuss at least five papers; do not invent citations or claim to have reviewed papers you have not read.

| Paper | Dataset/task | Models | Metrics/results reported by authors | Limitations | Relevance |
|---|---|---|---|---|---|
| [Verified paper 1] | [fill] | [fill] | [cite exact reported results] | [fill] | [fill] |
| [Verified paper 2] | [fill] | [fill] | [fill] | [fill] | [fill] |
| [Verified paper 3] | [fill] | [fill] | [fill] | [fill] | [fill] |
| [Verified paper 4] | [fill] | [fill] | [fill] | [fill] | [fill] |
| [Verified paper 5] | [fill] | [fill] | [fill] | [fill] | [fill] |

## 3. Methodology
Include a neat architecture diagram: CSV → validation/cleaning → reproducible train/test split → model training/tuning → held-out evaluation → best model → Streamlit interface → SQLite history. Explain that tuning uses only training-fold cross-validation.

## 4. Data Cleaning and Preprocessing
Report original row count, missing/invalid values, duplicate rows, final row count, features, target, class counts, and train/test split. Describe numeric conversion and removal policy. State random seed and whether stratification was used. Add a compact cleaning table using the app's measured cleaning report.

## 5. Results and Discussion
Report actual metrics from `model_comparison.csv`. Discuss trade-offs, not just the winning score. Explain macro vs weighted F1, training/prediction time, and errors in the confusion matrix. Add figure(s) exported from the app or recreated from real results.

## 6. State-of-the-art Model Comparison and Performance Analytics
Compare at least four models on the same split. Suggested candidates: Decision Tree, Random Forest, SVM, Gradient Boosting, KNN, Extra Trees. Include Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, training time, prediction time. Explain that a high score on one dataset is not proof of real-world generalization.

## 7. User Interface Development and Deployment
Explain Streamlit pages (Dashboard, Recommend Crop, Model Training & SOTA, Algorithm Lab, Prediction History, Report & Viva), input validation, model training flow, and SQLite persistence. Include screenshots. Explain local run and chosen public deployment platform if deployed; do not say deployed unless it is live.

## 8. Practicals as Features and Algorithm Lab (briefly fit within page limit)
Describe how each practical became a feature (see README table): Find-S suitability, ID3 explanations, KNN top picks/similar fields, Linear Regression input estimation, K-Means zones. Report the held-out numbers shown on the Field Insights page from your own run.

- Find-S: concept learning over positive examples; needs a binary concept target.
- Linear Regression: regression metrics MAE/RMSE/R²; not a crop-label classifier.
- Hyperparameter tuning: GridSearchCV and cross-validation.
- KNN: distance-based supervised classifier.
- Decision Tree ID3-style: entropy/information-gain splits.
- K-Means: unsupervised clustering; cluster IDs have no inherent crop label.

## 9. Conclusion and Limitations
Summarize measured results, limitations of the dataset, risks of distribution shift, lack of local agronomy inputs, and future work (external validation, explainability, calibrated probabilities, persistent cloud database).

## 10. References
Use a consistent citation style (IEEE recommended). Include dataset citation and at least five verified recent papers. Include DOI/URL and access date where required.
