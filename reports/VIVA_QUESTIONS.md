# CropWise Viva Questions and Short Answers

1. **What is the task?** Supervised multiclass classification: seven numeric inputs map to a crop label.
2. **Why clean data?** Invalid values, missing fields and duplicates can distort training and evaluation.
3. **Why split data?** Training learns parameters; test data estimates performance on unseen examples.
4. **Why stratify?** It approximately preserves class proportions in train and test sets when class counts allow it.
5. **What is accuracy?** Fraction of test examples classified correctly.
6. **Macro vs weighted F1?** Macro gives each class equal weight; weighted F1 weights each class by its support.
7. **Why compare models on the same split?** It makes comparisons more controlled and fair.
8. **What is hyperparameter tuning?** Selecting settings such as C, tree depth, or neighbor count using training-set cross-validation.
9. **Why not tune on test set?** That leaks test information into model selection and biases the final estimate.
10. **What does SVM scaling do?** Standardization helps distance/kernel calculations when features have different scales.
11. **What is Find-S?** It generalizes the most specific hypothesis consistent with positive examples; it does not learn from negative examples.
12. **Why is Linear Regression separate?** Crop labels are categorical; ordinary linear regression is not the appropriate primary classifier.
13. **What is ID3?** A decision-tree approach that chooses splits using entropy/information gain. This project labels sklearn's entropy tree as ID3-style.
14. **What is K-Means?** Unsupervised clustering that groups points by distance to cluster centroids.
15. **Does K-Means predict the correct crop?** No. Cluster IDs are not crop labels without further analysis.
16. **Why use SQLite?** It stores local prediction history and training run summaries.
17. **Is the model ready for real farming?** No; it is a student prototype and requires agronomic validation and local data.
18. **What is reproducibility?** Recording data source, versions, random seed, split, preprocessing, and parameters so the experiment can be repeated.
19. **How is Find-S used in the app?** For each crop, its samples are positives and all others negatives; the learned hypothesis is a suitability profile. Strict Find-S collapses to '?' on noisy data, so a ≥90% tolerant variant is also shown and labelled as an extension.
20. **Is the ID3 in the app the sklearn tree?** No. The feature uses a hand-coded ID3 on binned attributes (entropy, information gain, each attribute once per path). The benchmark's Decision Tree is sklearn's entropy tree.
21. **Why estimate inputs with Linear Regression?** Farmers may not know e.g. rainfall. The estimate is a weak guess when held-out R² is low, so the app shows R² and warns below 0.5.
22. **What do the K-Means zones mean?** Groups of similar growing conditions found without labels; not crop recommendations.
