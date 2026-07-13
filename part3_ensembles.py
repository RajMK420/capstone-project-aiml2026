"""
Part 3 - Advanced Modeling - Ensembles, Tuning, Full ML Pipeline
Reuses the same preprocessing + split as Part 2 (same random_state=42) so
X_train_scaled / X_test_scaled / y_clf_train / y_clf_test are identical.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Rebuild the exact same preprocessing + split as Part 2
# ---------------------------------------------------------------------------
df = pd.read_csv("cleaned_data.csv")
y_reg_full = df["median_house_value"]
y_clf_full = (y_reg_full > y_reg_full.median()).astype(int)
X_full = df.drop(columns=["median_house_value"])

order_map = {"Low": 0, "Medium": 1, "High": 2, "VeryHigh": 3}
X_full["income_category"] = X_full["income_category"].map(order_map)
X_full = pd.get_dummies(X_full, columns=["ocean_proximity"], drop_first=True)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_full, y_reg_full, y_clf_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X_full.columns.tolist()

print("Reproduced Part 2 split:", X_train.shape, X_test.shape, "\n")

# ---------------------------------------------------------------------------
# Task 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------------------
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train_scaled, y_clf_train)
train_acc_base = accuracy_score(y_clf_train, dt_base.predict(X_train_scaled))
test_acc_base = accuracy_score(y_clf_test, dt_base.predict(X_test_scaled))
print("=== Task 1: Unconstrained Decision Tree ===")
print(f"Train acc: {train_acc_base:.4f}   Test acc: {test_acc_base:.4f}\n")

# ---------------------------------------------------------------------------
# Task 2: Controlled Decision Tree
# ---------------------------------------------------------------------------
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_ctrl.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
print("=== Task 2: Controlled Decision Tree (max_depth=5, min_samples_split=20) ===")
print(f"Train acc: {train_acc_ctrl:.4f}   Test acc: {test_acc_ctrl:.4f}")
print(f"Train/test gap - unconstrained: {train_acc_base-test_acc_base:.4f}, controlled: {train_acc_ctrl-test_acc_ctrl:.4f}\n")

# ---------------------------------------------------------------------------
# Task 3: Gini vs Entropy
# ---------------------------------------------------------------------------
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))
acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))
print("=== Task 3: Gini vs Entropy (both max_depth=5) ===")
print(f"Gini test acc: {acc_gini:.4f}   Entropy test acc: {acc_entropy:.4f}\n")

# ---------------------------------------------------------------------------
# Task 4: Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("=== Task 4: Random Forest ===")
print(f"Train acc: {rf_train_acc:.4f}   Test acc: {rf_test_acc:.4f}   Test AUC: {rf_auc:.4f}")
print("Top 5 features by importance:\n", importances.head(5), "\n")

# ---------------------------------------------------------------------------
# Task 4a: Gradient Boosting
# ---------------------------------------------------------------------------
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print("=== Task 4a: Gradient Boosting ===")
print(f"Train acc: {gb_train_acc:.4f}   Test acc: {gb_test_acc:.4f}   Test AUC: {gb_auc:.4f}\n")

# ---------------------------------------------------------------------------
# Task 4b: Feature ablation study
# ---------------------------------------------------------------------------
lowest5 = importances.tail(5).index.tolist()
X_train_reduced = pd.DataFrame(X_train_scaled, columns=feature_names).drop(columns=lowest5).values
X_test_reduced = pd.DataFrame(X_test_scaled, columns=feature_names).drop(columns=lowest5).values

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print("=== Task 4b: Feature ablation (5 lowest-importance features removed) ===")
print("Removed features:", lowest5)
print(f"Full model test AUC: {rf_auc:.4f}   Reduced model test AUC: {rf_reduced_auc:.4f}")
print("AUC drop:", round(rf_auc - rf_reduced_auc, 4), "\n")

# ---------------------------------------------------------------------------
# Task 5: Cross-validated comparison
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models_for_cv = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

cv_results = {}
print("=== Task 5: 5-fold cross-validated ROC-AUC ===")
for name, model in models_for_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: mean AUC = {scores.mean():.4f}  std = {scores.std():.4f}")
print()

# ---------------------------------------------------------------------------
# Task 6: Hyperparameter tuning with GridSearchCV (full pipeline)
# ---------------------------------------------------------------------------
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42),
)
grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=grid_cv, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_clf_train)  # unscaled - pipeline handles scaling

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
total_fits = n_configs * 5

print("=== Task 6: GridSearchCV (Random Forest pipeline) ===")
print("Best params:", grid_search.best_params_)
print("Best CV AUC:", round(grid_search.best_score_, 4))
print(f"Total configurations evaluated: {n_configs} (x5 folds = {total_fits} fits)\n")

best_pipeline = grid_search.best_estimator_

# ---------------------------------------------------------------------------
# Task 7: Manual learning curve
# ---------------------------------------------------------------------------
print("=== Task 7: Manual learning curve (best pipeline) ===")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
lc_rows = []
for f in fractions:
    n = int(f * len(X_train))
    X_sub = X_train.iloc[:n]
    y_sub = y_clf_train.iloc[:n]
    best_pipeline.fit(X_sub, y_sub)
    train_auc = roc_auc_score(y_sub, best_pipeline.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    lc_rows.append((f, n, train_auc, test_auc))

lc_df = pd.DataFrame(lc_rows, columns=["Training fraction", "n_rows", "Training AUC", "Test AUC"])
print(lc_df.to_string(index=False), "\n")

# refit on full training data before saving
best_pipeline.fit(X_train, y_clf_train)

# ---------------------------------------------------------------------------
# Task 8: Serialize the best model
# ---------------------------------------------------------------------------
joblib.dump(best_pipeline, "best_model.pkl")
print("=== Task 8: Saved best_model.pkl ===")

loaded_pipeline = joblib.load("best_model.pkl")
sample_rows = X_test.iloc[:2]
preds = loaded_pipeline.predict(sample_rows)
probs = loaded_pipeline.predict_proba(sample_rows)[:, 1]
print("Reload-and-predict check on 2 test rows:")
print("Predictions:", preds, "Probabilities:", np.round(probs, 4), "\n")

# ---------------------------------------------------------------------------
# Task 9: Summary comparison table
# ---------------------------------------------------------------------------
test_auc_map = {
    "Logistic Regression": roc_auc_score(y_clf_test, LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(X_train_scaled, y_clf_train).predict_proba(X_test_scaled)[:, 1]),
    "Decision Tree (depth=5)": roc_auc_score(y_clf_test, dt_ctrl.predict_proba(X_test_scaled)[:, 1]),
    "Random Forest": rf_auc,
    "Gradient Boosting": gb_auc,
}

print("=== Task 9: Summary comparison table ===")
print(f"{'Model':<28}{'CV Mean AUC':>14}{'CV Std AUC':>13}{'Test AUC':>12}")
for name in models_for_cv:
    mean_auc, std_auc = cv_results[name]
    print(f"{name:<28}{mean_auc:>14.4f}{std_auc:>13.4f}{test_auc_map[name]:>12.4f}")
print(f"{'Tuned RF (GridSearchCV)':<28}{grid_search.best_score_:>14.4f}{'--':>13}{roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1]):>12.4f}")
