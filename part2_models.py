"""
Part 2 - Supervised Machine Learning - Build, Train, Evaluate
Loads cleaned_data.csv from Part 1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Task 1: Load data, define X, y_reg, y_clf
# ---------------------------------------------------------------------------
df = pd.read_csv("cleaned_data.csv")

y_reg_full = df["median_house_value"]
y_clf_full = (y_reg_full > y_reg_full.median()).astype(int)
X_full = df.drop(columns=["median_house_value"])

print("=== Task 1: Label definitions ===")
print("y_reg = median_house_value (continuous, dollars)")
print("y_clf = 1 if median_house_value > overall median, else 0 (binary)")
print("Class balance:\n", y_clf_full.value_counts(normalize=True), "\n")

# ---------------------------------------------------------------------------
# Task 2: Encode categorical columns
# ---------------------------------------------------------------------------
# income_category is ORDINAL (Low < Medium < High < VeryHigh) -> integer mapping
order_map = {"Low": 0, "Medium": 1, "High": 2, "VeryHigh": 3}
X_full["income_category"] = X_full["income_category"].map(order_map)

# ocean_proximity is NOMINAL (no natural order) -> one-hot, drop first to avoid
# multicollinearity (the dropped category becomes the implicit "baseline").
X_full = pd.get_dummies(X_full, columns=["ocean_proximity"], drop_first=True)

print("=== Task 2: Columns after encoding ===")
print(list(X_full.columns), "\n")

# ---------------------------------------------------------------------------
# Task 3: Leak-free train-test split and scaling
# ---------------------------------------------------------------------------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_full, y_reg_full, y_clf_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X_full.columns.tolist()

print("=== Task 3: Split shapes ===")
print("X_train:", X_train.shape, "X_test:", X_test.shape, "\n")

# ---------------------------------------------------------------------------
# Task 4: Regression - Linear Regression + Ridge
# ---------------------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)

coef_table = pd.DataFrame({"feature": feature_names, "coef": lin_reg.coef_})
coef_table["abs_coef"] = coef_table["coef"].abs()
coef_table = coef_table.sort_values("abs_coef", ascending=False)

print("=== Task 4: Linear Regression ===")
print(f"MSE = {mse_lin:,.2f}   R2 = {r2_lin:.4f}")
print("Top 3 |coefficient| features:")
print(coef_table.head(3).to_string(index=False), "\n")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)

print("=== Ridge (alpha=1.0) vs Linear Regression ===")
print(f"{'Model':<18}{'MSE':>15}{'R2':>10}")
print(f"{'Linear Regression':<18}{mse_lin:>15,.2f}{r2_lin:>10.4f}")
print(f"{'Ridge':<18}{mse_ridge:>15,.2f}{r2_ridge:>10.4f}\n")

# ---------------------------------------------------------------------------
# Task 5: Classification - Logistic Regression
# ---------------------------------------------------------------------------
print("=== Task 5: y_clf_train class balance ===")
print(y_clf_train.value_counts(normalize=True), "\n")
# Median-split target -> classes are ~50/50 by construction, no rebalancing needed.
# (Documented decision, see README.)

log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
report = classification_report(y_clf_test, y_pred_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)

print("=== Logistic Regression (C=1.0) ===")
print("Confusion matrix:\n", cm)
print(report)
print("AUC:", round(auc, 4), "\n")

fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (C=1.0)")
plt.annotate(f"AUC = {auc:.3f}", xy=(0.6, 0.2), fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig("plot_roc.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# Task 5b: Decision-threshold sensitivity
# ---------------------------------------------------------------------------
print("=== Task 5b: Threshold sensitivity (C=1.0 model) ===")
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
thresh_rows = []
for t in thresholds:
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t)
    r = recall_score(y_clf_test, preds_t)
    f1 = f1_score(y_clf_test, preds_t)
    thresh_rows.append((t, p, r, f1))

thresh_df = pd.DataFrame(thresh_rows, columns=["Threshold", "Precision", "Recall", "F1"])
print(thresh_df.to_string(index=False), "\n")
best_thresh_row = thresh_df.loc[thresh_df["F1"].idxmax()]
print("Threshold that maximises F1:", best_thresh_row["Threshold"], "\n")

# ---------------------------------------------------------------------------
# Task 6: Regularization experiment
# ---------------------------------------------------------------------------
log_reg_weak = LogisticRegression(max_iter=1000, C=0.01, random_state=42)
log_reg_weak.fit(X_train_scaled, y_clf_train)
y_pred_weak = log_reg_weak.predict(X_test_scaled)
y_proba_weak = log_reg_weak.predict_proba(X_test_scaled)[:, 1]

precision_1 = precision_score(y_clf_test, y_pred_clf)
recall_1 = recall_score(y_clf_test, y_pred_clf)
auc_1 = auc

precision_001 = precision_score(y_clf_test, y_pred_weak)
recall_001 = recall_score(y_clf_test, y_pred_weak)
auc_001 = roc_auc_score(y_clf_test, y_proba_weak)

print("=== Task 6: Regularization comparison ===")
print(f"{'Model':<12}{'Precision':>12}{'Recall':>10}{'AUC':>10}")
print(f"{'C=1.0':<12}{precision_1:>12.4f}{recall_1:>10.4f}{auc_1:>10.4f}")
print(f"{'C=0.01':<12}{precision_001:>12.4f}{recall_001:>10.4f}{auc_001:>10.4f}\n")

# ---------------------------------------------------------------------------
# Task 7: Bootstrap confidence interval for AUC difference
# ---------------------------------------------------------------------------
n_boot = 500
y_test_arr = y_clf_test.to_numpy()
diffs = []
for _ in range(n_boot):
    idx = rng.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
    y_sample = y_test_arr[idx]
    proba_1_sample = y_proba_clf[idx]
    proba_001_sample = y_proba_weak[idx]
    # skip degenerate bootstrap samples with only one class present
    if len(np.unique(y_sample)) < 2:
        continue
    auc_1_s = roc_auc_score(y_sample, proba_1_sample)
    auc_001_s = roc_auc_score(y_sample, proba_001_sample)
    diffs.append(auc_1_s - auc_001_s)

diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print("=== Task 7: Bootstrap CI for AUC difference (C=1.0 minus C=0.01) ===")
print(f"Valid bootstrap samples: {len(diffs)} / {n_boot}")
print(f"Mean AUC difference: {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print("Excludes zero?", not (ci_low <= 0 <= ci_high), "\n")

print("=== Part 2 complete ===")
