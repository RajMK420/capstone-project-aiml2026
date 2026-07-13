"""
Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: California Housing (the classic "housing.csv" version, one row per census
block group in California, sourced from the 1990 US Census; widely used in
Aurelien Geron's "Hands-On Machine Learning" and mirrored publicly on GitHub).

Why this dataset:
- 20,640 rows (>> 500 required)
- 8 numeric columns (longitude, latitude, housing_median_age, total_rooms,
  total_bedrooms, population, households, median_income) + numeric target
  (median_house_value) -> satisfies "at least 5 numeric columns"
- 1 native categorical column: ocean_proximity (5 values, no natural order -> nominal)
- A second categorical column, income_category, is engineered from median_income
  quartiles (Low < Medium < High < VeryHigh -> ordinal) so both an ordinal and a
  nominal categorical column are available for Part 2's encoding step.
- total_bedrooms has genuine missing values (~1% of rows) in the raw data.
- median_house_value is the continuous target for regression, and is binarized at
  its median for the classification task in Part 2.
- A small number of duplicate rows and some additional missingness are injected
  (seed=42, documented below) so the null-analysis and duplicate-removal tasks are
  exercised meaningfully -- the raw file only has one column with nulls and no
  duplicates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Task 1: Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("housing.csv")

print("=== Task 1: Load dataset ===")
print(df.head(), "\n")
print(df.dtypes, "\n")
print("Shape:", df.shape, "\n")

# ---------------------------------------------------------------------------
# Engineer a second categorical column (ordinal) from median_income
# ---------------------------------------------------------------------------
df["income_category"] = pd.qcut(
    df["median_income"], q=4, labels=["Low", "Medium", "High", "VeryHigh"]
)

# ---------------------------------------------------------------------------
# Inject a small amount of additional, controlled missingness + duplicates
# (reproducible with seed=42) so Tasks 2-3 have more to work with than the
# single real null column in the raw file.
# ---------------------------------------------------------------------------
def inject_nulls(frame, col, frac):
    n = len(frame)
    idx = rng.choice(n, size=int(n * frac), replace=False)
    frame.loc[frame.index[idx], col] = np.nan

inject_nulls(df, "total_rooms", 0.06)   # 6%  -> below threshold
inject_nulls(df, "households", 0.25)    # 25% -> ABOVE 20% threshold

dup_rows = df.sample(n=60, random_state=RNG_SEED)
df = pd.concat([df, dup_rows], ignore_index=True)

print("=== After adding income_category + injecting extra missingness/duplicates ===")
print("Shape:", df.shape, "\n")

# ---------------------------------------------------------------------------
# Task 2: Null value analysis
# ---------------------------------------------------------------------------
null_count = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_count, "null_pct": null_pct})
null_table = null_table[null_table["null_count"] > 0].sort_values("null_pct", ascending=False)

print("=== Task 2: Null value analysis ===")
print(null_table, "\n")

above_20 = null_table[null_table["null_pct"] > 20]
below_20 = null_table[null_table["null_pct"] <= 20]
print("Columns exceeding 20% nulls:", list(above_20.index))
print("Columns at/below 20% nulls (median-fill):", list(below_20.index), "\n")

for col in below_20.index:
    df[col] = df[col].fillna(df[col].median())

for col in above_20.index:
    df[col] = df[col].fillna(df[col].median())

print("Nulls remaining after fill:", df.isnull().sum().sum(), "\n")

# ---------------------------------------------------------------------------
# Task 3: Duplicate detection and removal
# ---------------------------------------------------------------------------
n_dupes = df.duplicated().sum()
print("=== Task 3: Duplicates ===")
print("Duplicate rows found:", n_dupes)

null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100

print("Shape after dedup:", df.shape)
print("Null % changed after dedup?", not null_pct_before.equals(null_pct_after), "\n")

# ---------------------------------------------------------------------------
# Task 4: Data type correction
# ---------------------------------------------------------------------------
mem_before = df.memory_usage(deep=True).sum()

df["housing_median_age"] = df["housing_median_age"].astype(str)
print("=== Task 4: Data type correction ===")
print("housing_median_age dtype before fix:", df["housing_median_age"].dtype)
df["housing_median_age"] = pd.to_numeric(df["housing_median_age"], errors="coerce")
print("housing_median_age dtype after fix:", df["housing_median_age"].dtype)

df["ocean_proximity"] = df["ocean_proximity"].astype("category")
df["income_category"] = df["income_category"].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage before: {mem_before/1024:.1f} KB, after: {mem_after/1024:.1f} KB\n")

# ---------------------------------------------------------------------------
# Task 5: Descriptive statistics and skewness
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("=== Task 5: Descriptive stats ===")
print(df[numeric_cols].describe(), "\n")

skew = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("Skewness (sorted by |skew|):\n", skew, "\n")
most_skewed_col = skew.index[0]
print("Most skewed column:", most_skewed_col, "skew =", skew.iloc[0], "\n")

# ---------------------------------------------------------------------------
# Task 6: Outlier detection with IQR
# ---------------------------------------------------------------------------
def iqr_outliers(frame, col):
    q1, q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (frame[col] < lower) | (frame[col] > upper)
    return mask.sum(), lower, upper

print("=== Task 6: IQR outliers ===")
for col in ["median_income", "total_rooms"]:
    n_out, lo, hi = iqr_outliers(df, col)
    print(f"{col}: {n_out} outliers outside [{lo:.2f}, {hi:.2f}]")
print()

# ---------------------------------------------------------------------------
# Task 7: Visualizations
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

plt.figure(figsize=(9, 4))
plt.plot(df["median_house_value"].sort_index().values[:300])
plt.title("Median House Value by Row Index (first 300 rows)")
plt.xlabel("Row index")
plt.ylabel("Median House Value ($)")
plt.tight_layout()
plt.savefig("plot_line.png", dpi=110)
plt.close()

plt.figure(figsize=(7, 4))
df.groupby("income_category", observed=True)["median_house_value"].mean().plot.bar(color="#4C72B0")
plt.title("Mean House Value by Income Category")
plt.xlabel("Income Category")
plt.ylabel("Mean Median House Value ($)")
plt.tight_layout()
plt.savefig("plot_bar.png", dpi=110)
plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(df[most_skewed_col], bins=20, kde=True)
plt.title(f"Distribution of {most_skewed_col} (most skewed)")
plt.xlabel(most_skewed_col)
plt.tight_layout()
plt.savefig("plot_hist.png", dpi=110)
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.2)
plt.title("Median Income vs Median House Value")
plt.tight_layout()
plt.savefig("plot_scatter.png", dpi=110)
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value")
plt.title("House Value Spread by Ocean Proximity")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot_box.png", dpi=110)
plt.close()

print("=== Task 7: Visualizations saved (plot_line/bar/hist/scatter/box.png) ===\n")

# ---------------------------------------------------------------------------
# Task 8: Correlation heat map
# ---------------------------------------------------------------------------
corr = df[numeric_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heat Map (numeric columns)")
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=110)
plt.close()

vals = corr.values.copy()
np.fill_diagonal(vals, 0)
corr_no_diag = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
max_pair = corr_no_diag.abs().unstack().sort_values(ascending=False).index[0]
print("=== Task 8: Correlation heat map ===")
print("Highest |correlation| pair:", max_pair, "value =", corr.loc[max_pair], "\n")

# ---------------------------------------------------------------------------
# Task 9a: Imputation strategy comparison for the two most-skewed columns
# ---------------------------------------------------------------------------
top2_skew_cols = skew.index[:2].tolist()
print("=== Task 9a: Imputation comparison (mean vs median) ===")
for col in top2_skew_cols:
    print(f"{col}: mean={df[col].mean():.4f}, median={df[col].median():.4f}, skew={skew[col]:.4f}")
    df[col] = df[col].fillna(df[col].median())
print("Remaining nulls after imputation:\n", df[top2_skew_cols].isnull().sum(), "\n")

# ---------------------------------------------------------------------------
# Task 9b: Spearman vs Pearson
# ---------------------------------------------------------------------------
pearson = df[numeric_cols].corr(method="pearson")
spearman = df[numeric_cols].corr(method="spearman")
diff = (spearman - pearson).abs()
dvals = diff.values.copy()
np.fill_diagonal(dvals, 0)
diff_no_diag = pd.DataFrame(dvals, index=diff.index, columns=diff.columns)
stacked = diff_no_diag.unstack().sort_values(ascending=False)
seen_pairs = set()
top3_pairs = []
for (a, b), d in stacked.items():
    key = frozenset([a, b])
    if key in seen_pairs:
        continue
    seen_pairs.add(key)
    top3_pairs.append((a, b, d))
    if len(top3_pairs) == 3:
        break

print("=== Task 9b: Spearman vs Pearson (top 3 differing pairs) ===")
for a, b, d in top3_pairs:
    print(f"{a} vs {b}: |Spearman-Pearson| = {d:.4f}  (Pearson={pearson.loc[a,b]:.3f}, Spearman={spearman.loc[a,b]:.3f})")
print()

# ---------------------------------------------------------------------------
# Task 9c: Grouped aggregation
# ---------------------------------------------------------------------------
grouped = df.groupby("ocean_proximity", observed=True)["median_house_value"].agg(["mean", "std", "count"])
print("=== Task 9c: Grouped aggregation (ocean_proximity -> median_house_value) ===")
print(grouped, "\n")
ratio = grouped["mean"].max() / grouped["mean"].min()
print("Highest-mean group:", grouped["mean"].idxmax())
print("Highest-std group:", grouped["std"].idxmax())
print("Max/min mean ratio:", round(ratio, 3), "\n")

# ---------------------------------------------------------------------------
# Task 10: Save cleaned dataset
# ---------------------------------------------------------------------------
df.to_csv("cleaned_data.csv", index=False)
print("=== Task 10: Saved cleaned_data.csv, shape =", df.shape, "===")
