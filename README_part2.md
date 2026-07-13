# Part 2 — Supervised ML: Regression + Classification

Loads `cleaned_data.csv` from Part 1.

## Labels
- `y_reg` = `median_house_value` (continuous, dollars)
- `y_clf` = 1 if `median_house_value` is above the overall median, else 0. I went
  with the median split instead of a different natural binary column because it
  guarantees a clean, meaningful "high value vs low value" split and, as a bonus,
  gives roughly balanced classes (~50/50) instead of an imbalanced one.

## Encoding
- `income_category` has a real order (Low < Medium < High < VeryHigh), so I mapped
  it straight to integers 0-3. Label encoding is fine here because the numeric
  order actually matches the real-world order — a model can use "bigger number =
  higher income bracket" correctly.
- `ocean_proximity` has no order (INLAND isn't "more" or "less" than NEAR BAY), so
  I one-hot encoded it and dropped the first dummy column. If I'd label-encoded it
  instead, the model would wrongly assume some categories are numerically "closer"
  to each other than others, which isn't true for a location label.

## Leak-free split and scaling
Split 80/20 with `random_state=42`, then fit `StandardScaler` only on the training
features and used that same fitted scaler to transform both train and test. If I
had fit the scaler on the whole dataset before splitting, the mean/std it learns
would already "know" something about the test set's distribution — that's data
leakage, since in a real deployment you'd never have future data available when
building the scaler.

## Regression — Linear Regression

MSE ≈ 4,975,000,183 (this is in dollars-squared, so the huge raw number is
expected), R² ≈ 0.620 — the model explains about 62% of the variance in house
value.

Top 3 coefficients (on the scaled features):
| Feature | Coefficient |
|---|---|
| median_income | +70,849 |
| latitude | -55,482 |
| longitude | -55,025 |

A large positive coefficient (median_income) means: holding everything else
fixed, a one-standard-deviation increase in income is associated with about a
$70,849 increase in predicted house value. A large negative coefficient
(latitude, longitude) means moving one standard deviation further north or west
is associated with a drop in predicted value — this isn't really about compass
direction itself, it's picking up that certain coastal/southern regions (LA, SF
Bay Area) are pricier.

**Ridge (alpha=1.0) vs Linear Regression:**
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 4,975,000,183 | 0.6203 |
| Ridge | 4,974,716,809 | 0.6204 |

Ridge adds a penalty on the size of the coefficients (controlled by `alpha` — the
bigger alpha is, the more the coefficients get shrunk toward zero), which usually
helps when features are highly correlated with each other, since it stops the
model from putting huge, unstable weights on redundant features. Here the two
models come out almost identical, which tells me the OLS solution wasn't
unstable to begin with — there isn't much multicollinearity being fixed, and
`alpha=1.0` isn't strong enough to actually change much anyway.

## Classification — Logistic Regression (C=1.0)

Class balance before fitting: `y_clf_train` is about 50.06% / 49.94%, so no class
imbalance handling (SMOTE, class_weight) was needed — I checked this explicitly
before deciding, rather than skipping the check.

Confusion matrix:
```
[[1738  339]
 [ 360 1691]]
```
Accuracy 0.83, Precision 0.83, Recall 0.82, F1 0.83. **AUC = 0.911.**

Formulas:
- Precision = TP / (TP + FP) — of everything the model flagged as "high value,"
  how much actually was.
- Recall = TP / (TP + FN) — of everything that actually was "high value," how
  much the model caught.

For this task, I'd say **recall matters slightly more** than precision if the
client's use case is something like flagging promising high-value listings for
follow-up — missing a genuinely high-value listing (a false negative) is more
costly than flagging one that turns out to be borderline (a false positive),
since a human can always double-check a flagged listing but can't act on one
that was never surfaced.

AUC = 0.911 means: if you pick one random "high value" block and one random "low
value" block, the model ranks the high-value one higher about 91% of the time.
That's a strong separation between the two classes.

### Threshold sensitivity (0.30–0.70)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.752 | 0.906 | 0.822 |
| 0.40 | 0.791 | 0.872 | 0.829 |
| 0.50 | 0.833 | 0.824 | 0.829 |
| 0.60 | 0.873 | 0.772 | 0.819 |
| 0.70 | 0.902 | 0.697 | 0.786 |

F1 is actually maximised at **threshold 0.40**, barely above the default 0.50
(0.8294 vs 0.8287 — basically a tie). Given the recall-matters-more reasoning
above, I'd lower the threshold slightly from 0.5 toward 0.4: this trades a bit of
precision for a real gain in recall (0.824 → 0.872), which fits the "don't miss
promising listings" framing better than optimizing for the tightest possible
F1 alone.

## Regularization experiment (C=0.01 vs C=1.0)
| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 | 0.833 | 0.824 | 0.9106 |
| C=0.01 | 0.816 | 0.821 | 0.9065 |

`C` is the inverse of the regularization strength — a small `C` (like 0.01) means
a *stronger* penalty on large coefficients, forcing the model toward simpler,
smaller weights. Here, reducing `C` to 0.01 slightly **hurt** performance across
the board. That tells me the C=1.0 model wasn't overfitting badly in the first
place, so squeezing the coefficients harder just throws away real signal instead
of removing noise.

## Bootstrap confidence interval for the AUC difference
Drew 500 bootstrap resamples of the test set (with replacement), computed the AUC
of both models on each resample, and took the difference (C=1.0 minus C=0.01)
each time.

- Mean AUC difference: **0.0041**
- 95% CI: **[0.0014, 0.0066]**

The interval **excludes zero**, so even though the AUC gap is small in absolute
terms, it's consistent — C=1.0 reliably outperforms C=0.01 across different
resamples of the test data, not just by chance in this one split.

## Files
- `part2_models.py` — all code above, runs top to bottom on `cleaned_data.csv`
- `plot_roc.png` — ROC curve for the C=1.0 logistic regression model
