# Part 3 — Ensembles, Tuning, and Full ML Pipeline

Reuses the exact same preprocessing and 80/20 split as Part 2 (`random_state=42`),
rebuilt at the top of this script so it can be graded on its own.

## Decision Tree baseline vs controlled tree

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Unconstrained (`max_depth=None`) | 1.0000 | 0.8556 | 0.1444 |
| Controlled (`max_depth=5, min_samples_split=20`) | 0.8287 | 0.8152 | 0.0136 |

The unconstrained tree gets 100% training accuracy — it's basically memorized the
training set. Decision trees are called high-variance models because they build
splits greedily, chasing whatever reduces impurity most at each node, without ever
going back to reconsider an earlier split. Left unconstrained, they'll keep
splitting until every training example sits in its own tiny leaf, which fits noise
as much as signal — a small change in the training data can produce a completely
different tree. That's exactly what the huge train/test gap shows here.

`max_depth` limits how many splits deep the tree can go, which directly caps how
finely it can carve up the training data — this trades a bit of bias (it can't
fit every training quirk anymore) for a lot less variance. `min_samples_split`
stops the tree from splitting a node that has fewer than that many samples, which
prevents it from making a split based on just a handful of possibly-noisy points.
Together they shrink the train/test gap from 0.144 down to 0.014.

## Gini vs Entropy
Both trained with `max_depth=5`.

- Gini impurity: `1 - Σ p_i²`
- Entropy: `-Σ p_i * log2(p_i)`

Gini test accuracy: 0.8152, Entropy test accuracy: 0.8147 — basically a tie, which
is typical; the two criteria usually produce very similar trees. A node with
Gini = 0 means every sample in that node belongs to the same class — the node is
"pure," so there's nothing left to split on.

## Random Forest
`n_estimators=100, max_depth=10`. Train acc 0.9145, test acc 0.8702, test AUC
**0.9444**.

Top 5 features by importance:
| Feature | Importance |
|---|---|
| median_income | 0.263 |
| ocean_proximity_INLAND | 0.194 |
| income_category | 0.157 |
| longitude | 0.112 |
| latitude | 0.107 |

Random Forest computes feature importance by averaging, across every tree and
every split in the forest, how much that feature reduced Gini impurity when it
was used. This is different from a linear regression coefficient: a coefficient
tells you the size and direction of a feature's effect assuming a straight-line
relationship, while a Random Forest importance score just tells you how useful a
feature was for splitting the data, with no direction and no assumption of
linearity — it can pick up on complex, non-linear patterns a coefficient can't.

Bagging, in one paragraph: each tree in the forest is trained on a bootstrap
sample — a random sample of rows drawn with replacement from the training data,
so some rows appear multiple times and others not at all in any given tree. On
top of that, at every split, only a random subset of √(number of features) is
even considered, not the whole feature set. Because each tree ends up a little
different from the others (different data, different candidate features), their
individual mistakes tend not to line up — averaging their predictions cancels out
a lot of that individual noise, which is why the forest as a whole has much lower
variance than any single deep tree, even though each tree on its own might still
overfit.

## Gradient Boosting
`n_estimators=100, learning_rate=0.1, max_depth=3`. Train acc 0.8812, test acc
0.8699, test AUC **0.9442** — essentially tied with Random Forest.

## Feature ablation study
Removed the 5 lowest-importance features from the Random Forest
(`total_bedrooms`, `households`, `ocean_proximity_NEAR OCEAN`,
`ocean_proximity_NEAR BAY`, `ocean_proximity_ISLAND`):

- Full model test AUC: 0.9444
- Reduced model test AUC: 0.9441
- Drop: 0.0003 (essentially nothing)

That tiny a drop means these five features were genuinely close to uninformative
for this model — removing them barely changed predictive power. In production,
that's a real argument for shipping the simpler, 8-feature model instead of the
13-feature one: fewer features means lower inference cost and less to maintain
(fewer things that can break if a data source changes upstream), and that
trade-off is only worth it because the AUC cost here is negligible. If the drop
had been, say, 0.02+, I'd have kept all the features — the cost of a meaningfully
worse model isn't worth a marginal efficiency gain.

## Cross-validated comparison (5-fold, StratifiedKFold, ROC-AUC)
| Model | CV mean AUC | CV std |
|---|---|---|
| Logistic Regression | 0.9199 | 0.0024 |
| Decision Tree (depth=5) | 0.9045 | 0.0024 |
| Random Forest | 0.9461 | 0.0030 |
| Gradient Boosting | 0.9475 | 0.0026 |

A single train/test split gives you one number, and that number depends partly on
which rows happened to land in the test set — a "lucky" or "unlucky" split can
make a model look better or worse than it really is. Cross-validation runs the
same model on 5 different train/test partitions and averages the result, which
gives a much more stable estimate of how the model would actually perform on new
data, plus a standard deviation that tells you how much that estimate might swing.

## Hyperparameter tuning — GridSearchCV
Grid: `n_estimators` [50, 100, 200] × `max_depth` [5, 10, None] ×
`min_samples_leaf` [1, 5] = **18 configurations, × 5 folds = 90 total fits**.

Best params: `max_depth=None, min_samples_leaf=1, n_estimators=200`.
Best CV AUC: **0.9560**.

Exhaustive Grid Search tries every single combination in the grid, so it's
guaranteed to find the best combination *within that grid* — but the cost grows
multiplicatively with every parameter you add, which gets expensive fast.
Randomized Search instead samples a fixed number of random combinations from the
grid (or from a distribution), which is much cheaper and in practice finds a
result close to the grid search optimum most of the time, especially when only a
couple of the hyperparameters actually matter much — which is often the case.

## Manual learning curve (best pipeline, 5 training-set fractions)
| Training fraction | Rows | Training AUC | Test AUC |
|---|---|---|---|
| 20% | 3,302 | 1.0000 | 0.9387 |
| 40% | 6,604 | 1.0000 | 0.9464 |
| 60% | 9,907 | 1.0000 | 0.9502 |
| 80% | 13,209 | 1.0000 | 0.9526 |
| 100% | 16,512 | 1.0000 | 0.9550 |

Note: the tasks reference `X_test_scaled` here, but the *tuned* pipeline includes
its own internal `StandardScaler`, so I passed the raw (unscaled) `X_test` into
it — scaling it a second time beforehand would have double-scaled the features
and given wrong predictions. The fixed test set itself (same rows) is unchanged.

Training AUC stays pinned at 1.0 at every fraction — the tuned Random Forest
(`max_depth=None`) can still perfectly fit whatever training subset it's given,
so this isn't shrinking as expected for a typical high-variance model, meaning it
would probably benefit from a slight depth cap in a production setting despite
its very strong test AUC. Test AUC keeps climbing at every step, all the way to
100% of the data, with no sign of flattening out yet. My conclusion: the model is
currently **limited by data quantity, not by model capacity** — collecting more
training rows would likely push test AUC higher still, since it hasn't plateaued.

## Serialization
Saved the best pipeline to `best_model.pkl` (~54 MB, under the 100 MB limit, so
it's committed directly rather than regenerated by script). Reloaded it and ran
`.predict()` on 2 real test rows to confirm it works after a fresh load —
predictions and probabilities matched what the in-memory model gave before
saving.

## Summary comparison table
| Model | 5-fold CV mean AUC | 5-fold CV std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.9199 | 0.0024 | 0.9106 |
| Decision Tree (depth=5) | 0.9045 | 0.0024 | 0.8992 |
| Random Forest | 0.9461 | 0.0030 | 0.9444 |
| Gradient Boosting | 0.9475 | 0.0026 | 0.9442 |
| Tuned RF (GridSearchCV) | 0.9560 | -- | 0.9550 |

**Recommendation:** the tuned Random Forest from GridSearchCV. It has the highest
CV mean AUC and the highest test AUC of everything tried, and cross-validation
confirms that edge isn't a fluke from one lucky split. Gradient Boosting is close
behind and would be a reasonable second choice, but given the learning curve
showed the Random Forest still improving with more data and no sign of
overfitting getting worse, I'd ship the tuned Random Forest to the client and
suggest collecting more data down the line if even higher accuracy is needed.

## Files
- `part3_ensembles.py` — all code above, runs top to bottom
- `best_model.pkl` — the tuned, serialized Random Forest pipeline
