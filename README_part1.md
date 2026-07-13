# Part 1 — Data Acquisition, Cleaning, and EDA

## Dataset

I used the California housing dataset (the "housing.csv" one — 20,640 rows, one row
per census block group from the 1990 census, with columns like location, income,
rooms, population, and house value). I picked it because it has real housing and
location data, not something abstract or made up, so the numbers actually mean
something when you look at them.

It has 8 numeric columns plus the target (median_house_value), and a categorical
column, ocean_proximity, with 5 values (INLAND, NEAR BAY, etc.). That's only one
categorical column though, so I made a second one myself: income_category, splitting
median_income into four buckets (Low/Medium/High/VeryHigh) by quartile. This one has
a natural order (Low < Medium < High < VeryHigh), so it's ordinal, while
ocean_proximity has no order, so it's nominal.

One thing worth being upfront about: the raw file is pretty clean. Only one column
(total_bedrooms) had missing values, and there were no duplicate rows at all. Since
this project is supposed to demonstrate handling messy data, I added some missing
values on purpose — to total_rooms (6%) and households (25%, deliberately above the
20% cutoff so that part of the task actually gets tested) — plus 60 duplicate rows.
All done with a fixed random seed (42) so it's reproducible if you rerun the script.
I did this because otherwise there wasn't enough messiness in the data to actually
practice cleaning on.

## What I did, step by step

1. Loaded the CSV, checked the shape (20,640 rows, 10 columns before I added the
   income category), and printed dtypes and the first 5 rows.
2. Checked nulls in every column. After my injected missingness, three columns had
   nulls: households (~25%), total_rooms (~6%), total_bedrooms (~1.3%, the real one).
   households was above the 20% cutoff, the other two were below. I filled all of
   them with the column median.
   - Why median and not mean: several of these columns (population, total_rooms,
     households) are heavily right-skewed — a few really dense or really big block
     groups pull the mean way up. The median isn't affected by those outliers the
     same way, so it's a more honest "typical" value to fill in.
3. Found 60 duplicate rows (the ones I added) and dropped them with
   `drop_duplicates()`. Checked whether this changed the null percentages — it
   didn't, since the duplicated rows didn't happen to be the ones with nulls.
4. For the dtype task, I forced housing_median_age into a string column on purpose
   (to simulate a badly-typed CSV column, which happens a lot in real data) and then
   converted it back to numeric with `pd.to_numeric`. I also converted
   ocean_proximity and income_category to the `category` dtype, which cut memory
   usage from about 2622 KB to about 1492 KB — categories store the values as small
   integer codes internally instead of repeating the same strings over and over.
5. Ran `.describe()` and checked skewness for every numeric column. The most skewed
   column is **population**, with a skew of about 4.94.
   - What that means: population has a long right tail — most block groups have a
     "normal" population, but a handful have way more, dragging the distribution's
     tail out to the right. If I used the mean to fill missing values here, it would
     end up way higher than what's typical for most rows, because a few huge values
     drag the mean up. The median is a much safer choice for this column.
6. Ran an IQR outlier check on median_income and total_rooms:
   - median_income: 681 rows fall outside [-0.71, 8.01]. Since income can't be
     negative, this basically means all the flagged outliers are on the high end —
     a chunk of unusually wealthy block groups.
   - total_rooms: 1,391 rows fall outside [-863.12, 5397.88]. Same story — the low
     bound is impossible (can't have negative rooms), so every flagged row is a
     block group with an unusually large number of total rooms. I didn't drop any
     of these; I'm keeping them for now and deciding in Part 2 whether they need
     capping, since removing them could throw away real signal about denser or
     bigger neighborhoods.
7. Made all five required plots:
   - Line plot of median_house_value for the first 300 rows (mostly just shows the
     data isn't sorted by any meaningful order — it jumps around).
   - Bar chart of mean house value by income category — a clear, expected step up
     from Low to VeryHigh.
   - Histogram of population (the most skewed column) — strongly right-skewed, most
     values bunched on the low end with a long thin tail stretching right.
   - Scatter plot of median_income vs median_house_value — a fairly clear positive
     trend, though there's a lot of spread, and you can see a flat line at the very
     top where house values are capped at $500,001 in this dataset.
   - Box plot of median_house_value by ocean_proximity — INLAND is clearly lower and
     tighter than the other groups; ISLAND is high but that's only 5 rows so I
     wouldn't read too much into it.
8. Correlation heat map of all numeric columns. The strongest correlation by far is
   longitude vs latitude, at about -0.92. That's not really a meaningful causal
   relationship — it's just geography. California's coastline runs on a diagonal
   (northwest to southeast), so as you move west (more negative longitude) you also
   tend to move north (higher latitude) if you're following the coast. A more useful
   "real" correlation for modeling is median_income vs median_house_value, at about
   0.69.
9. a. **Mean vs median for the two most-skewed columns** (population and
   total_rooms): for both, I used the median for imputation, because both are
   right-skewed (positive skew), which means the mean gets pulled upward by a
   handful of extreme high values and stops being representative of a "typical"
   row. The median stays anchored near where most of the data actually sits.
   b. **Spearman vs Pearson.** The three column pairs where Spearman and Pearson
   disagree the most are: total_rooms vs median_income (diff ≈ 0.072), 
   median_house_value vs total_rooms (diff ≈ 0.072), and longitude vs latitude
   (diff ≈ 0.046). In all three cases Spearman is a bit higher than Pearson, which
   suggests the relationship is monotonic but not perfectly linear — the variables
   move together consistently, just not at a constant rate. For example, more
   rooms tends to go with higher income, but probably not in a straight-line way at
   the extremes. Since none of these gaps are huge, I'll mostly use Pearson for
   feature-selection guidance in Part 2, but I'll keep Spearman in mind as a
   sanity check for these specific pairs.
   c. **Grouped aggregation** (ocean_proximity → median_house_value): ISLAND has
   the highest mean (~$380,440) but it's only 5 rows, so I don't trust it much.
   NEAR BAY has the highest standard deviation (~$122,819), meaning house values in
   that group are all over the place — that's a sign ocean_proximity alone doesn't
   fully explain price for NEAR BAY listings; there's a lot of extra variation a
   model would need other features to capture. The ratio between the highest and
   lowest group means (ISLAND vs INLAND) is about 3.05, which is large enough that
   ocean_proximity clearly does carry some real predictive signal, even if it's not
   the whole story.
10. Saved the final cleaned dataset to `cleaned_data.csv` (20,640 rows, 11 columns
    including the new income_category column), which Parts 2 and 3 load from.

## Files in this folder
- `part1_eda.py` — all the code above, runs top to bottom with no errors
- `housing.csv` — the raw dataset
- `cleaned_data.csv` — the cleaned output
- `plot_line.png`, `plot_bar.png`, `plot_hist.png`, `plot_scatter.png`, `plot_box.png`,
  `plot_heatmap.png` — the six visualizations
