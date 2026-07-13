# Applied AI & ML Capstone — End-to-End Data Intelligence System

## Dataset
California housing data (the "housing.csv" version — one row per census block
group from the 1990 census, 20,640 rows). I picked it because it's real housing
and location data rather than something abstract. Full justification is in each
part's own README, since the assignment asks for that in every part.

One repo, one folder per part, as allowed by the submission guidelines.

## Repo structure
```
.
├── README.md                 <- this file
├── part1_eda.py
├── README_part1.md
├── housing.csv                <- raw dataset
├── cleaned_data.csv           <- output of Part 1, used by Parts 2 and 3
├── plot_line.png
├── plot_bar.png
├── plot_hist.png
├── plot_scatter.png
├── plot_box.png
├── plot_heatmap.png
├── part2_models.py
├── README_part2.md
├── plot_roc.png
├── part3_ensembles.py
├── README_part3.md
├── best_model.pkl              <- serialized tuned Random Forest pipeline (Part 3)
├── part4_llm_feature.py
└── README_part4.md
```

## How to run
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib jsonschema requests

python3 part1_eda.py     # produces cleaned_data.csv + 6 plots
python3 part2_models.py  # produces plot_roc.png
python3 part3_ensembles.py  # produces best_model.pkl (~54MB, takes a few minutes - GridSearchCV)
python3 part4_llm_feature.py  # set MOCK_LLM = False and export GROQ_API_KEY first
```
Each script reloads `cleaned_data.csv` fresh and rebuilds the same train/test
split with `random_state=42`, so Parts 2, 3, and 4 are reproducible independent
of each other as long as Part 1 has been run first.

## Part-by-part summary

| Part | Topic | Marks | Status |
|---|---|---|---|
| 1 | Data cleaning + EDA | 25 | Done — see `README_part1.md` |
| 2 | Regression + classification models | 30 | Done — see `README_part2.md` |
| 3 | Ensembles, tuning, pipeline | 25 | Done — see `README_part3.md` |
| 4 | LLM-powered explanation feature (Track C) | 20 | Done — see `README_part4.md` |

## Overall takeaway
The tuned Random Forest from Part 3 (GridSearchCV, `max_depth=None,
n_estimators=200, min_samples_leaf=1`) was the strongest classifier overall
(5-fold CV AUC 0.956, test AUC 0.955), and it's the model Part 4 loads to
generate plain-language explanations of individual predictions for a
non-technical client.
