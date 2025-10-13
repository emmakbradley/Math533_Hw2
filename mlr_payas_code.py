### PART 1 : LOADING & VALIDATING DATA ### 


import pandas as pd
import numpy as np
from pathlib import Path

# config
csv_path = Path("alzheimer_data.csv")
random_seed = 42

# load
df = pd.read_csv(csv_path)

# build target safely
df = df.copy()
df["target_ratio"] = df["lhippo"] / df["naccicv"]

# exact predictors from the prompt (matching your columns)
predictors = [
    # cognitive and behavioral
    "naccmmse", "motsev", "disnsev", "anxsev", "naccgds",
    # physiological
    "bpsys", "bpdias", "hrate",
    # demographics
    "age", "educ", "female", "height", "weight",
]

# keep only the columns we need
cols = ["target_ratio"] + predictors
df = df.loc[:, cols].copy()

# show quick shape and missingness
print("rows x cols:", df.shape)
print("\nmissing counts:\n", df.isna().sum().sort_values(ascending=False))


### PART 2 : BUILDING AND EVALUATING THE MODEL ### 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import (
    train_test_split, KFold, LeaveOneOut, cross_val_score
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------
# 1. load and prepare data

df = pd.read_csv("alzheimer_data.csv")

# target: left hippocampus / intracranial volume
y = df["lhippo"] / df["naccicv"]

# predictors (saturated set)
predictors = [
    "naccmmse","motsev","disnsev","anxsev","naccgds",
    "bpsys","bpdias","hrate","age","educ","female","height","weight"
]
X = df[predictors]

# ------------------------------------------------------------
# 2. helper function to compute cv metrics

def evaluate_model(model, X, y, name="model"):
    results = []
    
    # single train/test split (80/20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    results.append({
        "model": name,
        "cv_type": "single split (80/20)",
        "rmse_mean": mean_squared_error(yte, ypred, squared=False),
        "rmse_std": np.nan,
        "r2_mean": r2_score(yte, ypred),
        "r2_std": np.nan
    })

    # 5-fold
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse5 = -cross_val_score(model, X, y, cv=kf5,
                             scoring="neg_root_mean_squared_error")
    r2_5 = cross_val_score(model, X, y, cv=kf5, scoring="r2")
    results.append({
        "model": name,
        "cv_type": "5-fold",
        "rmse_mean": rmse5.mean(),
        "rmse_std": rmse5.std(),
        "r2_mean": r2_5.mean(),
        "r2_std": r2_5.std()
    })

    # 10-fold
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse10 = -cross_val_score(model, X, y, cv=kf10,
                              scoring="neg_root_mean_squared_error")
    r2_10 = cross_val_score(model, X, y, cv=kf10, scoring="r2")
    results.append({
        "model": name,
        "cv_type": "10-fold",
        "rmse_mean": rmse10.mean(),
        "rmse_std": rmse10.std(),
        "r2_mean": r2_10.mean(),
        "r2_std": r2_10.std()
    })

    # leave-one-out (r² not defined)
    loo = LeaveOneOut()
    rmse_loocv = -cross_val_score(model, X, y, cv=loo,
                                  scoring="neg_root_mean_squared_error")
    results.append({
        "model": name,
        "cv_type": "LOOCV",
        "rmse_mean": rmse_loocv.mean(),
        "rmse_std": rmse_loocv.std(),
        "r2_mean": np.nan,
        "r2_std": np.nan
    })

    return pd.DataFrame(results)

# ------------------------------------------------------------
# 3. saturated model

saturated = LinearRegression()
res_saturated = evaluate_model(saturated, X, y, name="saturated")

# ------------------------------------------------------------
# 4. lasso feature selection

# automatically tunes alpha with 10-fold CV
lasso = LassoCV(cv=10, random_state=42)
lasso.fit(X, y)

# extract selected features (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0].tolist()
print("\nselected features by LASSO:")
print(selected_features)

# reduced dataset
X_reduced = X[selected_features]

# ------------------------------------------------------------
# 5. reduced model (OLS on selected features)

reduced = LinearRegression()
res_reduced = evaluate_model(reduced, X_reduced, y, name="reduced (lasso-selected)")

# ------------------------------------------------------------
# 6. combine results into one comparison table

results_all = pd.concat([res_saturated, res_reduced], ignore_index=True)
pd.set_option("display.precision", 6)
print("\n=== model comparison table ===")
print(results_all)

# ------------------------------------------------------------
# 7. optional: simple interpretation summary
best_reduced = results_all.query("model == 'reduced (lasso-selected)' and cv_type == '10-fold'")
best_saturated = results_all.query("model == 'saturated' and cv_type == '10-fold'")

if not best_reduced.empty and not best_saturated.empty:
    rm_diff = best_saturated.rmse_mean.values[0] - best_reduced.rmse_mean.values[0]
    r2_diff = best_reduced.r2_mean.values[0] - best_saturated.r2_mean.values[0]
    print("\n=== summary ===")
    print(f"lasso reduced model improves rmse by {rm_diff:.6f} "
          f"and r² by {r2_diff:.3f} relative to saturated model (10-fold CV).")
    print("→ prefer reduced model if accuracy is similar or slightly better, "
          "since it is simpler and more interpretable.")







