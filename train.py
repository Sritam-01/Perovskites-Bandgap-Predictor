import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

from xgboost import XGBRegressor

import joblib

# LOAD DATASET
df = pd.read_csv(
    "dataset/featurized_perovskites.csv"
)

# REMOVE MISSING
df = df.dropna()

# TARGET
y = df["band_gap"]

# FEATURES
X = df.drop(
    "band_gap",
    axis=1
)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# RANDOM FOREST


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)

rf_mse = mean_squared_error(
    y_test,
    rf_pred
)

print("\nRANDOM FOREST")
print("R2:", rf_r2)
print("MSE:", rf_mse)


# XGBOOST


xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_r2 = r2_score(y_test, xgb_pred)

xgb_mse = mean_squared_error(
    y_test,
    xgb_pred
)

print("\nXGBOOST")
print("R2:", xgb_r2)
print("MSE:", xgb_mse)


# BEST MODEL


if xgb_r2 > rf_r2:

    best_model = xgb
    best_name = "XGBoost"

else:

    best_model = rf
    best_name = "Random Forest"

print("\nBEST MODEL:")
print(best_name)

# SAVE MODEL
joblib.dump(
    best_model,
    "model/best_model.pkl"
)

# SAVE BOTH MODELS
joblib.dump(
    rf,
    "model/random_forest.pkl"
)

joblib.dump(
    xgb,
    "model/xgboost.pkl"
)

# FEATURE NAMES
joblib.dump(
    X.columns.tolist(),
    "model/features.pkl"
)


# FEATURE IMPORTANCE


importance = xgb.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# PLOT
plt.figure(figsize=(12,8))

plt.barh(
    importance_df["Feature"][:20],
    importance_df["Importance"][:20]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 20 Feature Importances"
)

plt.tight_layout()

plt.savefig(
    "model/feature_importance.png"
)

print("\nMODEL TRAINING COMPLETE")