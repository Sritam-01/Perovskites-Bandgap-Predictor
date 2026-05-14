import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pymatgen.core import Composition

from matminer.featurizers.composition import (
    ElementProperty
)


# PAGE CONFIG


st.set_page_config(
    page_title="Perovskite Bandgap Predictor",
    layout="centered"
)

st.title(
    "Perovskite Bandgap Prediction System"
)

st.write(
    "Predict bandgap of unknown perovskite materials using Machine Learning."
)


# LOAD MODELS


rf_model = joblib.load(
    "model/random_forest.pkl"
)

xgb_model = joblib.load(
    "model/xgboost.pkl"
)

feature_names = joblib.load(
    "model/features.pkl"
)


# MODEL PERFORMANCE

rf_r2 = 0.81
rf_mse = 0.12

xgb_r2 = 0.87
xgb_mse = 0.08


# FEATURIZER


featurizer = ElementProperty.from_preset(
    "magpie"
)


# USER INPUT


formula = st.text_input(
    "Enter Perovskite Formula",
    placeholder="Example: CsPbI3"
)


# PREDICT BUTTON


if st.button("Predict Bandgap"):

    try:

        # CREATE COMPOSITION
        composition = Composition(formula)

        # CREATE FEATURES
        features = featurizer.featurize(
            composition
        )

        # CONVERT TO DATAFRAME
        feature_df = pd.DataFrame(
            [features],
            columns=featurizer.feature_labels()
        )

        # KEEP TRAINED FEATURES ONLY
        feature_df = feature_df[
            feature_names
        ]

 
        # PREDICTIONS
        

        rf_prediction = rf_model.predict(
            feature_df
        )[0]

        xgb_prediction = xgb_model.predict(
            feature_df
        )[0]

        
        # BEST MODEL
        

        if xgb_r2 > rf_r2:

            best_model = "XGBoost"

            best_prediction = xgb_prediction

        else:

            best_model = "Random Forest"

            best_prediction = rf_prediction

        
        # DISPLAY RESULTS
        

        st.success(
            "Prediction Completed Successfully"
        )

        st.subheader("Model Predictions")

        st.write(
            f"Random Forest Prediction: "
            f"{rf_prediction:.3f} eV"
        )

        st.write(
            f"XGBoost Prediction: "
            f"{xgb_prediction:.3f} eV"
        )

        st.subheader("Best Model")

        st.write(
            f"Selected Model: {best_model}"
        )

        st.write(
            f"Predicted Bandgap: "
            f"{best_prediction:.3f} eV"
        )

        
        # METRICS
        

        st.subheader("Model Performance")

        metrics_df = pd.DataFrame({
            "Model": [
                "Random Forest",
                "XGBoost"
            ],
            "R2 Score": [
                rf_r2,
                xgb_r2
            ],
            "MSE": [
                rf_mse,
                xgb_mse
            ]
        })

        st.dataframe(metrics_df)

    except Exception as e:

        st.error(
            f"Invalid material formula: {e}"
        )