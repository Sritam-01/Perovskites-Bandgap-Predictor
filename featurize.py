import pandas as pd

from pymatgen.core import Composition

from matminer.featurizers.composition import (
    ElementProperty
)

def main():

    # LOAD DATA
    df = pd.read_csv(
        "dataset/perovskites.csv"
    )

    # REMOVE MISSING
    df = df.dropna()

    print(df.head())

    # CONVERT FORMULA
    df["composition"] = df["formula"].apply(
        Composition
    )

    # FEATURIZER
    featurizer = ElementProperty.from_preset(
        "magpie"
    )

    # CREATE FEATURES
    features = featurizer.featurize_dataframe(
        df,
        col_id="composition",
        ignore_errors=True
    )

    # # KEEP NUMERIC
    # features = features.select_dtypes(
    #     include=['float64', 'int64']
    # )
    
    # KEEP NUMERIC
    features = features.select_dtypes(
        include=['float64', 'int64']
    )

    # REMOVE NON-GENERALIZABLE FEATURES
    columns_to_remove = [
        "density",
        "energy_per_atom",
        "formation_energy_per_atom",
        "volume"
    ]

    features = features.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    
    
    # ADD TARGET
    features["band_gap"] = df["band_gap"]

    print("\nFEATURED DATASET:")
    print(features.head())

    print("\nDATASET SHAPE:")
    print(features.shape)

    # SAVE
    features.to_csv(
        "dataset/featurized_perovskites.csv",
        index=False
    )

    print("\nFEATURED DATASET SAVED")

# IMPORTANT WINDOWS FIX
if __name__ == "__main__":
    main()