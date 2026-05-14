import pandas as pd
from pymatgen.core import Composition

# LOAD DATASET
df = pd.read_csv("dataset/full_materials_data.csv")

print("ORIGINAL DATASET SIZE:")
print(df.shape)


# FUNCTION TO CHECK PEROVSKITE-LIKE ABX3 MATERIALS
def is_perovskite_like(formula):

    try:
        comp = Composition(formula)

        # Must contain exactly 3 unique elements
        if len(comp.elements) != 3:
            return False

        # Get stoichiometric amounts
        amounts = sorted(comp.get_el_amt_dict().values())

        # Normalize smallest amount to 1
        smallest = amounts[0]
        normalized = [round(a / smallest) for a in amounts]

        # Check ABX3 ratio
        return normalized == [1, 1, 3]

    except:
        return False


# REMOVE NULL BAND GAP VALUES
df = df.dropna(subset=["band_gap"])

# KEEP ONLY BAND GAP > 0
df = df[df["band_gap"] > 0]

# FILTER PEROVSKITE MATERIALS
filtered_df = df[
    df["formula"].apply(is_perovskite_like)
]

print("\nFILTERED DATASET SIZE:")
print(filtered_df.shape)

print("\nSAMPLE MATERIALS:")
print(filtered_df.head(20))

# SAVE FILTERED DATASET
filtered_df.to_csv(
    "dataset/perovskites.csv",
    index=False
)

print("\nPEROVSKITE DATASET SAVED SUCCESSFULLY")