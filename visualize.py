# 7. FULL DATASET PROPERTY VISUALIZATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET
# ---------------------------------
df = pd.read_csv("dataset/materials_data.csv")

# MATERIAL INDEX
material_index = range(len(df))

properties = [
    'band_gap',
    'density',
    'energy_per_atom',
    'formation_energy_per_atom',
    'volume'
]

for prop in properties:

    plt.figure(figsize=(14,6))

    plt.plot(
        material_index,
        df[prop],
        linewidth=0.5
    )

    plt.title(f"All Materials vs {prop}")

    plt.xlabel("Material Index")

    plt.ylabel(prop)

    plt.tight_layout()

    plt.savefig(f"plots/all_materials_{prop}.png")

    plt.show()
    