from mp_api.client import MPRester
import pandas as pd

API_KEY = "pGrctzX97uO4icM3SUnZUNV3dLggJNpu"

data = []

print("CONNECTING TO MATERIALS PROJECT...")

with MPRester(API_KEY) as mpr:

    docs = mpr.materials.summary.search(
        band_gap=(0, 10),
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "density",
            "energy_per_atom",
            "formation_energy_per_atom",
            "volume"
        ]
    )

    print("FETCHING DATA...")

    count = 0

    for doc in docs:

        try:

            data.append({
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "density": doc.density,
                "energy_per_atom":
                    doc.energy_per_atom,
                "formation_energy_per_atom":
                    doc.formation_energy_per_atom,
                "volume": doc.volume
            })

            count += 1

            # PROGRESS UPDATE
            if count % 1000 == 0:

                print(
                    f"{count} materials fetched..."
                )

        except Exception as e:

            print(
                f"ERROR AT MATERIAL {count}:",
                e
            )

# CREATE DATAFRAME
df = pd.DataFrame(data)

print("\nFINAL DATASET SHAPE:")
print(df.shape)

# SAVE CSV
df.to_csv(
    "dataset/full_materials_data.csv",
    index=False
)

print("\nFULL DATASET SAVED")