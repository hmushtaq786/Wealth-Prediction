import optuna
import pandas as pd
import json
import glob
import os

# -----------------------------------------------------
# Parse model + indices from filename:
# Example: "efficientnet_mndwi.db" → model="efficientnet", indices=["mndwi"]
# Example: "efficientnet_mndwi_ndbi.db" → model="efficientnet", indices=["mndwi","ndbi"]
# -----------------------------------------------------
def parse_model_and_indices(filename):
    name = os.path.basename(filename).replace(".db", "")
    parts = name.split("_")

    model = parts[0]
    indices = parts[1:]  # everything after the model name
    return model, indices


# -----------------------------------------------------
# Recursively find ALL .db files in optuna/
# -----------------------------------------------------
db_files = glob.glob("../optuna/storage/multi/**/*.db", recursive=True)

records = []

for db in db_files:
    try:
        model, indices = parse_model_and_indices(db)

        storage = f"sqlite:///{db}"

        # Each DB contains exactly one study → find it
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        if not summaries:
            print(f"No studies inside DB: {db}")
            continue

        study_name = summaries[0].study_name
        study = optuna.load_study(study_name=study_name, storage=storage)

        best = study.best_trial

        record = {
            "model_name": model,
            "indices_used": ",".join(indices),
            "best_trial_number": best.number,
            "r2_score": best.value,
            "best_params": json.dumps(best.params)
        }

        records.append(record)

    except Exception as e:
        print(f"Error reading {db}: {e}")
        record = {
            "model_name": model,
            "indices_used": ",".join(indices),
            "best_trial_number": "-",
            "r2_score": "-",
            "best_params": "-"
        }
        records.append(record)


# -----------------------------------------------------
# Save output CSV
# -----------------------------------------------------
df = pd.DataFrame(records)
output_csv = "optuna_best_models_multi_summary.csv"
df.to_csv(output_csv, index=False)

print(f"Saved summary to {output_csv}")
