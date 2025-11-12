import joblib
import numpy as np
import matplotlib.pyplot as plt
import optuna

def smooth_curve(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='valid')

storage = "sqlite:///optuna/storage/resnet34.db"
study = optuna.load_study(study_name="resnet34_parallel", storage=storage)

scores_path = "optuna/scores/resnet34/trial"

plt.figure(figsize=(12, 6))
for trial in study.trials:
    trial_id = trial.user_attrs.get("slurm_id", trial.number)
    try:
        r2_scores = np.load(f"{scores_path}_{trial_id}_r2.npy")
        r2_scores = np.clip(r2_scores, 0, 1)  # Prevent outlier distortion
        smoothed = smooth_curve(r2_scores, window=2)  # Apply smoothing
        if np.mean(r2_scores) > 0:  # "successful" trial
            plt.plot(smoothed, alpha=0.9, label=f'Trial {trial_id}')
        else:  # faded failed trials
            plt.plot(smoothed, alpha=0.3, linestyle="--")
    except FileNotFoundError:
        print(f"⚠️ Skipping trial {trial_id} — no .npy file found.")

plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("R² per Epoch for Each Trial")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("optuna/plots/resnet34_1.png")
plt.close()

# Save the Optuna study
joblib.dump(study, "optuna/studies/resnet34.pkl")
print("✅ Plot and study saved successfully!")