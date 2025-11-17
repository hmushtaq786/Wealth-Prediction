import os
import numpy as np
import matplotlib.pyplot as plt
import optuna

def smooth_curve(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='valid')

def main():
    models = ["efficientnet", "resnet", "vgg"]
    indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True)
    colors = plt.cm.tab10.colors

    for model_idx, model in enumerate(models):
        ax = axs[model_idx]
        color_idx = 0

        for index in indices:
            study_name = f"{model}_{index}"
            db_path = f"sqlite:///optuna/storage/single/{model}/{study_name}.db"
            scores_dir = f"optuna/r2_scores/single/{model}/{index}"

            try:
                study = optuna.load_study(study_name=study_name, storage=db_path)
            except Exception as e:
                print(f"❌ Could not load study {study_name}: {e}")
                continue

            # Get best trial
            best_trial = study.best_trial
            trial_id = best_trial.user_attrs.get("slurm_id", best_trial.number)

            npy_path = os.path.join(scores_dir, f"trial_{trial_id}.npy")
            if not os.path.exists(npy_path):
                print(f"⚠️ Skipping {model}_{index} (missing {npy_path})")
                continue

            r2_scores = np.load(npy_path)
            r2_scores = np.clip(r2_scores, 0, 1)
            smoothed = smooth_curve(r2_scores, window=2)

            ax.plot(smoothed, label=index.upper(), color=colors[color_idx % len(colors)])
            color_idx += 1

        ax.set_title(f"{model.upper()} – Best Trial R² Curves")
        ax.set_ylabel("R² Score")
        ax.grid(True)
        ax.legend(title="Index")

    axs[-1].set_xlabel("Epoch")
    plt.tight_layout()

    os.makedirs("optuna/plots/single", exist_ok=True)
    plot_path = "optuna/plots/single/best_trials_comparison_by_model.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot to {plot_path}")

if __name__ == "__main__":
    main()


## to generate grouped bar plot of final R² scores for best trials (commented out) ##

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import optuna

# def load_best_r2(model, index):
#     """Load best R² for given model-index combo."""
#     study_name = f"{model}_{index}"
#     db_path = f"sqlite:///optuna/storage/single/{model}/{study_name}.db"
#     scores_dir = f"optuna/r2_scores/single/{model}/{index}"

#     try:
#         study = optuna.load_study(study_name=study_name, storage=db_path)
#         best_trial = study.best_trial
#         trial_id = best_trial.user_attrs.get("slurm_id", best_trial.number)
#         npy_path = os.path.join(scores_dir, f"trial_{trial_id}.npy")

#         if not os.path.exists(npy_path):
#             print(f"⚠️ Missing file: {npy_path}")
#             return None

#         r2_scores = np.load(npy_path)
#         r2_scores = np.clip(r2_scores, 0, 1)
#         return r2_scores[-1]

#     except Exception as e:
#         print(f"❌ Error loading {study_name}: {e}")
#         return None

# def main():
#     models = ["efficientnet", "resnet", "vgg"]
#     indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]
#     display_names = {
#         "efficientnet": "EfficientNet",
#         "resnet": "ResNet34",
#         "vgg": "VGG16"
#     }

#     # Collect data
#     data = []
#     for model in models:
#         for index in indices:
#             r2 = load_best_r2(model, index)
#             if r2 is not None:
#                 data.append({
#                     "Model": display_names[model],
#                     "Index": index.upper(),
#                     "R2": r2
#                 })

#     import pandas as pd
#     df = pd.DataFrame(data)

#     # Plot
#     plt.figure(figsize=(12, 6))
#     sns.set(style="whitegrid")

#     ax = sns.barplot(
#         data=df,
#         x="Index", y="R2", hue="Model",
#         palette="Set2"
#     )
#     ax.set_title("Final R² Scores of Best Trials (Grouped by Model)")
#     ax.set_ylabel("Final R²")
#     ax.set_ylim(0, 1)
#     ax.legend(title="Model")
#     plt.tight_layout()

#     os.makedirs("optuna/plots/single", exist_ok=True)
#     plt.savefig("optuna/plots/single/grouped_final_r2_barplot.png", dpi=300)
#     plt.close()

#     print("✅ Saved grouped bar plot to optuna/plots/single/grouped_final_r2_barplot.png")

# if __name__ == "__main__":
#     main()
