import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("predictions_to_visualize.csv")  # Replace with your dataset file

# Ensure dataset contains necessary columns
if not {"actual", "predicted", "URBAN_RURA"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'actual', 'predicted', and 'URBAN_RURA' columns.")

# Separate Urban and Rural data
urban_df = df[df["URBAN_RURA"] == "U"]
rural_df = df[df["URBAN_RURA"] == "R"]

# Compute R^2 scores
r2_urban = r2_score(urban_df["actual"], urban_df["predicted"])
r2_rural = r2_score(rural_df["actual"], rural_df["predicted"])
r2_overall = r2_score(df["actual"], df["predicted"])
print(f"Overall R²: {r2_overall:.2f}")

# Create a figure with joint density plots (Switched Axes)
g = sns.jointplot(
    data=df, 
    x="actual",  # Switched x to actual
    y="predicted",  # Switched y to predicted
    hue="URBAN_RURA", 
    palette={"U": "blue", "R": "red"},
    kind="scatter", 
    alpha=0.3, 
    marginal_kws={"fill": True}
)

# Add regression lines for urban and rural
sns.regplot(
    x=urban_df["actual"],  # Switched axes here
    y=urban_df["predicted"], 
    scatter=False, 
    color="darkblue", 
    ax=g.ax_joint
)
sns.regplot(
    x=rural_df["actual"],  # Switched axes here
    y=rural_df["predicted"], 
    scatter=False, 
    color="darkred", 
    ax=g.ax_joint
)

# Add text for R² values
g.ax_joint.text(1.15, -1.1, f"$r^2$ (Urban) = {r2_urban:.2f}", color="darkblue", fontsize=10)
g.ax_joint.text(1.15, -1.3, f"$r^2$ (Rural) = {r2_rural:.2f}", color="darkred", fontsize=10)

# Set Labels and Title
g.set_axis_labels("Actual Wealthpooled Index", "Predicted Wealthpooled Index")  # Updated labels

# Correctly position the title at the top of the figure
# plt.subplots_adjust(top=0.90)  # Adjust spacing to prevent overlap
# g.fig.suptitle("Urban vs. Rural Wealthpooled Prediction Performance", fontsize=12)

plt.show()
