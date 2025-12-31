import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Load dataset
df = pd.read_csv("predictions_to_visualize.csv")

import seaborn as sns

# Compute residuals
df["residual"] = df["actual"] - df["predicted"]

# Plot histogram
plt.figure(figsize=(8, 6))
sns.histplot(df["residual"], bins=30, kde=True, color="purple")
plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.legend()
plt.grid(True)
plt.show()
