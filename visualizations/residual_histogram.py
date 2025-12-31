import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

dhs_data_path = 'predictions_to_visualize.csv'
df = pd.read_csv(dhs_data_path)

# Compute residuals
df["residual"] = df["actual"] - df["predicted"]

skewness_value = skew(df["residual"])

print(f"Skewness of Residuals: {skewness_value:.4f}")
# Plot histogram
plt.figure(figsize=(8, 6))
sns.histplot(df["residual"], bins=30, kde=True, color="purple")
plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
plt.text(
    x=-1.90, 
    y=plt.gca().get_ylim()[1] * 0.9,  # Position near the top of the plot
    s=f"Skewness: {skewness_value:.4f}",
    fontsize=12, color="black", bbox=dict(facecolor="white", alpha=0.6)
)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.legend()
plt.grid(True)
plt.show()
