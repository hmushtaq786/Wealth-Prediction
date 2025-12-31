import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data (replace 'file_path' with the actual file path)
combined_data = pd.read_csv('model_results_combined_wealth.csv')

# Create a bar plot for R² scores by Model and Index
plt.figure(figsize=(14, 8))

sns.barplot(
    data=combined_data,
    x="Model",
    y="Test R²",
    hue="Index",
    # palette="tab20",
    ci=None  # Disable confidence intervals for cleaner visualization
)

plt.ylim(0.69, 0.74)  # Set a lower limit for the y-axis to start at 0

plt.title("R² Scores for Different Combined Indexes by Model (wealth)")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.legend(title="Indexes", loc="best")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
