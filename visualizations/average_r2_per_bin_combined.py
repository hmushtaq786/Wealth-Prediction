import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace 'file_path' with the actual file path)
data = pd.read_csv('model_results_combined_wealth.csv')
data1 = pd.read_csv('model_results_combined_wealthpooled.csv')

# Group data by 'Bins' and calculate the average R² score for each bin
grouped_bins = data.groupby("Index")["Test R²"].mean().reset_index()
grouped_bins1 = data1.groupby("Index")["Test R²"].mean().reset_index()

# Visualize the average R² score for each bin
plt.figure(figsize=(10, 6))
plt.plot(grouped_bins["Index"], grouped_bins["Test R²"], marker='o', label='Wealth')
plt.plot(grouped_bins1["Index"], grouped_bins1["Test R²"], marker='o', label='Wealthpooled')
plt.title("Average R² Score by Number of Bins")
plt.xlabel("Number of Bins")
plt.ylabel("Average R² Score")
plt.legend(title="Ground truth")
plt.grid()
plt.show()

# Save the processed data to a CSV file (optional)
# grouped_bins.to_csv('average_r2_per_bin.csv', index=False)
