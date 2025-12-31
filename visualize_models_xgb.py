
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib
import os

gb_model_path = 'models/individual_index/wealthpooled/ndvi_gradient_boost_bins_10.pkl'

feature_names = ['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA']
gb = joblib.load(gb_model_path)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


tree_index = 0  # choose any tree
plt.figure(figsize=(20, 10))
plot_tree(
    gb.estimators_[tree_index][0],
    filled=True,
    rounded=True,
    max_depth=3  # show only first 3 levels
)
plt.show()