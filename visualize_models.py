from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib

model_path = 'models/individual_index/wealthpooled/ndvi_random_forest_bins_10.pkl'

feature_names = ['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA']
rf = joblib.load(model_path)
feature_names = [f"feature_{i}" for i in range(rf.n_features_in_)]
print(feature_names)
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], 
        feature_names=feature_names,
        class_names=True,
        filled=True,
        rounded=True,
        max_depth=3, 
        fontsize=8)
plt.show()

from sklearn.tree import export_graphviz
import graphviz

dot = export_graphviz(
    rf.estimators_[0],
    out_file=None,
    feature_names=feature_names,
    max_depth=3,
    filled=True,
    rounded=True
)

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

graph = graphviz.Source(dot)
graph.render("tree_visual", format="png", cleanup=True)