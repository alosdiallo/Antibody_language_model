import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read your data
similarity_matrix = pd.read_csv("Sequence_Similarity_Matrix_DNA.csv")
VDJ_Data = pd.read_csv("Immcantation_Clonotypes.csv")

# Convert the dataframe to numeric, errors='coerce' will replace non-numeric values with NaN
similarity_matrix = similarity_matrix.apply(pd.to_numeric, errors='coerce')

# Clip values greater than 1
similarity_matrix = similarity_matrix.clip(upper=1)

# Now perform the comparison
greater_than_one = similarity_matrix > 1
print("Number of values greater than 1 per column:")
print(greater_than_one.sum())


print("Starting UMAP")
# UMAP dimensionality reduction
reducer = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(similarity_matrix)
print("Finished UMAP and starting DBScan")

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusters = clusterer.fit_predict(embedding)
print("Finished DBScan")

# Add clusters and clonotype data to UMAP results
results = pd.DataFrame(embedding, columns=['x', 'y'])
results['cluster'] = clusters
results['clonotype'] = VDJ_Data['Clonotype']

print("Starting Plot clusters_plot.png")
# Visualization
sns.scatterplot(x='x', y='y', hue='cluster', data=results, palette='viridis', s=10)
plt.savefig('clusters_plot.png')

# Find top clonotype
top_clonotype = VDJ_Data['Clonotype'].value_counts().idxmax()

print("Starting Plot top_clonotype_plot.png")
# Plot with top clonotype
results_top = results[results['clonotype'] == top_clonotype]
sns.scatterplot(x='x', y='y', hue='cluster', data=results_top, palette='viridis', s=10)
plt.savefig('top_clonotype_plot.png')

# Find top 10 clonotypes
top_clonotypes = VDJ_Data['Clonotype'].value_counts().nlargest(10).index

print("Starting Plot top_10_clonotypes_plot.png")
# Plot with top 10 clonotypes
results_top10 = results[results['clonotype'].isin(top_clonotypes)]
sns.scatterplot(x='x', y='y', hue='clonotype', data=results_top10, palette='viridis', s=10)
plt.savefig('top_10_clonotypes_plot.png')
print("Done")
