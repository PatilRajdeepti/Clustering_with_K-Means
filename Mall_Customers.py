# kmeans_mall.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Step 1: Load Dataset
data = pd.read_csv("Mall_Customers.csv")
print("First 5 rows:\n", data.head())

# Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Optional: PCA for 2D visualization (only if >2 features)
# pca = PCA(n_components=2)
# X = pca.fit_transform(X)

# Step 2: Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# Step 3: Fit KMeans with optimal K (say 5)
k = 5
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

# Step 4: Visualize Clusters
plt.figure(figsize=(6,4))
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=labels, palette='Set2')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()

# Step 5: Evaluate using Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.2f}")
