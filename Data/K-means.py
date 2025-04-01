# Abdikarim Jimale
# HW07-Clustering 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler   # for Scaling

data = pd.read_csv("/home/abdikarim/hw7-ajimale/Data/Spotify_Youtube.csv",sep=",")#reads in the data from a file
liveness = data["Liveness"]
energy = data["Energy"]
loudness = data["Loudness"]

# 3D points
x = [[liv, en, loud] for liv, en, loud in zip(liveness, energy, loudness)]

# Scaling feature 
scaler = StandardScaler()
x = scaler.fit_transform(x) # Normalize all features to same scale

# Elbow Method
inertia= []
K_range  = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(x)
    inertia.append(km.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for optimal K")
plt.show()

# K-Mean with optimal K=3
optimal_k = 3
km = KMeans(n_clusters=optimal_k, random_state=42)
y_km = km.fit_predict(x)  # Return a list of what data pointbelongs to what cluster

# Print results
print("Cluster distribution:", dict(zip(*np.unique(y_km, return_counts=True))))
print("Cluster Centers:\n", km.cluster_centers_)
print("Inertia (SSE):", km.inertia_)

# Visualize the clusters in 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection= "3d")
scatter = ax.scatter(x[:,0], x[:,1], x[:,2], c=y_km, cmap="viridis", s=50)
ax.set_xlabel("Livense (Scaled)")
ax.set_ylabel("Energy (Scaled)")
ax.set_zlabel("Loudness (Scaled)")
plt.colorbar(scatter, label="Cluster")
plt.show()

# Hierarchical Clustering 
plt.figure(figsize=(10,7))
linked = linkage(x, method="ward")
dendrogram(linked, orientation="top", distance_sort="descending")
plt.title("Dendrogram")
plt.xlabel("Songs")
plt.ylabel("Distance")
plt.show()

