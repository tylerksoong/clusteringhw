from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def pipeline(X, k,distance_metric="euclidean", linkage_metric="ward"): 
    #perform k_means and linkage on the data set
    k_model = KMeans(n_clusters=k)
    km_model = KMedoids(n_clusters=k)
    Z = linkage(X, method=linkage_metric, metric=distance_metric)

    k_model.fit(X)
    km_model.fit(X)

    #preparing output data with new cluster labels.
    df = pd.DataFrame(X)
    df['kmeans'] = k_model.labels_
    df['kmedoids'] = km_model.labels_
    df['hierarchal'] = np.array(fcluster(Z, t=k, criterion='maxclust')) - 1

    # Convert the relevant columns to a NumPy array for speed
    cluster_data = df[['kmeans', 'kmedoids', 'hierarchal']].sort_values('kmeans').values

    n_samples = cluster_data.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))


    for col_idx in range(cluster_data.shape[1]):
        # by turn
        labels = cluster_data[:, col_idx]
        consensus_matrix += (labels[:, None] == labels[None, :])

    consensus_matrix /= cluster_data.shape[1]

    #Get intertia values for k 2-10
    inertia_vals = []
    silhouette_vals = []
    for k in range(2,11):
        k_model = KMeans(k)
        k_model.fit(X)
        inertia_vals.append(k_model.inertia_)
        silhouette_vals.append(silhouette_score(X, k_model.labels_))


    fig, axes = plt.subplots(1,3)

    axes[0].plot(list(range(2,11)), inertia_vals)
    
    dendrogram(
        Z,
        ax=axes[1]
    )

    axes[2].imshow(consensus_matrix)

    plt.show()
    return df
