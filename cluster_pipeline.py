from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

def pipeline(X, k,distance_metric="euclidean", linkage_metric="ward"): 
    k_model = KMeans(n_clusters=k)
    Z = linkage(X, method=linkage_metric, metric=distance_metric)

    k_model.fit(X)

    df = pd.DataFrame(X)
    df['cluster_labels'] = k_model.labels_
    df['hierarchal_labels'] = fcluster(Z, t=k, criterion='maxclust')
    print(df.head())

    fig, axes = plt.subplots(1,2)

    axes[0].scatter(X[:,0], X[:,1], c=k_model.labels_)
    
    dendrogram(
        Z,
        ax=axes[1],
        labels=df['hierarchal_labels']
    )

    plt.show()
    return df

print(pipeline(X, 4))