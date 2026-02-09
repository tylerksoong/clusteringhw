import numpy as np
import pandas as pd

def kmeans(X: pd.DataFrame, k:int):
    data = X.to_numpy()
    num_samples, num_features = data.shape

    # Create the labels (repeated until they match the length of data)
    labels = np.tile(np.arange(k), int(np.ceil(num_samples / k)))[:len(data)]

    # Attach them as a new column
    data = np.column_stack((data, labels))
    print(data)

    
    for i in range(20):
        centers = [np.mean(data[data[:,-1] == i][:,:num_features], axis=0) for i in range(k)]
        print(centers)
        for datapoint in data:
            distances = [np.linalg.norm( center - datapoint[:num_features]) for center in centers]
            datapoint[-1] = np.argmin(distances) 

    return data[:,-1]


