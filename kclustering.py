import numpy as np
import pandas as pd

def kmeans(X: pd.DataFrame, k:int):
    data = X.to_numpy()
    num_samples, num_features = data.shape

    # Create the labels (repeated until they match the length of data)
    labels = np.tile(np.arange(k), int(np.ceil(num_samples / k)))[:len(data)]

    # Attach them as a new column
    data = np.column_stack((data, labels))

    previous = [0]
    centers = [1]
    n_iter = 0

    while not np.allclose(previous, centers):
        n_iter += 1
        for i in range(k):
            count = np.sum(data[:,-1] == i)
            if count == 0:
                data[np.random.randint(0,num_samples)][-1] = i

        previous = centers
        centers = [np.mean(data[data[:,-1] == i][:,:num_features], axis=0) for i in range(k)]
        for datapoint in data:
            distances = [np.linalg.norm( center - datapoint[:num_features]) for center in centers]
            datapoint[-1] = np.argmin(distances) 

    print(f'Number of iterations: {n_iter}')
    return data[:,-1]


