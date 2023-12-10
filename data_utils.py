
def resample_data(data, sample_size=1000):
    """Resamples the data to a new shape of (sample_size, feature_dim).

  Args:
    data: The data to be resampled. It should be a NumPy array with shape
      (batch_size, feature_dim).
    sample_size :

  Returns:
    A NumPy array with shape (sample_size, feature_dim).
  """

    batch_size, feature_dim = data.shape
    resampled_data = np.zeros((sample_size, feature_dim))

    dist = cdist(data, data)
    indexes = np.argsort(dist, axis=1)
    dist_sort = np.sort(dist, axis=1)
    mean_array = np.zeros(dist_sort.shape[0])
    for i in range(dist_sort.shape[0]):
        mean_array[i] = np.mean(dist[i, 0:10])
    index_list = []
    for i in range(resampled_data.shape[0]):
        index = random.randint(0, batch_size - 1)
        while mean_array[index] > np.mean(mean_array) + 0.5 * np.std(mean_array):
            if not index in index_list:
                resampled_data[i] = data[index]
                index_list.append(index)

            index = random.randint(0, batch_size - 1)

        dist_neigh = dist_sort[index, :5] * np.random.uniform(0.85,1,5)
        weight = scipy.special.softmax(dist_neigh)
        resampled_data[i] = weight @ data[indexes[index,:5]]

    return resampled_data


def resample_data_normal(data, sample_size=1000):
    """Resamples the data to a new shape of (sample_size, feature_dim).

  Args:
    data: The data to be resampled. It should be a NumPy array with shape
      (batch_size, feature_dim).
    sample_size :

  Returns:
    A NumPy array with shape (sample_size, feature_dim).
  """

    batch_size, feature_dim = data.shape
    resampled_data = np.zeros((sample_size, feature_dim))
    dist = cdist(data, data)
    indexes = np.argsort(dist, axis=1)

    for i in range(resampled_data.shape[0]):
        index = random.randint(0, batch_size - 1)
        neighboures_index = indexes[index, random.randint(1, 7)]
        alpha = random.uniform(0, 0.1)

        resampled_data[i] = (1 - alpha) * data[index] + alpha * data[neighboures_index]

    return resampled_data