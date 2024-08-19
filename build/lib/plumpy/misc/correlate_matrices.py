'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# generate random data: n_trials x n_channels x n_timepoints
data = np.random.random(size=(15, 32, 10))

# correlate across trials
correlations, distance = correlate_matrices(data)

# plot
plt.figure()
plt.subplot(121)
plt.imshow(correlations)
plt.title('Correlations')
plt.subplot(122)
plt.imshow(distance)
plt.title('Distances')
'''
import numpy as np
import pandas as pd
def get_fnorm(matrix1, matrix2):
    diff_mat = matrix1 - matrix2
    frob_norm = np.linalg.norm(diff_mat)
    return frob_norm

def correlate_matrices(data3):
    '''
    Compute correlations across the dimension of trials in the input data. Time and channel dimension are combined
    together for this using the Frobenius norm
    Args:
        data3: a 3d array of data: trials x channels x time
    Returns:
        correlations: trials x trials a correlation matrix
    '''
    n_trials, n_channels, n_time = data3.shape
    difference_matrix = np.zeros(shape=(n_trials, n_trials))

    for row_trial in range(n_trials):
        for column_trial in range(n_trials):
            frob_norm = get_fnorm(data3[row_trial], data3[column_trial])
            difference_matrix[row_trial, column_trial] = frob_norm
            difference_matrix[column_trial, row_trial] = frob_norm

    distance_mat = pd.DataFrame(difference_matrix)
    correlation_mat = distance_mat.corr()
    return correlation_mat.values, distance_mat.values

