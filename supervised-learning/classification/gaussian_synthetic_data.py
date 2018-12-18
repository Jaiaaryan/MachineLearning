import pandas as pd
import numpy as np


# path to synthetic data
FILENAME = 'MachineLearning/supervised-learning/classification/gaussian_synthetic_data.csv'
# set numb of classes
K = 2
# set numb of features
M = 2
# sample size
N = 2


column_names = ['ft_{}'.format(i) for i in range(M)] + ['class']
# gaussian normal features
features = np.random.randn(N, M)
classes = np.random.randint(low=0, high=K, size=(N, 1))
data = np.hstack((features, classes))
# save data to csv file
np.savetxt(FILENAME, data, header=','.join(
    column_names), delimiter=',', comments='')
