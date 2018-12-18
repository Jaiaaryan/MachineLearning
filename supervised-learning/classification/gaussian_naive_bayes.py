import pandas as pd
import scipy.stats
import numpy as np
# comment: yes I realised how useless is to use pandas dataframe instead of arrays - column names are useless

FILENAME = 'MachineLearning/supervised-learning/classification/gaussian_synthetic_data.csv'

data = pd.read_csv(FILENAME)


class GaussianNB():

    def __init__(self):
        # parameter estimates for each class dependent feature distribution
        self.mean = None
        self.std = None
        # numb of observations
        self.N = None
        # maybe no - do not store as attributes as we'll know
        self.numb_features = None
        self.numb_targets = None

    def train(self, data, target, prior=None):
        """
        Obtaining parameter estimates for normal distributions 
        to each class dependent feature via Max Likelihood

        Arguments: 
        data: pandas dataframe
        target: string - column name of class in data
        prior: pandas series mapping each class (index) to a prior probability
        """
        self.N = data.shape[0]
        self.numb_features = data.shape[1] - 1

        data = data.groupby(by=target, as_index=True)
        temp_df_mean = data.mean()
        self.targets = temp_df_mean.index.values
        # probably do not need to store as attribute
        self.mean = temp_df_mean.values
        # unbiased standard deviation
        self.std = data.std().values
        # numb of obs per class
        numb_obs_class = data.count().values[:,0]
        # number of classes
        self.numb_targets = len(self.targets)

        # evaluate prior if prior is not supplied
        if prior == None:
            self.prior = np.log(numb_obs_class/self.N)
        else:
            self.prior = prior

    def predict(self, data):
        """
        Predict classes for unlabelled dataset

        Arguments:
        data: pandas dataframe of features only
        """
        numb_obs = data.shape[0]
        # array to store predicted values
        prediction = np.zeros(numb_obs)
        # assuming order of features is the same as when trained
        data = data.values
        for k in range(numb_obs):
            obs = data[k]
            distr = np.zeros((self.numb_targets, self.numb_features))
            for i in range(self.numb_targets):
                for j in range(self.numb_features):
                    distr[i,j] = scipy.stats.norm(self.mean[i,j], self.std[i, j]).pdf(obs[j])
            posterior = np.log(distr).sum(axis = 1) + self.prior
            prediction[k] = self.targets[np.argmax(posterior)]
        return prediction
