import pandas as pd
import scipy.stats
import numpy as np
# comment: yes I realised how useless is to use pandas dataframe instead of arrays - column names are useless

FILENAME = 'MachineLearning/supervised-learning/classification/gaussian_synthetic_data.csv'

data = pd.read_csv(FILENAME)


class GaussianNB():

    def train(self, data, target, prior=None):
        """
        Obtaining Gaussian distributions parameters estimates for  
        each target dependent feature via max likelihood

        Arguments: 
        data: pandas dataframe
        target: string - column name of target in data
        prior: pandas series mapping each class (index) to a prior probability
        """

        data = data.groupby(by=target, as_index=True, sort=False)
        # numb of obs per target
        numb_obs_target = data[target].count().values
        # number of observations - for evaluating prior if not given
        N = np.sum(numb_obs_target)
        # temp so as to extract names of targets
        temp_df_mean = data.mean()

        # targets' names
        self.targets = temp_df_mean.index.values
        # number of targets
        self.numb_targets = len(self.targets)
        # probably do not need to store as attribute
        self.mean = temp_df_mean.values
        # unbiased standard deviation
        self.std = data.std().values
        self.numb_features = data.shape[1] - 1
        # evaluate prior 
        if prior == None:
            self.prior = np.log(numb_obs_target/N)
        else:
            self.prior = np.log(prior.values)

    def predict(self, data):
        """
        Predict targets for unlabelled dataset

        Arguments:
        data: pandas dataframe of features only
        assuming features are provided in the 
        same order
        """

        # assuming order of features is the same as when trained
        data = data.values
        # numb of observations
        numb_obs = data.shape[0]
        # array to store and return predicted values
        prediction = np.zeros(numb_obs)
        # maximise posterior for each possible obs
        distr = np.zeros((self.numb_targets, self.numb_features))
        for k in range(numb_obs):
            obs = data[k]
            for i in range(self.numb_targets):
                for j in range(self.numb_features):
                    distr[i, j] = scipy.stats.norm(
                        self.mean[i, j], self.std[i, j]).pdf(obs[j])
            # posterior prob of each target - max log is equivalent
            posterior = np.log(distr).sum(axis=1) + self.prior
            # choose target with max prob
            prediction[k] = self.targets[np.argmax(posterior)]
        return prediction
