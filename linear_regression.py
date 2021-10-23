import numpy as np
from sklearn.linear_model import LinearRegression


class SampleLinearRegression:
    def __init__(self, regressors, predictor):
        self.regressors = regressors
        self.predictor = predictor

    def reshape_regressors(self):
        self.regressors = self.regressors.reshape((-1, 1))
        return self.regressors

    def train_model(self):
        reshaped_regressors = self.reshape_regressors()
        model = LinearRegression()
        model.fit(reshaped_regressors, self.predictor)
        return model

    def predict_regressors(self):
        model = self.train_model()
        return model.predict(self.regressors)

    def make_prediction_on_unseen_data(self, single_input):
        model = self.train_model()
        return model.predict(single_input.reshape(-1, 1))

