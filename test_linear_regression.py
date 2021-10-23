import pytest
import numpy as np
from linear_regression import SampleLinearRegression


@pytest.fixture
def regression():
    """Provides a SampleLinearRegression"""
    regression = SampleLinearRegression(np.array([5, 15, 25, 35, 45, 55]),
                                        np.array([5, 20, 14, 32, 22, 38]))
    yield regression


def test_reshape_input(regression):
    regression.reshape_input()
    assert regression.reshape_input().shape == (6, 1)


def test_train_model_intercept(regression):
    model = regression.train_model()
    assert round(model.intercept_, 2) == 5.63


@pytest.mark.skip("WIP")
def make_prediction_on_unseen_data(regression):
    regression.make_single_prediction(np.array([1, 2, 3, 4, 6]))
    assert round(regression.make_single_prediction(), 2) == 8.33
