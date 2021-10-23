import pytest
import numpy as np
from linear_regression import SampleLinearRegression


@pytest.fixture
def regression():
    """Provides a SampleLinearRegression"""
    regression = SampleLinearRegression(np.array([5, 15, 25, 35, 45, 55]),
                                        np.array([5, 20, 14, 32, 22, 38]))
    yield regression


def test_reshape_regressors(regression):
    result = regression.reshape_regressors()
    assert result.shape == (6, 1)


def test_train_model_intercept(regression):
    model = regression.train_model()
    assert round(model.intercept_, 2) == 5.63


def test_predict_regressors(regression):
    predicted_array = regression.predict_regressors()
    predicted_list = []
    for a in range(len(predicted_array)):
        predicted_list.append(round(predicted_array[a], 2))
    assert predicted_list == [8.33, 13.73, 19.13, 24.53, 29.93, 35.33]


@pytest.mark.skip("WIP")
def test_train_model_slope(regression):
    model = regression.train_model()
    assert model.coef_ == np.array([0.54])


@pytest.mark.parametrize("input_array, list_with_predictions",
                         [
                             (np.array([1, 2, 3, 4]), [6.17, 6.71, 7.25, 7.79]),
                             (np.array([5, 10, 15, 20]), [8.33, 11.03, 13.73, 16.43]),
                             (np.array([12, 22, 23, 24]), [12.11, 17.51, 18.05, 18.59]),
                         ])
def test_make_prediction_on_unseen_data(input_array, list_with_predictions, regression):
    predicted_array = regression.make_prediction_on_unseen_data(input_array)
    predicted_list = []
    for a in range(len(predicted_array)):
        predicted_list.append(round(predicted_array[a], 2))
    assert predicted_list == list_with_predictions
