from src.training_initializer import TrainingInitializer
from src.args_project import parse_args
from argparse import Namespace
from old_initializer import *
from old_utils import *

import tensorflow as tf
import scipy.io as sio
import numpy as np
import pytest


@pytest.fixture
def tf_test() -> tf.test.TestCase:
    return tf.test.TestCase()


@pytest.fixture
def args() -> Namespace:
    return parse_args('Train Args', 512, 32, default_num_epochs=100, is_test=True)


@pytest.fixture
def num_samples() -> int:
    return 4096


@pytest.fixture
def batch_size() -> int:
    return 1024


@pytest.fixture
def model_settings(mars_settings, mars_file: dict):
    return old_get_model_parameters(mars_settings, mars_file)


@pytest.fixture
def mars_settings(mars_file: dict):
    return old_unpack_mars_settings(mars_file)


@pytest.fixture
def mars_file() -> dict:
    return sio.loadmat('settings/ResultsForYad.mat')


@pytest.fixture
def simulated_states(mars_settings, model_settings, num_samples: int) -> tuple[tf.Tensor, tf.Tensor]:
    np.random.seed(0)
    _, NUM_VARS, _, NUM_STATES, _, _, PHI_0, PHI_1, *_ = mars_settings
    COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = model_settings
    states, matrix = old_get_states_simulation(num_samples, UNCONDITIONAL_MEAN, PHI_0, PHI_1, COVARIANCE_MATRIX, NUM_STATES, NUM_VARS)
    return states, matrix


@pytest.fixture
def alpha_allocation(mars_settings, simulated_states):
    np.random.seed(0)
    states = simulated_states[0]
    alpha = old_jv_allocation_period(0, states, mars_settings[4], mars_settings[5])
    return alpha, states


@pytest.fixture
def initializer(mars_settings, model_settings, num_samples: int) -> TrainingInitializer:
    _, NUM_VARS, _, NUM_STATES, A0, A1, PHI_0, PHI_1, *_ = mars_settings
    COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = model_settings
    return TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
