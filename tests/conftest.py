from tensorflow import keras as K
from src.training_initializer import TrainingInitializer
import tensorflow as tf
import scipy.io as sio
import numpy as np
import pytest


@pytest.fixture
def tf_test() -> tf.test.TestCase:
    return tf.test.TestCase()


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

# ------------------ OLD FUNCTIONS ------------------


@tf.function
def old_jv_allocation_period(period, simulated_states, a0, a1):
    JV_original = tf.expand_dims(a0[:, period], axis=0) + tf.matmul(simulated_states, a1[:, :, period], transpose_b=True)
    cash = 1 - tf.expand_dims(tf.reduce_sum(JV_original, axis=1), axis=1)
    return tf.concat([cash, JV_original], 1)


def old_get_states_simulation(num_samples, initial_state, phi0, phi1, covariance_matrix, num_states, num_vars):
    state_simulations = np.zeros((num_samples, num_states))
    state_simulations[0] = initial_state
    error_epsilon = np.random.multivariate_normal(np.zeros(num_vars), np.eye(num_vars), size=num_samples)
    for n in range(num_samples-1):
        w1 = phi1@state_simulations[n]
        w2 = covariance_matrix@error_epsilon[n]
        state_simulations[n+1] = phi0.T + w1 + w2
    states = tf.constant(state_simulations, tf.float32)
    return states, tf.constant(tf.expand_dims(states, axis=1) @ phi1.T + phi0.T, tf.float32)


def old_unpack_mars_settings(mars_file: dict):
    settings, parameters = mars_file["settings"], mars_file["parameters"]
    GAMMA: float = settings["allocationSettings"][0][0][0][0][0][0][0]

    NUM_VARS = settings["nstates"][0][0][0][0]
    NUM_ASSETS = settings["allocationSettings"][0][0][0][0][2][0][0]
    NUM_STATES = NUM_ASSETS + settings["allocationSettings"][0][0][0][0][3][0][0] + 1

    PHI_0 = parameters["Phi0_C"][0][0]
    PHI_1 = parameters["Phi1_C"][0][0]
    SIGMA_VARS = parameters["Sigma"][0][0]

    P = settings["p"][0][0][0][0]
    NUM_PERIODS = 120  # 320 #settings["allocationSettings"][0][0][0][0][1][0][0]

    A0 = tf.cast(mars_file["A0"], tf.float32)
    A1 = tf.cast(mars_file["A1"], tf.float32)

    return GAMMA, NUM_VARS, NUM_ASSETS + 1, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


def old_get_model_parameters(settings, mars_file: dict):
    _, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _ = settings
    COVARIANCE_MATRIX = np.vstack((np.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), np.zeros((NUM_STATES-NUM_VARS, NUM_VARS))))
    assert (np.isclose(COVARIANCE_MATRIX@COVARIANCE_MATRIX.T, mars_file["parameters"]["Sigma_vC"][0][0]).all())

    UNCONDITIONAL_MEAN = tf.transpose((np.linalg.inv(np.eye(NUM_STATES) - PHI_1)@PHI_0))[0, :]

    SIGMA_DIAGONAL_SQRT_VARS = np.sqrt(np.diag(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL = np.tile(SIGMA_DIAGONAL_SQRT_VARS, P)

    HETA_RF = np.zeros((1, NUM_STATES))
    HETA_RF[0, 0] = 1

    HETA_R = np.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R[np.arange(NUM_ASSETS), np.arange(NUM_ASSETS)+1] = np.ones(NUM_ASSETS)

    return COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R
