import numpy as np
import tensorflow as tf
import scipy.io as sio
import pytest


@pytest.fixture
def model_settings(mars_settings, mars_file):
    return old_get_model_parameters(mars_settings, mars_file)


@pytest.fixture
def mars_settings(mars_file):
    return old_unpack_mars_settings(mars_file)


@pytest.fixture
def mars_file():
    return sio.loadmat('settings/ResultsForYad.mat')


@pytest.fixture
def states_and_parameters(mars_settings, model_settings):
    _, NUM_VARS, _, NUM_STATES, _, _, PHI_0, PHI_1, _, _, _ = mars_settings
    _, _, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, _, _ = model_settings

    NUM_SAMPLES = 1024
    np.random.seed(0)
    states, matrix = old_get_states_simulation(NUM_SAMPLES, UNCONDITIONAL_MEAN, PHI_0, PHI_1, COVARIANCE_MATRIX, NUM_STATES, NUM_VARS)
    return states, matrix, NUM_SAMPLES, UNCONDITIONAL_MEAN, PHI_0, PHI_1, COVARIANCE_MATRIX, NUM_STATES, NUM_VARS


def old_get_states_simulation(num_samples, initial_state, phi0, phi1, covariance_matrix, num_states, num_vars):
    state_simulations = np.zeros((num_samples, num_states))
    state_simulations[0, :] = initial_state
    error_epsilon = np.random.multivariate_normal(np.zeros(num_vars), np.eye(num_vars), size=num_samples)
    for n in range(num_samples-1):
        state_simulations[n+1, :] = phi0.T + phi1@state_simulations[n, :] + covariance_matrix@error_epsilon[n, :]
    states = tf.constant(state_simulations, tf.float32)
    return states, tf.constant(tf.expand_dims(states, axis=1) @ phi1.T + phi0.T, tf.float32)


def old_unpack_mars_settings(MARS_FILE):
    settings, parameters = MARS_FILE["settings"], MARS_FILE["parameters"]
    GAMMA: float = settings["allocationSettings"][0][0][0][0][0][0][0]

    NUM_VARS = settings["nstates"][0][0][0][0]
    NUM_ASSETS = settings["allocationSettings"][0][0][0][0][2][0][0]
    NUM_STATES = NUM_ASSETS + settings["allocationSettings"][0][0][0][0][3][0][0] + 1

    PHI_0 = parameters["Phi0_C"][0][0]
    PHI_1 = parameters["Phi1_C"][0][0]
    SIGMA_VARS = parameters["Sigma"][0][0]

    P = settings["p"][0][0][0][0]
    NUM_PERIODS = 50  # 320 #settings["allocationSettings"][0][0][0][0][1][0][0]

    A0 = tf.cast(MARS_FILE["A0"], tf.float32)
    A1 = tf.cast(MARS_FILE["A1"], tf.float32)

    return GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


def old_get_model_parameters(settings, MARS_FILE):
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _ = settings
    GAMMA_MINUS = 1 - GAMMA
    GAMMA_INVERSE = 1 / GAMMA_MINUS
    COVARIANCE_MATRIX = np.vstack((np.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), np.zeros((NUM_STATES-NUM_VARS, NUM_VARS))))
    assert (np.isclose(COVARIANCE_MATRIX@COVARIANCE_MATRIX.T, MARS_FILE["parameters"]["Sigma_vC"][0][0]).all())

    UNCONDITIONAL_MEAN = tf.transpose((np.linalg.inv(np.eye(NUM_STATES) - PHI_1)@PHI_0))[0, :]

    SIGMA_DIAGONAL_SQRT_VARS = np.sqrt(np.diag(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL = np.tile(SIGMA_DIAGONAL_SQRT_VARS, P)

    HETA_RF = np.zeros((1, NUM_STATES))
    HETA_RF[0, 0] = 1

    HETA_R = np.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R[np.arange(NUM_ASSETS), np.arange(NUM_ASSETS)+1] = np.ones(NUM_ASSETS)

    return GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R
