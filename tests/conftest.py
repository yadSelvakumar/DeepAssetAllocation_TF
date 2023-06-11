import numpy as np
import tensorflow as tf
import scipy.io as sio
import pytest

from src.training import TrainingInitializer

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
def simulated_states(mars_settings, model_settings):
    np.random.seed(0)
    _, NUM_VARS, _, NUM_STATES, _, _, PHI_0, PHI_1, _, _, _ = mars_settings
    _, _, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, _, _ = model_settings
    states, matrix = old_get_states_simulation(1024, UNCONDITIONAL_MEAN, PHI_0, PHI_1, COVARIANCE_MATRIX, NUM_STATES, NUM_VARS)
    return states, matrix

@pytest.fixture
def alpha_allocation(mars_settings, simulated_states):
    np.random.seed(0)
    _, _, _, _, A0, A1, _, _, _, _, _ = mars_settings
    states = simulated_states[0]
    alpha = old_jv_allocation_period(0, states, A0, A1)
    return alpha, states

@pytest.fixture
def initializer(mars_settings, model_settings):
    _, NUM_VARS, _, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, _ = mars_settings
    _, _, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, _, _ = model_settings
    return TrainingInitializer(1024, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)

# ------------------ OLD FUNCTIONS ------------------

# TODO: Create fixture
@tf.function
def old_optimal_alpha_start(states_prime, value_prime, alpha):
    alpha.assign(old_normalize(alpha))
    with tf.GradientTape() as tape:
        loss = old_get_loss(states_prime, value_prime, alpha)

    grads_eu = tape.gradient(loss, alpha)
    alpha_grad = old_gradient_projection(alpha, grads_eu)
    return alpha_grad

# TODO: Create fixture
@tf.function
def old_gradient_projection(alpha: tf.Variable, grad_alpha: tf.Tensor) -> tf.Tensor:
    num_assets = alpha.shape[1]
    return grad_alpha - tf.reduce_sum(grad_alpha, axis=1, keepdims=True) / num_assets

# TODO: Create fixture
@tf.function
def old_normalize(alpha: tf.Variable) -> tf.Tensor:
    num_assets = alpha.shape[1]
    return alpha - (tf.reduce_sum(alpha, axis=1, keepdims=True) - 1) / num_assets

# TODO: Create fixture
@tf.function(reduce_retracing=True)
def old_get_loss(states_prime, value_prime, alpha, inverse_batch_size):
    return -tf.reduce_sum(old_value_function_MC_V(states_prime, value_prime, alpha))*inverse_batch_size

# TODO: Create fixture
@tf.function(reduce_retracing=True)
def old_value_prime_repeated_fn(epsilon, value_prime_func, simulated_states_matrix, covariance_matrix_transpose, prime_array_shape, prime_repeated_shape):
    states_prime = simulated_states_matrix + tf.matmul(epsilon, covariance_matrix_transpose)
    value_prime_array = value_prime_func(tf.reshape(states_prime, prime_array_shape))
    value_prime_repeated = tf.reshape(value_prime_array, prime_repeated_shape)

    return states_prime, value_prime_repeated

# TODO: Create fixture
@tf.function(reduce_retracing=True)
def old_value_function_MC_V(states_prime, value_prime, alpha, gamma_minus, gamma_inverse, num_assets):
    Rf = tf.expand_dims(tf.exp(states_prime[:, :, 0]), 2)
    R = Rf * tf.exp(states_prime[:, :, 1:num_assets])
    omega = tf.matmul(tf.concat((Rf, R), 2), tf.expand_dims(alpha, -1))

    return gamma_inverse*tf.pow(omega*value_prime, gamma_minus)

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

    return GAMMA, NUM_VARS, NUM_ASSETS + 1, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


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
