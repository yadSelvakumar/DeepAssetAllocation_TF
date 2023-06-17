from tensorflow import keras as K
from src.training_initializer import TrainingInitializer
from src.alpha_model import AlphaModel
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
def lr_optim() -> K.optimizers.schedules.ExponentialDecay:
    return K.optimizers.schedules.ExponentialDecay(1e-3, .5, 400, staircase=True)


@pytest.fixture
def simulated_states(mars_settings, model_settings, num_samples: int) -> tuple[tf.Tensor, tf.Tensor]:
    np.random.seed(0)
    _, NUM_VARS, _, NUM_STATES, _, _, PHI_0, PHI_1, *_ = mars_settings
    _, _, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = model_settings
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
    _, _, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = model_settings
    return TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)


@pytest.fixture
def alpha(mars_settings, num_samples: int):
    NUM_ASSETS = mars_settings[2]
    return tf.Variable(1/(1+NUM_ASSETS)*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)


@pytest.fixture
@tf.function
def initial_prime_fn(z): return tf.ones((z.shape[0], 1))


@pytest.fixture
def value_prime_repeat(initial_prime_fn, shapes, simulated_states, model_settings) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE = shapes
    SIMULATED_STATES_MATRIX = simulated_states[1]
    COVARIANCE_MATRIX_TRANSPOSE = tf.transpose(model_settings[2])
    tf.random.set_seed(0)
    epsilon = tf.random.normal(EPSILON_SHAPE)
    return old_value_prime_repeated_fn(epsilon, initial_prime_fn, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX_TRANSPOSE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)


@pytest.fixture
def value_fn_result(value_fn_params) -> tf.Tensor:
    return old_value_function_MC_V(*value_fn_params)


@pytest.fixture
def value_fn_params(mars_settings, model_settings, alpha, value_prime_repeat) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    NUM_ASSETS = mars_settings[2]
    GAMMA_MINUS, GAMMA_INVERSE, *_ = model_settings

    states_prime, value_prime = value_prime_repeat

    return states_prime, value_prime, alpha, GAMMA_MINUS, GAMMA_INVERSE, NUM_ASSETS


@pytest.fixture
@tf.function
def normalized_alpha(alpha: tf.Variable) -> tf.Variable:
    num_assets = alpha.shape[1]
    return alpha - (tf.reduce_sum(alpha, axis=1, keepdims=True) - 1) / num_assets


@pytest.fixture
@tf.function
def loss(value_fn_result: tf.Tensor, batch_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    return -tf.reduce_sum(value_fn_result) / batch_size


@pytest.fixture
@tf.function
def grad_alpha(alpha: tf.Variable, value_fn_params, batch_size: int) -> tf.Tensor:
    alpha.assign(normalized_alpha(alpha))
    with tf.GradientTape() as tape:
        result = old_value_function_MC_V(*value_fn_params)
        loss_value = loss(result, batch_size)

    return tape.gradient(loss_value, alpha)


@pytest.fixture
@tf.function
def gradient_projection(alpha: tf.Variable, grad_alpha: tf.Tensor) -> tf.Tensor:
    num_assets = alpha.shape[1]
    return grad_alpha - tf.reduce_sum(grad_alpha, axis=1, keepdims=True) / num_assets


@pytest.fixture
def shapes(mars_settings, batch_size: int, num_samples: int) -> tuple[tuple[int, int, int], list[int], list[int]]:
    _, NUM_VARS, _, NUM_STATES, *_ = mars_settings
    EPSILON_SHAPE = (num_samples, batch_size, NUM_VARS)
    PRIME_ARRAY_SHAPE = [num_samples * batch_size, NUM_STATES]
    PRIME_REPEATED_SHAPE = [num_samples, batch_size, 1]

    return EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE


@pytest.fixture
def alpha_model(mars_settings, model_settings, simulated_states, alpha, batch_size, lr_optim, shapes) -> AlphaModel:
    _, _, NUM_ASSETS, _, A0, A1, *_ = mars_settings
    GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, *_ = model_settings

    SIMULATED_STATES, SIMULATED_STATES_MATRIX = simulated_states
    EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE = shapes
    alpha_JV_unc = old_jv_allocation_period(0, SIMULATED_STATES, A0, A1)

    return AlphaModel(0, alpha, alpha_JV_unc, lr_optim, EPSILON_SHAPE, NUM_ASSETS, GAMMA_INVERSE, GAMMA_MINUS, batch_size, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)


@pytest.fixture
def optimal_alpha_start(alpha: tf.Variable, gradient_projection: tf.Tensor, lr_optim: K.optimizers.schedules.ExponentialDecay) -> K.optimizers.Adam:
    opt = K.optimizers.Adam(lr_optim)
    opt.apply_gradients(zip([gradient_projection], [alpha]))
    return opt

# ------------------ OLD FUNCTIONS ------------------


@tf.function(reduce_retracing=True)
def old_value_prime_repeated_fn(epsilon, value_prime_func, simulated_states_matrix, covariance_matrix_transpose, prime_array_shape, prime_repeated_shape):
    states_prime = simulated_states_matrix + tf.matmul(epsilon, covariance_matrix_transpose)
    value_prime_array = value_prime_func(tf.reshape(states_prime, prime_array_shape))
    value_prime_repeated = tf.reshape(value_prime_array, prime_repeated_shape)

    return states_prime, value_prime_repeated


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
    NUM_PERIODS = 50  # 320 #settings["allocationSettings"][0][0][0][0][1][0][0]

    A0 = tf.cast(mars_file["A0"], tf.float32)
    A1 = tf.cast(mars_file["A1"], tf.float32)

    return GAMMA, NUM_VARS, NUM_ASSETS + 1, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


def old_get_model_parameters(settings, mars_file: dict):
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _ = settings
    GAMMA_MINUS = 1 - GAMMA
    GAMMA_INVERSE = 1 / GAMMA_MINUS
    COVARIANCE_MATRIX = np.vstack((np.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), np.zeros((NUM_STATES-NUM_VARS, NUM_VARS))))
    assert (np.isclose(COVARIANCE_MATRIX@COVARIANCE_MATRIX.T, mars_file["parameters"]["Sigma_vC"][0][0]).all())

    UNCONDITIONAL_MEAN = tf.transpose((np.linalg.inv(np.eye(NUM_STATES) - PHI_1)@PHI_0))[0, :]

    SIGMA_DIAGONAL_SQRT_VARS = np.sqrt(np.diag(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL = np.tile(SIGMA_DIAGONAL_SQRT_VARS, P)

    HETA_RF = np.zeros((1, NUM_STATES))
    HETA_RF[0, 0] = 1

    HETA_R = np.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R[np.arange(NUM_ASSETS), np.arange(NUM_ASSETS)+1] = np.ones(NUM_ASSETS)

    return GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R
