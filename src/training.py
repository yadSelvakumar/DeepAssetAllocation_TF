from argparse import Namespace
from scipy.io import loadmat
from tensorflow import keras as K
import tensorflow as tf
import numpy as np
import utils

# TODO: Move to own file
class TrainingInitializer:
    def __init__(self, num_samples, num_states, num_vars, covariance_matrix, phi0, phi1, a0, a1, initial_state):
        self.num_samples = num_samples
        self.num_states = num_states
        self.num_vars = num_vars

        self.covariance_matrix = np.array(covariance_matrix)

        self.phi0_t = tf.cast(tf.transpose(phi0), tf.float32)
        self.phi1_t = tf.cast(tf.transpose(phi1), tf.float32)
        self.phi1 = np.array(phi1)

        self.a0 = a0
        self.a1 = a1

        self.initial_state = initial_state

    # TODO: Use tensorflow for optimization
    def get_states_simulation(self) -> tuple[tf.Tensor, tf.Tensor]:
        state_simulations = np.zeros((self.num_samples, self.num_states))
        state_simulations[0] = self.initial_state
        error_epsilon = np.random.multivariate_normal(np.zeros(self.num_vars), np.eye(self.num_vars), size=self.num_samples)
        for n in range(self.num_samples-1):
            w1 = self.phi1 @ state_simulations[n]
            w2 = self.covariance_matrix @ error_epsilon[n]
            state_simulations[n+1] = self.phi0_t + w1 + w2
        states = tf.constant(state_simulations, dtype=tf.float32)
        states_matrix = tf.expand_dims(states, axis=1) @ self.phi1_t + self.phi0_t
        return states, states_matrix

    @tf.function
    def jv_allocation_period(self, period: int, simulated_states: tf.Tensor) -> tf.Tensor:
        JV_original = tf.expand_dims(self.a0[:, period], axis=0) + tf.matmul(simulated_states, self.a1[:, :, period], transpose_b=True)
        cash = 1 - tf.expand_dims(tf.reduce_sum(JV_original, axis=1), axis=1)
        return tf.concat([cash, JV_original], 1)


class AlphaModel(K.Model):
    def __init__(self, period, alpha, initial_alpha, lr_optim, epsilon_shape, num_assets, gamma_inverse, gamma_minus, batch_size, sim_states_matrix, cov_matrix, prime_shape, prime_rep_shape):
        super(AlphaModel, self).__init__()
        alpha.assign(initial_alpha)

        self.period = period
        self.alpha = alpha
        self.optimizer = K.optimizers.Adam(lr_optim)  # type: ignore
        self.sim_states_matrix = sim_states_matrix

        self.num_assets = num_assets
        self.inverse_batch_size = 1 / batch_size

        # TODO: Get Gamma, and calculate this to reduce parameters
        self.gamma_inverse = gamma_inverse
        self.gamma_minus = gamma_minus

        self.epsilon_shape = epsilon_shape
        self.prime_shape = prime_shape
        self.prime_rep_shape = prime_rep_shape

        self.transposed_cov_matrix = tf.transpose(cov_matrix)

    def call(self, value_prime_func):
        epsilon = tf.random.normal(self.epsilon_shape)
        states_prime, value_prime = self.value_prime_repeated_fn(epsilon, value_prime_func)

        alpha_grad = self.optimal_alpha_start(states_prime, value_prime)
        self.optimizer.apply_gradients(zip([alpha_grad], [self.alpha]))

    # TODO: Create test
    @tf.function
    def normalize(self, alpha: tf.Variable) -> tf.Tensor:
        num_assets = alpha.shape[1]
        alpha_normalized = alpha - (tf.reduce_sum(alpha, axis=1, keepdims=True) - 1) / num_assets
        return alpha_normalized

    # TODO: Create test
    @tf.function
    def gradient_projection(self, grad_alpha: tf.Tensor) -> tf.Tensor:
        num_assets = self.alpha.shape[1]
        return grad_alpha - tf.reduce_sum(grad_alpha, axis=1, keepdims=True) / num_assets

    # TODO: This definitely needs a another class
    @tf.function(reduce_retracing=True)
    def value_function_MC_V(self, states_prime, value_prime):
        Rf = tf.expand_dims(tf.exp(states_prime[:, :, 0]), 2)
        R = Rf * tf.exp(states_prime[:, :, 1:self.num_assets])
        omega = tf.matmul(tf.concat((Rf, R), 2), tf.expand_dims(self.alpha, -1))

        return self.gamma_inverse*tf.pow(omega*value_prime, self.gamma_minus)

    # TODO: Create test
    @tf.function(reduce_retracing=True)
    def get_loss(self, states_prime, value_prime):
        return -tf.reduce_sum(self.value_function_MC_V(states_prime, value_prime))*self.inverse_batch_size

    # TODO: Create test
    @tf.function
    def optimal_alpha_start(self, states_prime, value_prime):
        self.alpha.assign(self.normalize(self.alpha))
        with tf.GradientTape() as tape:
            loss = self.get_loss(states_prime, value_prime)

        grads_eu = tape.gradient(loss, self.alpha)
        alpha_grad = self.gradient_projection(grads_eu)
        return alpha_grad

    # TODO: Create test
    @tf.function(reduce_retracing=True)
    def value_prime_repeated_fn(self, epsilon, value_prime_func):
        states_prime = self.sim_states_matrix + tf.matmul(epsilon, self.transposed_cov_matrix)
        value_prime_array = value_prime_func(tf.reshape(states_prime, self.prime_shape))
        value_prime_repeated = tf.reshape(value_prime_array, self.prime_rep_shape)

        return states_prime, value_prime_repeated


class TrainingModel(K.Model):
    pass


def train_model(args: Namespace):
    # --- Settings ---
    utils.create_dir_if_missing(args.logs_dir, args.figures_dir, args.results_dir)

    MARS_FILE = loadmat(args.settings_file)
    SETTINGS = utils.unpack_mars_settings(MARS_FILE)
    _, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, NUM_PERIODS = SETTINGS
    GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, HETA_RF, HETA_R = utils.get_model_settings(SETTINGS, MARS_FILE)

    log = utils.create_logger(args.logs_dir, 'training')

    DEVICE: str = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
    log.info(f'Using device {DEVICE}')

    NUM_SAMPLES = args.num_samples
    log.info(f'Number of samples: {NUM_SAMPLES}')

    BATCH_SIZE = args.batch_size
    log.info(f'Batch size: {BATCH_SIZE}')

    # TODO: This are only used once, try to inline
    EPSILON_SHAPE = tf.constant((NUM_SAMPLES, BATCH_SIZE, NUM_VARS), dtype=tf.int32)
    PRIME_ARRAY_SHAPE = tf.constant([NUM_SAMPLES * BATCH_SIZE, NUM_STATES], dtype=tf.int32)
    PRIME_REPEATED_SHAPE = tf.constant([NUM_SAMPLES, BATCH_SIZE, 1], dtype=tf.int32)
    log.info(f'EPSILON_SHAPE: {EPSILON_SHAPE}')
    log.info(f'PRIME_ARRAY_SHAPE: {PRIME_ARRAY_SHAPE}')
    log.info(f'PRIME_REPEATED_SHAPE: {PRIME_REPEATED_SHAPE}')

    # --- End Settings ---

    log.info('Creating training initializer')

    init = TrainingInitializer(NUM_SAMPLES, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)

    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    prime_functions = [initial_prime_function]

    log.info('Initializing alpha')

    SIMULATED_STATES, SIMULATED_STATES_MATRIX = init.get_states_simulation()
    alpha = tf.Variable(1/(1+NUM_ASSETS)*tf.ones((NUM_SAMPLES, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    alpha_JV_unc = init.jv_allocation_period(0, SIMULATED_STATES)

    log.info('Initializing alpha optimizer')

    log.info(f'PERIOD:0/{NUM_PERIODS}')
    lr_optim = K.optimizers.schedules.ExponentialDecay(args.learning_rate, args.decay_steps, args.decay_rate, staircase=True)
    alpha_optm = AlphaModel(0, alpha, alpha_JV_unc, lr_optim, EPSILON_SHAPE, NUM_ASSETS, GAMMA_INVERSE, GAMMA_MINUS, BATCH_SIZE, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)
    alpha_optm(prime_functions[0])
     
    log.info('Done')
