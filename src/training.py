from argparse import Namespace
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import utils

# TODO: Move to own file
class Training:
    def get_states_simulation(self, num_samples, initial_state, phi0, phi1, covariance_matrix, num_states, num_vars):
        state_simulations = np.zeros((num_samples, num_states))
        state_simulations[0, :] = initial_state
        error_epsilon = np.random.multivariate_normal(np.zeros(num_vars), np.eye(num_vars), size=num_samples)
        for n in range(num_samples-1):
            state_simulations[n+1, :] = phi0.T + phi1@state_simulations[n, :] + covariance_matrix@error_epsilon[n, :]
        states = tf.constant(state_simulations, tf.float32)
        states_matrix = tf.constant(tf.expand_dims(states, axis=1) @ phi1.T + phi0.T, tf.float32)
        return states, states_matrix

def train_model(args: Namespace):
    # --- Settings ---

    # WARNING: May slow down training, depends on hardware
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

    utils.create_dir_if_missing(args.logs_dir, args.figures_dir, args.results_dir)
    
    MARS_FILE = loadmat(args.settings_file)
    SETTINGS = utils.unpack_mars_settings(MARS_FILE)
    _, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, NUM_PERIODS = SETTINGS
    GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, HETA_RF, HETA_R = utils.get_model_settings(SETTINGS, MARS_FILE)

    logger = utils.create_logger(args.logs_dir, 'training')

    DEVICE: str = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
    logger.info(f'Using device {DEVICE}')
    
    NUM_SAMPLES = args.num_samples
    logger.info(f'Number of samples: {NUM_SAMPLES}')

    # --- End Settings ---

    # TODO: temporal name
    var = Training()

    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    prime_functions = [initial_prime_function]

    SIMULATED_STATES, SIMULATED_STATES_MATRIX = var.get_states_simulation(NUM_SAMPLES, UNCONDITIONAL_MEAN)
    initial_alpha = tf.Variable(1/(1+NUM_ASSETS)*tf.ones((NUM_SAMPLES, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
