from argparse import Namespace
from scipy.io import loadmat
import tensorflow as tf
import utils

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

    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    prime_functions = [initial_prime_function]
