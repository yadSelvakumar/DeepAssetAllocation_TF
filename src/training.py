from src.alpha_model import AlphaModel
from src.training_initializer import TrainingInitializer
from src.training_model import TrainingModel
from matplotlib import pyplot as plt
from tensorflow import keras as K
from argparse import Namespace
from scipy.io import loadmat
from logging import Logger
from typing import Callable
from time import time

import tensorflow as tf
import numpy as np
import utils

def plot_loss(losses, title, filepath):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.savefig(filepath)
    plt.close()

# TODO: also reduce parameters with passing settings
# Notice that alpha_JV is only for logging purposes

def train_period_model(period, log: Logger, args: Namespace, prime_function: Callable, alpha_JV: tf.Tensor, initial_alpha: tf.Tensor, alpha_model: AlphaModel, simulated_states: tf.Tensor, num_states: int, alpha_decay_steps: int, model_decay_steps: int, num_periods: int, weights: list[tf.Tensor]):
    log.info('Initializing alpha optimizer')
    log.info(f'PERIOD:{period}/{num_periods}')

    NUM_SAMPLES = args.num_samples

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

    lr_optim_alpha = K.optimizers.schedules.ExponentialDecay(args.learning_rate_alpha, alpha_decay_steps, args.decay_rate_alpha, staircase=True)
    alpha_model.initialize(prime_function, initial_alpha, lr_optim_alpha)

    log.info('Training alpha')

    data = np.zeros((NUM_SAMPLES, num_states+1))

    start_time = time()
    alpha_neuralnet, J, loss = alpha_model(prime_function, args.num_epochs_alpha, alpha_JV)

    log.info(f'Done...took: {(time() - start_time)/60} mins')

    mean_abs_diff = 100*np.mean(np.abs(alpha_neuralnet-alpha_JV), axis=0)
    max_alpha_diff = 100*np.max(np.abs(alpha_neuralnet-alpha_JV), axis=0)
    mean_diff = 100*np.mean(alpha_neuralnet-alpha_JV, axis=0)
    total_mean_abs_error = 100*np.mean(np.abs(alpha_neuralnet-alpha_JV))

    log.info(f'Mean abs diff (ppts): {mean_abs_diff}, Max alpha difference (ppts): {max_alpha_diff}, Mean diff (ppts): {mean_diff}, Loss = {loss[-1]}, Total mean abs error: {total_mean_abs_error}')

    # TODO: can tensorflow all this numpy, and get it from alpha_optm
    V = (alpha_model.gamma_minus * J) ** alpha_model.gamma_minus_inverse

    data[:, :num_states] = simulated_states
    data[:, -1] = V[:, 0]

    data = tf.cast(data[:NUM_SAMPLES], tf.float32)

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})

    # ------------------- Plotting -------------------

    plot_loss(loss, f'Optim loss, period {period}', f'{args.figures_dir}/losses_period_{period}.png')

    assets = ["Cash", "Equity", "Bond", "Commodity"]
    plt.figure(figsize=(12, 10))
    for j in range(alpha_model.alpha.shape[1]):
        plt.subplot(2, 2, j+1)
        plt.plot(alpha_neuralnet[:, j], color='tab:green', label='NN', linewidth=1.0)
        plt.plot(alpha_JV[:, j], color='black', linestyle='--', label='JV', linewidth=0.8)

        plt.title(f'{assets[j]}')
        if j == 0:
            plt.legend()
    plt.savefig(f'{args.figures_dir}/allocations_period_{period}.png')

    # ------------------------------------------------

    log.info('Initializing neural network')

    lr_optim_model = K.optimizers.schedules.ExponentialDecay(args.learning_rate, model_decay_steps, args.decay_rate, staircase=True)
    model = TrainingModel(weights, args, num_states, lr_optim_model)

    log.info('Training neural network')

    model.compile(optimizer=model.optimizer, loss='mse')
    losses = model.train(data, args.num_epochs)

    plot_loss(losses[-20000:], f'Optim loss, period {period}', f'{args.figures_dir}/NN_losses_period_{period}.png')
    model.save(f"{args.results_dir}/value_{period}", options=tf.saved_model.SaveOptions(experimental_io_device="/job:localhost"))

    return model, alpha_neuralnet


def train_model(args: Namespace):
    # --- Settings ---
    MARS_FILE = loadmat(args.settings_file)
    SETTINGS = utils.unpack_mars_settings(MARS_FILE)
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, NUM_PERIODS = SETTINGS
    COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = utils.get_model_settings(SETTINGS, MARS_FILE)

    utils.create_dir_if_missing(args.logs_dir, args.figures_dir, args.results_dir)

    log = utils.create_logger(args.logs_dir, 'training')

    def set_var(name, value):
        log.info(f'{name}: {value}')
        return value

    DEVICE = set_var('Device', '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0')

    NUM_SAMPLES = set_var('Number of Samples', args.num_samples)
    BATCH_SIZE = set_var('Batch Size:', args.batch_size)
    ALPHA_BOUNDS = set_var('Alpha bounds:', ([-5, -5, -5, -5], [5, 5, 5, 5]))

    EPSILON_SHAPE = set_var('Epsilon Shape:', tf.constant((NUM_SAMPLES, BATCH_SIZE, NUM_VARS), dtype=tf.int32))
    PRIME_ARRAY_SHAPE = set_var('Prime Array Shape:', tf.constant([NUM_SAMPLES * BATCH_SIZE, NUM_STATES], dtype=tf.int32))
    PRIME_REPEATED_SHAPE = set_var('Prime Repeated Shape:', tf.constant([NUM_SAMPLES, BATCH_SIZE, 1], dtype=tf.int32))

    ALPHA_CONSTRAINT = set_var('Alpha bounds:', args.alpha_constraint)

    # --- End Settings ---

    log.info('Creating training initializer')

    init = TrainingInitializer(NUM_SAMPLES, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)

    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    prime_functions = [initial_prime_function]

    log.info('Initializing alpha')

    SIMULATED_STATES, SIMULATED_STATES_MATRIX = init.get_states_simulation()
    alpha = tf.Variable(1*tf.random.uniform((NUM_SAMPLES, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    # alpha = tf.Variable(tf.constant([2.0,2.0,2.0,2.0])*tf.ones((NUM_SAMPLES, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    alpha_JV_unc = init.jv_allocation_period(0, SIMULATED_STATES)

    alpha_optm = AlphaModel(alpha, ALPHA_CONSTRAINT, args.iter_per_epoch, NUM_SAMPLES, NUM_ASSETS, GAMMA, BATCH_SIZE, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX, EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)

    model, last_alpha = train_period_model(0, log, args, prime_functions[0], alpha_JV_unc, alpha_JV_unc, alpha_optm, SIMULATED_STATES, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])
    prime_functions.append(model)

    for period in range(1, NUM_PERIODS):
        alpha_JV_unc = init.jv_allocation_period(period, SIMULATED_STATES)
        
        weights = model.trainable_variables

        start_time = time()
        # FIX: This is a hack to get the alpha from the previous period
        # I know where the error is, but this is the fastest way to fix it
        alpha_optm = AlphaModel(alpha, ALPHA_CONSTRAINT, args.iter_per_epoch, NUM_SAMPLES, NUM_ASSETS, GAMMA, BATCH_SIZE, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX, EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)
        model, last_alpha = train_period_model(period, log, args, prime_functions[period], alpha_JV_unc, last_alpha, alpha_optm, SIMULATED_STATES, NUM_STATES, args.decay_steps_alpha, args.decay_steps, NUM_PERIODS, weights)
        time_taken = time() - start_time

        expected_time = time_taken * (NUM_PERIODS - period) / 60
        log.info(f'Period {period} took {time_taken/60} minutes')
        log.info(f'Expected time remaining: {expected_time} minutes')

        prime_functions.append(model)

        K.backend.clear_session()

    log.info('Training complete')

