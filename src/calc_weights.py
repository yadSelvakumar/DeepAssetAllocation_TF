from src.alpha_model import AlphaModel
from src.training_initializer import TrainingInitializer
from pandas.tseries.offsets import MonthEnd
from matplotlib import pyplot as plt
from tensorflow import keras as K
from argparse import Namespace
from scipy.io import loadmat
from logging import Logger
from typing import Callable, cast
from time import time

import tensorflow as tf
import scipy as sp
import numpy as np
import pandas as pd
import utils


def get_logger(args: Namespace, name: str) -> Logger:
    utils.create_dir_if_missing(args.logs_dir, args.figures_dir, args.results_dir)
    return utils.create_logger(args.logs_dir, name)


def load_nn_results(args, horizons) -> list[Callable]:
    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    v_prime_fn: list[Callable] = [initial_prime_function]

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)
    for period in range(horizons):
        print(f'Loading value neuralnet for period: {period}')
        value_neuralnet_fn = tf.keras.models.load_model(f"{args.results_dir}/value_{period}", options=load_options, compile=False)
        v_prime_fn.append(cast(Callable, value_neuralnet_fn))

    return v_prime_fn


def save_results(filedir: str, filename: str, alphas_tactical, alphas_strategic, alphas_tactical_JV, alphas_strategic_JV, dates, investment_horizon):
    dict_save = {
        "alphas_tactical": alphas_tactical,
        "alphas_strategic": alphas_strategic,
        "alphas_tactical_JV": alphas_tactical_JV,
        "alphas_strategic_JV": alphas_strategic_JV,
        "dates": dates,
        "investment_horizon": investment_horizon
    }
    sp.io.savemat(f'{filedir}/{filename}.mat', dict_save, format='4')


def plot_and_save(filedir: str, filename: str, pandas_dates, nns: list, jvs: list):
    assets = ["Cash", "Equity", "Bond", "Commodity"]
    plt.figure(figsize=(12, 10))
    for i in range(len(assets)):
        plt.subplot(2, 2, i+1)
        plt.title(f'{assets[i]}')
        for j in range(len(jvs)):
            label, style, width = ('JV', None, 0.8) if j == 0 else (None, ':', 1.0)
            plt.plot(pandas_dates, jvs[j][:, i], color='black', label=label, linestyle=style, linewidth=width)
        for j in range(len(nns)):
            label, color, style = ('NN', 'tab:red', None) if j == 0 else (None, 'red', ':')
            plt.plot(pandas_dates, nns[j][:, i], color=color, label=label, linestyle=style, linewidth=1.0)
        if i == 0:
            plt.legend()
    plt.savefig(f'{filedir}/{filename}.png')


def run_allocation(log: Logger, allocation_name: str, invest_horizon: int, data: tf.Tensor, v_prime_fn: Callable, args: Namespace, SETTINGS: tuple, COMPUTED_SETTINGS: tuple) -> tuple[tf.Tensor, tf.Tensor]:
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, *_ = SETTINGS
    COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, *_ = COMPUTED_SETTINGS

    log.info(f"Initializing allocation: {allocation_name}")
    states, states_prime_expected, num_samples, epsilon_shape, prime_array_shape, prime_repeated_shape = init_shapes(data, NUM_VARS, NUM_STATES, PHI_0, PHI_1, args)

    init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)

    alpha_t = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    alpha_t_JV_unc = init.jv_allocation_period(invest_horizon, states)

    log.info('Initializing alpha optimizer')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

    lr_optim_alpha = K.optimizers.schedules.ExponentialDecay(args.learning_rate_alpha, args.first_decay_steps_alpha, args.decay_rate_alpha, staircase=True)
    alpha_t_optm = AlphaModel(alpha_t, args.alpha_constraint, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, args.batch_size, states_prime_expected, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
    alpha_t_optm.initialize(v_prime_fn, alpha_t_JV_unc, lr_optim_alpha)

    log.info('Training alpha')
    start_time = time()
    alphas_tactical_t, *_ = alpha_t_optm(v_prime_fn, args.num_epochs_alpha)
    log.info(f'Done...took: {(time() - start_time)/60} mins')

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})

    return alphas_tactical_t, alpha_t_JV_unc

# TODO: tf.function this for performance
def init_shapes(data, num_vars, num_states, PHI_0, PHI_1, args):
    states = tf.constant(data, tf.float32)
    phi0_t = tf.cast(tf.transpose(PHI_0), tf.float32)
    phi1_t = tf.cast(tf.transpose(PHI_1), tf.float32)
    states_prime_exp = tf.expand_dims(states, axis=1) @ phi1_t + phi0_t

    num_samples = states.shape[0]

    epsilon_shape = tf.constant((num_samples, args.batch_size, num_vars), dtype=tf.int32)
    prime_array_shape = tf.constant([num_samples * args.batch_size, num_states], dtype=tf.int32)
    prime_repeated_shape = tf.constant([num_samples, args.batch_size, 1], dtype=tf.int32)

    return states, states_prime_exp, num_samples, epsilon_shape, prime_array_shape, prime_repeated_shape


def get_settings(args: Namespace) -> tuple[dict, tuple, tuple]:
    MARS_FILE = loadmat(args.settings_file)

    SETTINGS = utils.unpack_mars_settings(MARS_FILE)
    COMPUTED_SETTINGS = utils.get_model_settings(SETTINGS, MARS_FILE)

    return MARS_FILE, SETTINGS, COMPUTED_SETTINGS

# TODO: tf.function this for performance


def weighted_average(weight, alphas_left, alphas_right):
    return (1 - weight) * alphas_left + weight * alphas_right


def calc_fixed_horizon_allocations(args: Namespace, invest_horizon: int):
    MARS_FILE, SETTINGS, COMPUTED_SETTINGS = get_settings(args)
    UNCONDITIONAL_MEAN = COMPUTED_SETTINGS[1]

    log = get_logger(args, 'training')
    log.info(f'Investment horizon: {invest_horizon}')
    log.info(f"Device: {'/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'}")

    # --- End Settings ---

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)
    v_prime_fn = tf.keras.models.load_model(f"./{args.results_dir}/value_{invest_horizon}", options=load_options, compile=False)

    # --------------------------- Tactical allocations --------------------------- #
    alphas_tactical_t, alpha_t_JV_unc = run_allocation(log, 'Alpha t', invest_horizon, MARS_FILE["states_history"], v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)
    alphas_tactical_tplus1, alpha_tplus1_JV_unc = run_allocation(log, 'Alpha t+1', invest_horizon, MARS_FILE["states_history2"], v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)

    # ------- Calc weight ------ #
    # TODO: Simplify this
    pandas_dates = pd.to_datetime(MARS_FILE["dates"][:, 0]-719529, unit='d')

    days = np.array(pandas_dates.day)
    eomonth_day = np.array((pandas_dates + MonthEnd(0)).day)

    # FIX: Using some tf operations outside of tf.function may be slower than using numpy
    weight = tf.expand_dims(tf.constant(days/eomonth_day, tf.float32), axis=1)

    alphas_tactical = weighted_average(weight, alphas_tactical_t, alphas_tactical_tplus1)
    alphas_tactical_JV = weighted_average(weight, alpha_t_JV_unc, alpha_tplus1_JV_unc)

    # --------------------------- Strategic allocations -------------------------- #
    alphas_strategic, alphas_strategic_JV = run_allocation(log, 'Alpha Strategic', invest_horizon, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)

    plot_and_save(args.figures_dir_save, 'realized_allocations_horizon_{invest_horizon}_new', pandas_dates, [alphas_tactical_JV], [alphas_tactical])

    save_results(args.results_dir_save, 'fixed_horizon_allocations',
                 alphas_tactical, alphas_strategic, alphas_tactical_JV, alphas_strategic_JV, MARS_FILE["dates"], invest_horizon)


def calc_term_fund_allocations(args: Namespace, invest_horizon: int):
    MARS_FILE, SETTINGS, COMPUTED_SETTINGS = get_settings(args)

    NUM_ASSETS = SETTINGS[2]
    UNCONDITIONAL_MEAN = COMPUTED_SETTINGS[1]

    log = get_logger(args, 'training')
    log.info(f"Device: {'/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'}")

    # --- End Settings ---
    v_prime_fns = load_nn_results(args, invest_horizon)

    # --------------------------- Term calculations --------------------------- #
    pandas_dates = pd.to_datetime(MARS_FILE["dates"][2606:, 0]-719529, unit='d')
    investment_end = pandas_dates[0] + pd.DateOffset(months=invest_horizon)

    remaining_horizons = np.int64(((investment_end - pandas_dates)/np.timedelta64(1, 'M')))
    unique_remaining_horizons = np.unique(remaining_horizons)[::-1]
    # --------------------------------- Data --------------------------------- #
    datat, datatplus1 = MARS_FILE["states_history"][2606:], MARS_FILE["states_history2"][2606:]
    empty_alpha = np.zeros((datat.shape[0], NUM_ASSETS))

    alphas_tactical_t, alphas_tactical_t_JV = np.copy(empty_alpha), np.copy(empty_alpha)
    alphas_tactical_tplus1, alphas_tactical_tplus1_JV = np.copy(empty_alpha), np.copy(empty_alpha)
    alphas_strategic_h, alphas_strategic_h_JV = np.copy(empty_alpha), np.copy(empty_alpha)
    alphas_strategic_hminus1, alphas_strategic_hminus1_JV = np.copy(empty_alpha), np.copy(empty_alpha)

    @tf.function
    def repeat_and_assign(alphas, update_alpha, idx) -> tf.Tensor:
        return tf.tensor_scatter_nd_update(alphas, [idx], tf.repeat(update_alpha, repeats=datat.shape[0], axis=0))

    for horizon in unique_remaining_horizons:
        start_time = time()
        log.info(f"inv horizon: {horizon}")

        horizon_idx = horizon == remaining_horizons
        data_h_t = datat[horizon_idx]
        data_h_tplus = datatplus1[horizon_idx]

        alpha_tactical_t, alpha_tactical_t_JV = run_allocation(log, 'Tactical (t)', horizon, data_h_t, v_prime_fns[horizon], args, SETTINGS, COMPUTED_SETTINGS)
        alpha_tactical_tplus1, alpha_tactical_tplus1_JV = run_allocation(log, 'Tactical (t+1)', horizon, data_h_tplus, v_prime_fns[horizon-1], args, SETTINGS, COMPUTED_SETTINGS)
        alpha_strategic_t_h, alpha_strategic_t_h_JV = run_allocation(log, 'Strategic (h)', horizon, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fns[horizon], args, SETTINGS, COMPUTED_SETTINGS)
        alpha_strategic_t_hminus1, alpha_strategic_t_hminus1_JV = run_allocation(log, 'Strategic (h-1)', horizon-1, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fns[horizon-1], args, SETTINGS, COMPUTED_SETTINGS)

        # ---------------------------------- Saving ---------------------------------- #
        alphas_strategic_h = repeat_and_assign(alphas_strategic_h, alpha_strategic_t_h, horizon_idx)
        alphas_strategic_h_JV = repeat_and_assign(alphas_strategic_h_JV, alpha_strategic_t_h_JV,  horizon_idx)

        alphas_strategic_hminus1 = repeat_and_assign(alphas_strategic_hminus1, alpha_strategic_t_hminus1, horizon_idx)
        alphas_strategic_hminus1_JV = repeat_and_assign(alphas_strategic_hminus1_JV, alpha_strategic_t_hminus1_JV, horizon_idx)

        alphas_tactical_t[horizon_idx], alphas_tactical_t_JV[horizon_idx] = alpha_tactical_t, alpha_tactical_t_JV
        alphas_tactical_tplus1[horizon_idx], alphas_tactical_tplus1_JV[horizon_idx] = alpha_tactical_tplus1, alpha_tactical_tplus1_JV

        log.info(f'!!!Done...took: {(time() - start_time)/60} mins')

    # Calculate daily weights
    days = np.array(pandas_dates.day)
    eomonth_day = np.array((pandas_dates + MonthEnd(0)).day)
    weight = tf.expand_dims(tf.constant(days/eomonth_day, tf.float32), axis=1)

    alphas_tactical = weighted_average(weight, alphas_tactical_t, alphas_tactical_tplus1)
    alphas_tactical_JV = weighted_average(weight, alphas_tactical_t_JV, alphas_tactical_tplus1_JV)

    alphas_strategic = weighted_average(weight, alphas_strategic_h, alphas_strategic_hminus1)
    alphas_strategic_JV = weighted_average(weight, alphas_strategic_h_JV, alphas_strategic_hminus1_JV)

    plot_and_save(args.figures_dir_save, 'realized_allocations_target_date_investor', pandas_dates, [alphas_tactical, alphas_strategic], [alphas_tactical_JV, alphas_strategic_JV])

    save_results(args.results_dir_save, 'target_date_investor_allocations', 
                 alphas_tactical, alphas_strategic, alphas_tactical_JV, alphas_strategic_JV, MARS_FILE["dates"][2606:], invest_horizon)
