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
import datetime as dt
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


def save_results(filedir: str, filename: str, alphas_tactical_fixed_horizon, alphas_strategic_fixed_horizon, alphas_tactical_JV_fixed_horizon, alphas_strategic_JV_fixed_horizon, dates_fixed_horizon, investment_horizon_fixed_horizon,alphas_tactical_target_date, alphas_strategic_target_date, alphas_tactical_JV_target_date, alphas_strategic_JV_target_date, dates_target_date, investment_horizon_target_date):
    dict_save = {
        "alphas_tactical_fixed_horizon": alphas_tactical_fixed_horizon,
        "alphas_strategic_fixed_horizon": alphas_strategic_fixed_horizon,
        "alphas_tactical_JV_fixed_horizon": alphas_tactical_JV_fixed_horizon,
        "alphas_strategic_JV_fixed_horizon": alphas_strategic_JV_fixed_horizon,
        "dates_fixed_horizon": dates_fixed_horizon,
        "investment_horizon_fixed_horizon": investment_horizon_fixed_horizon,
        "alphas_tactical_target_date": alphas_tactical_target_date,
        "alphas_strategic_target_date": alphas_strategic_target_date,
        "alphas_tactical_JV_target_date": alphas_tactical_JV_target_date,
        "alphas_strategic_JV_target_date": alphas_strategic_JV_target_date,
        "dates_target_date": dates_target_date,
        "investment_horizon_target_date": investment_horizon_target_date        
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

    lr_optim_alpha = K.optimizers.schedules.ExponentialDecay(args.learning_rate_alpha, args.decay_steps_alpha, args.decay_rate_alpha, staircase=True)
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


def weighted_average(weight, alphas_t, alphas_tplus1):
    return (1 - weight) * alphas_t + weight * alphas_tplus1


def calc_fixed_horizon_allocations(args: Namespace, invest_horizon: int):
    log = get_logger(args, 'training')
    MARS_FILE, SETTINGS, COMPUTED_SETTINGS = get_settings(args)
    UNCONDITIONAL_MEAN = COMPUTED_SETTINGS[1]

    # -------------------------------- Check dates ------------------------------- #
    date_last_busday = pd.to_datetime('2023-06-14')#+pd.tseries.offsets.BDay(1)#pd.to_datetime('today').normalize()  - pd.tseries.offsets.BDay(4)
    pandas_date_last_busday = pd.to_datetime(MARS_FILE["dates"][:, 0]-719529, unit='d')[-1]

    assert date_last_busday == pandas_date_last_busday, log.info(f'Date check failed. Last business day:{date_last_busday}, data date: {pandas_date_last_busday}')

    # --- End Settings ---

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)
    v_prime_fn = tf.keras.models.load_model(f"./{args.results_dir}/value_{invest_horizon}", options=load_options, compile=False)

    # --------------------------- Tactical allocations --------------------------- #    
    log.info(f'Investment horizon: {invest_horizon}')
    log.info(f"Device: {'/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'}")

    data_t = np.expand_dims(MARS_FILE["states_history"][-1,:],axis = 0)
    data_tplus1 = np.expand_dims(MARS_FILE["states_history2"][-1,:],axis = 0)
    alphas_tactical_t, alpha_t_JV_unc = run_allocation(log, 'Alpha t', invest_horizon, data_t, v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)
    alphas_tactical_tplus1, alpha_tplus1_JV_unc = run_allocation(log, 'Alpha t+1', invest_horizon, data_tplus1, v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)

    # ------- Calc weight ------ #
    # TODO: Simplify this
    

    days = np.array(pandas_date_last_busday.day)
    eomonth_day = np.array((pandas_date_last_busday + MonthEnd(0)).day)

    # FIX: Using some tf operations outside of tf.function may be slower than using numpy
    weight = tf.constant(days/eomonth_day, tf.float32)

    alphas_tactical = weighted_average(weight, alphas_tactical_t, alphas_tactical_tplus1)
    alphas_tactical_JV = weighted_average(weight, alpha_t_JV_unc, alpha_tplus1_JV_unc)

    # --------------------------- Strategic allocations -------------------------- #
    alphas_strategic, alphas_strategic_JV = run_allocation(log, 'Alpha Strategic', invest_horizon, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fn, args, SETTINGS, COMPUTED_SETTINGS)

    return alphas_tactical, alphas_strategic, alphas_tactical_JV, alphas_strategic_JV, MARS_FILE["dates"][-1, 0], invest_horizon


def calc_term_fund_allocations(args: Namespace, term_date: str):
    log = get_logger(args, 'training')
    MARS_FILE, SETTINGS, COMPUTED_SETTINGS = get_settings(args)
    NUM_ASSETS = SETTINGS[2]
    UNCONDITIONAL_MEAN = COMPUTED_SETTINGS[1]    
    # -------------------------------- Check dates ------------------------------- #
    date_last_busday = pd.to_datetime('2023-06-14')#+pd.tseries.offsets.BDay(1)#pd.to_datetime('today').normalize()  - pd.tseries.offsets.BDay(4)
    pandas_date_last_busday = pd.to_datetime(MARS_FILE["dates"][:, 0]-719529, unit='d')[-1]

    assert date_last_busday == pandas_date_last_busday, log.info(f'Date check failed. Last business day:{date_last_busday}, data date: {pandas_date_last_busday}')
    # ----------------------- Calculate investment horizon ----------------------- #
    invest_horizon = np.int64(((pd.to_datetime(term_date) - pandas_date_last_busday)/np.timedelta64(1, 'M')))

    # --- End Settings ---
    v_prime_fns = load_nn_results(args, invest_horizon)
    horizon = invest_horizon
    # --------------------------------- Data --------------------------------- #
    data_t = np.expand_dims(MARS_FILE["states_history"][-1,:],axis = 0)
    data_tplus1 = np.expand_dims(MARS_FILE["states_history2"][-1,:],axis = 0)

    log.info(f"inv horizon: {horizon}")

    start_time = time()
    alpha_tactical_t, alpha_tactical_t_JV = run_allocation(log, 'Tactical (t)', invest_horizon, data_t, v_prime_fns[invest_horizon-1], args, SETTINGS, COMPUTED_SETTINGS)
    alpha_tactical_tplus1, alpha_tactical_tplus1_JV = run_allocation(log, 'Tactical (t+1)', horizon, data_tplus1, v_prime_fns[invest_horizon-2], args, SETTINGS, COMPUTED_SETTINGS)
    alpha_strategic_t_h, alpha_strategic_t_h_JV = run_allocation(log, 'Strategic (h)', horizon, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fns[invest_horizon-1], args, SETTINGS, COMPUTED_SETTINGS)
    alpha_strategic_t_hminus1, alpha_strategic_t_hminus1_JV = run_allocation(log, 'Strategic (h-1)', horizon-1, tf.expand_dims(UNCONDITIONAL_MEAN, axis=0), v_prime_fns[invest_horizon-2], args, SETTINGS, COMPUTED_SETTINGS)

    log.info(f'!!!Done...took: {(time() - start_time)/60} mins')

    # Calculate daily weights
    days = np.array(pandas_date_last_busday.day)
    eomonth_day = np.array((pandas_date_last_busday + MonthEnd(0)).day)
    weight = tf.constant(days/eomonth_day, tf.float32)

    alphas_tactical = weighted_average(weight, alpha_tactical_t, alpha_tactical_tplus1)
    alphas_tactical_JV = weighted_average(weight, alpha_tactical_t_JV, alpha_tactical_tplus1_JV)

    alphas_strategic = weighted_average(weight, alpha_strategic_t_h, alpha_strategic_t_hminus1)
    alphas_strategic_JV = weighted_average(weight, alpha_strategic_t_h_JV, alpha_strategic_t_hminus1_JV)

    # plot_and_save(args.figures_dir_save, 'realized_allocations_target_date_investor_check', pandas_dates, [alphas_tactical, alphas_strategic], [alphas_tactical_JV, alphas_strategic_JV])
    return alphas_tactical, alphas_strategic, alphas_tactical_JV, alphas_strategic_JV, MARS_FILE["dates"][-1, 0], invest_horizon
