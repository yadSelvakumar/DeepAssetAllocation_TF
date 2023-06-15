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
import scipy as sp
import numpy as np
import utils

def load_nn_results(args,horizons):
    @tf.function
    def initial_prime_function(z): return tf.ones((z.shape[0], 1))
    v_prime_fn = [initial_prime_function]

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint = False,experimental_io_device = "/job:localhost",experimental_skip_checkpoint = False)    
    for period in range(horizons):
        print(f'Loading value neuralnet for period: {period}')    
        value_neuralnet_fn = tf.keras.models.load_model(f"{args.results_dir}/value_{period}",options = load_options,compile = False)
        v_prime_fn.append(value_neuralnet_fn)
    return v_prime_fn

def init_shapes(data,num_vars,num_states,args):
    states = tf.constant(data,tf.float32)
    states_matrix = tf.expand_dims(states,axis =1)
    num_samples = states.shape[0]

    epsilon_shape = tf.constant((num_samples, args.batch_size, num_vars), dtype=tf.int32)
    prime_array_shape = tf.constant([num_samples * args.batch_size, num_states], dtype=tf.int32)
    prime_repeated_shape = tf.constant([num_samples, args.batch_size, 1], dtype=tf.int32)

    return states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape


def calc_fixed_horizon_allocations(args: Namespace,invest_horizon:int):
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
    BATCH_SIZE = set_var('Batch Size:', args.batch_size)
    ALPHA_CONSTRAINT = set_var('Alpha bounds:', args.alpha_constraint)

    # --- End Settings ---

    load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint = False,experimental_io_device = "/job:localhost",experimental_skip_checkpoint = False)
    v_prime_fn = tf.keras.models.load_model(f"./{args.results_dir}/value_{invest_horizon}",options = load_options,compile = False)

    # --------------------------- Tactical allocations --------------------------- #    
    # log.info('Creating training initializer')
    data = MARS_FILE["states_history"][2606:,:]
    # states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(data,NUM_VARS,NUM_STATES,args)

    # init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
    
    # log.info('Initializing alpha t')
    # alpha_t = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    # alpha_t_JV_unc = init.jv_allocation_period(invest_horizon, states)

    # alpha_t_optm = AlphaModel(alpha_t, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
    # alphas_tactical_t = train_alpha(invest_horizon, log, args, v_prime_fn, alpha_t_JV_unc, alpha_t_JV_unc, alpha_t_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])

    log.info('Initializing alpha t plus1')

    data = MARS_FILE["states_history2"][2606:,:]
    states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(data,NUM_VARS,NUM_STATES,args)
    init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
    alpha_tplus1 = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    alpha_tplus1_JV_unc = init.jv_allocation_period(invest_horizon, states)

    alpha_tplus1_optm = AlphaModel(alpha_tplus1, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
    alphas_tactical_tplus1 = train_alpha(invest_horizon, log, args, v_prime_fn, alpha_tplus1_JV_unc, alpha_tplus1_JV_unc, alpha_tplus1_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])

    import pandas as pd
    from pandas.tseries.offsets import MonthEnd
    pandas_dates = pd.to_datetime(MARS_FILE["dates"][2606:,0]-719529,unit = 'd')
    days = np.array(pandas_dates.day)
    eomonth_day = np.array((pandas_dates + MonthEnd(0)).day)
    weight = tf.expand_dims(tf.constant(days/eomonth_day,tf.float32),axis = 1)

    alphas_tactical = alphas_tactical_tplus1#(1-weight)*alphas_tactical_t + weight*alphas_tactical_tplus1
    alphas_tactical_JV = alpha_tplus1_JV_unc#(1-weight)*alpha_t_JV_unc + weight*alpha_tplus1_JV_unc

    # return_portfolio = np.zeros((num_samples))
    # return_portfolio[0] = 1

    # return_portfolio_JV = np.zeros((num_samples))
    # return_portfolio_JV[0] = 1
    # for t in range(num_samples-1):
    #     Rf = tf.expand_dims(tf.exp(states_matrix[t+1,:, 0]), 1)
    #     R = Rf * tf.exp(states_matrix[t+1,:, 1:NUM_ASSETS])
    #     return_portfolio[t+1] = tf.matmul(tf.concat((Rf, R), 1), tf.expand_dims(alphas_tactical[t,:], -1))
    #     return_portfolio_JV[t+1] = tf.matmul(tf.concat((Rf, R), 1), tf.expand_dims(alphas_tactical_JV[t,:], -1))

    # --------------------------- Strategic allocations -------------------------- #
    
    log.info('Creating training initializer')
    states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(tf.expand_dims(UNCONDITIONAL_MEAN,axis = 0),NUM_VARS,NUM_STATES,args)
    init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
    log.info('Initializing alpha t')
    
    alpha = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    alpha_strategic_JV = init.jv_allocation_period(invest_horizon, states)

    alpha_optm = AlphaModel(alpha, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
    alphas_strategic = train_alpha(invest_horizon, log, args, v_prime_fn, alpha_strategic_JV, alpha_strategic_JV, alpha_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])

    assets = ["Cash", "Equity", "Bond", "Commodity"]
    plt.figure(figsize=(12, 10))
    for j in range(NUM_ASSETS):
        plt.subplot(2, 2, j+1)
        plt.plot(pandas_dates,alphas_tactical_JV[:, j], color='black', label='JV', linewidth=0.8)
        plt.hlines(alpha_strategic_JV[0,j],pandas_dates[0],pandas_dates[-1], color='black', linestyle = ':', linewidth=1.0)

        plt.plot(pandas_dates,alphas_tactical[:, j], color='tab:red', label='NN', linewidth=1.0)
        plt.hlines(alphas_strategic[0,j],pandas_dates[0],pandas_dates[-1], color='red',linestyle = ':',linewidth=1.0)


        plt.title(f'{assets[j]}')
        if j == 0:
            plt.legend()
    plt.savefig(f'{args.figures_dir}/realized_allocations_horizon_{invest_horizon}.png')

    # ------------------------------- Save matfile ------------------------------- #
    dict_save = {"alphas_tactical":alphas_tactical,
                    "alphas_strategic":alphas_strategic,
                    "alphas_tactical_JV":alphas_tactical_JV,
                    "alphas_strategic_JV":alpha_strategic_JV,
                    "dates":MARS_FILE["dates"][2606:,:],
                    "investment_horizon":invest_horizon}
    sp.io.savemat(f'{args.results_dir}/fixed_horizon_allocations.mat',dict_save,format = '4')

    return alphas_tactical,alphas_strategic


def calc_term_fund_allocations(args: Namespace, invest_horizon:int):
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
    BATCH_SIZE = set_var('Batch Size:', args.batch_size)
    ALPHA_CONSTRAINT = set_var('Alpha bounds:', args.alpha_constraint)

    # --- End Settings ---
    v_prime_fns = load_nn_results(args,invest_horizon)

    # --------------------------- Tactical allocations --------------------------- #    
    # log.info('Creating training initializer')
    # states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(MARS_FILE["states_history"],NUM_VARS,NUM_STATES,args)

    # init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
    
    # log.info('Initializing alpha t')
    # alpha_t = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
    # alpha_t_JV_unc = init.jv_allocation_period(invest_horizon, states)

    # alpha_t_optm = AlphaModel(alpha_t, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
    # alphas_tactical_t = train_alpha(invest_horizon, log, args, v_prime_fn, alpha_t_JV_unc, alpha_t_JV_unc, alpha_t_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])
    
    import pandas as pd
    from pandas.tseries.offsets import MonthEnd
    pandas_dates = pd.to_datetime(MARS_FILE["dates"][2606:,0]-719529,unit = 'd')
    investment_start = pandas_dates[0]
    investment_end = investment_start + pd.DateOffset(months = invest_horizon)

    data = MARS_FILE["states_history2"][2606:,:]
    alphas_tactical = np.zeros((data.shape[0],NUM_ASSETS))
    alphas_strategic = np.zeros((data.shape[0],NUM_ASSETS))

    alphas_tactical_JV = np.zeros((data.shape[0],NUM_ASSETS))
    alphas_strategic_JV = np.zeros((data.shape[0],NUM_ASSETS))

    for t in range(data.shape[0]):
        remaining_horizon = np.int64(((investment_end - pandas_dates[t])/np.timedelta64(1, 'M')))
        print(f"Calculating: date: {pandas_dates[t]}, inv horizon: {remaining_horizon}")
        # --------------------------------- Tactical --------------------------------- #
        data_t = tf.expand_dims(tf.constant(data[t,:],tf.float32),axis = 0)
        states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(data_t,NUM_VARS,NUM_STATES,args)

        init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
        alpha_tactical_t = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
        alpha_tactical_t_JV = init.jv_allocation_period(remaining_horizon, states)
        
        alpha_tactical_t_optm = AlphaModel(alpha_tactical_t, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
        alpha_tactical_t = train_alpha(remaining_horizon, log, args, v_prime_fns[remaining_horizon], alpha_tactical_t_JV, alpha_tactical_t_JV, alpha_tactical_t_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])

        # --------------------------------- Strategic -------------------------------- #
        states,states_matrix,num_samples,epsilon_shape,prime_array_shape,prime_repeated_shape = init_shapes(tf.expand_dims(UNCONDITIONAL_MEAN,axis = 0),NUM_VARS,NUM_STATES,args)

        init = TrainingInitializer(num_samples, NUM_STATES, NUM_VARS, COVARIANCE_MATRIX, PHI_0, PHI_1, A0, A1, UNCONDITIONAL_MEAN)
        alpha_strategic_t = tf.Variable(0.25*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
        alpha_strategic_t_JV = init.jv_allocation_period(remaining_horizon, states)
        
        alpha_strategic_t_optm = AlphaModel(alpha_strategic_t, ALPHA_CONSTRAINT, args.iter_per_epoch, num_samples, NUM_ASSETS, GAMMA, BATCH_SIZE, states_matrix, COVARIANCE_MATRIX, epsilon_shape, prime_array_shape, prime_repeated_shape)
        alpha_strategic_t = train_alpha(remaining_horizon, log, args, v_prime_fns[remaining_horizon], alpha_strategic_t_JV, alpha_strategic_t_JV, alpha_strategic_t_optm, states_matrix, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, NUM_PERIODS, [])

        # Save
        alphas_tactical[t,:] = alpha_tactical_t
        alphas_strategic[t,:] = alpha_strategic_t

        alphas_tactical_JV[t,:] = alpha_tactical_t_JV
        alphas_strategic_JV[t,:] = alpha_strategic_t_JV



    log.info('Initializing alpha t plus1')

    assets = ["Cash", "Equity", "Bond", "Commodity"]
    plt.figure(figsize=(12, 10))
    for j in range(NUM_ASSETS):
        plt.subplot(2, 2, j+1)
        plt.plot(pandas_dates,alphas_tactical_JV[:, j], color='black', label='JV', linewidth=0.8)
        plt.plot(pandas_dates,alphas_strategic_JV[:,j], color='black', linestyle = ':', linewidth=1.0)

        plt.plot(pandas_dates,alphas_tactical[:, j], color='tab:red', label='NN', linewidth=1.0)
        plt.plot(pandas_dates,alphas_strategic[:,j], color='red',linestyle = ':',linewidth=1.0)


        plt.title(f'{assets[j]}')
        if j == 0:
            plt.legend()
    plt.savefig(f'{args.figures_dir}/realized_allocations_target_date_investor.png')

    dict_save = {"alphas_tactical":alphas_tactical,
                    "alphas_strategic":alphas_strategic,
                    "alphas_tactical_JV":alphas_tactical_JV,
                    "alphas_strategic_JV":alphas_strategic_JV,
                    "dates":MARS_FILE["dates"][2606:,:],
                    "investment_horizon":invest_horizon}
    sp.io.savemat(f'{args.results_dir}/target_date_investor_allocations.mat',dict_save,format = '4')



def train_alpha(horizon, log: Logger, args: Namespace, prime_function: Callable, alpha_JV: tf.Tensor, initial_alpha: tf.Tensor, alpha_model: AlphaModel, simulated_states: tf.Tensor, num_states: int, alpha_decay_steps: int, model_decay_steps: int, num_periods: int, weights: list[tf.Tensor]):
    log.info('Initializing alpha optimizer')

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

    lr_optim_alpha = K.optimizers.schedules.ExponentialDecay(args.learning_rate_alpha, alpha_decay_steps, args.decay_rate_alpha, staircase=True)
    alpha_model.initialize(prime_function, initial_alpha, lr_optim_alpha)

    log.info('Training alpha')

    start_time = time()
    alpha_neuralnet, J, loss = alpha_model(prime_function, args.num_epochs_alpha, alpha_JV)

    log.info(f'Done...took: {(time() - start_time)/60} mins')

    return alpha_neuralnet

