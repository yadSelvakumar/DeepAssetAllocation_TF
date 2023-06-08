import os
from tensorflow import keras as tfk
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy as sp
import sys

# TODO: tf.data
# ------------------------------- TENSORFLOW SETUP ------------------------------- #
# WARNING: May slow down training, depends on hardware
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

DEVICE = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
print('Using device', DEVICE)

# ------------------------------- FILE LOCATIONS ------------------------------- #

FILES_PATH = '.' # '/run/user/12406999/gvfs/smb-share:server=fam-ldn-nas01.local,share=fulcrum/Macro Research/yad/deep_dynamic_programming/model_checks/MARS_v1NN_v3'
FIGURES_PATH = '/figures'
RESULTS_PATH = '/results'
SAVE_OPTIONS = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
# LOAD_OPTIONS = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)

os.makedirs(FILES_PATH + FIGURES_PATH, exist_ok=True)
os.makedirs(FILES_PATH + RESULTS_PATH, exist_ok=True)

# -------------------------- FULCRUM MODEL SETTINGS -------------------------- #


def unpack_mars_settings(MARS_FILE):
    settings, parameters = MARS_FILE["settings"], MARS_FILE["parameters"]
    GAMMA = np.float32(settings["allocationSettings"][0][0][0][0][0][0][0])

    NUM_VARS = settings["nstates"][0][0][0][0]
    NUM_ASSETS = settings["allocationSettings"][0][0][0][0][2][0][0]
    NUM_STATES = NUM_ASSETS + settings["allocationSettings"][0][0][0][0][3][0][0] + 1

    PHI_0 = parameters["Phi0_C"][0][0]
    PHI_1 = parameters["Phi1_C"][0][0]
    SIGMA_VARS = parameters["Sigma"][0][0]

    # TODO: better name for P, profitability?
    P = settings["p"][0][0][0][0]
    NUM_PERIODS = 50  # 185  # settings["allocationSettings"][0][0][0][0][1][0][0]

    # TODO: better naming for these 2
    A0 = tf.cast(MARS_FILE["A0"], tf.float32)
    A1 = tf.cast(MARS_FILE["A1"], tf.float32)

    # realized data
    states = MARS_FILE["States"]["states"][0][0]
    states_lags = MARS_FILE["States"]["states_lags"][0][0]

    Z_DATA = tf.cast(tf.concat([states, states_lags], axis=1), tf.float32)
    return GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS, Z_DATA


def get_model_parameters(settings, MARS_FILE):
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _, _ = settings
    GAMMA_MINUS = 1 - GAMMA
    GAMMA_INVERSE = 1 / GAMMA_MINUS
    COVARIANCE_MATRIX = np.vstack((np.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), np.zeros((NUM_STATES-NUM_VARS, NUM_VARS))))
    assert (np.isclose(COVARIANCE_MATRIX@COVARIANCE_MATRIX.T, MARS_FILE["parameters"]["Sigma_vC"][0][0]).all())

    UNCONDITIONAL_MEAN = tf.transpose((np.linalg.inv(np.eye(NUM_STATES) - PHI_1)@PHI_0))[0, :]

    SIGMA_DIAGONAL_SQRT_VARS = np.sqrt(np.diag(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL = np.tile(SIGMA_DIAGONAL_SQRT_VARS, P)

    # TODO: reach, effieciency? Better names
    HETA_RF = np.zeros((1, NUM_STATES))
    HETA_RF[0, 0] = 1

    HETA_R = np.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R[np.arange(NUM_ASSETS), np.arange(NUM_ASSETS)+1] = np.ones(NUM_ASSETS)

    return GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R


MARS_FILE = sp.io.loadmat(f'{FILES_PATH}/settings/ResultsForYad.mat')
SETTINGS = unpack_mars_settings(MARS_FILE)
PARAMETERS = get_model_parameters(SETTINGS, MARS_FILE)

# ------------------------------ OPTIM SETTINGS ------------------------------ #

ALPHA_BOUNDS = ([-5, -5, -5], [5, 5, 5])
DRAWDOWN_LIMIT = 0.03

NUM_EPOCHS_OPTIM = 16
NUM_EPOCHS_FIRST_OPTIM = 256

ITER_PER_EPOCH_OPTIM = 50
INVERSE_ITER_PER_EPOCH_OPTIM = tf.constant(1/ITER_PER_EPOCH_OPTIM, dtype=tf.float32)

BATCH_SIZE_OPTIM = 1024
INVERSE_BATCH_SIZE = 1/BATCH_SIZE_OPTIM

INITIAL_LEARNING_RATE_OPTIM = 1e-3
LEARNING_RATE_DECAY_OPTIM = 0.5
LEARNING_RATE_STEP_FIRST_OPTIM = 4_000
LEARNING_RATE_STEP_OPTIM = 500
LEARNING_RATE_STAIRCASE_OPTIM = True

# ------------------------------ CONSTANTS ------------------------------ #

_, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, NUM_PERIODS, Z_DATA = SETTINGS
GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, HETA_RF, HETA_R = PARAMETERS

PHI_0_T = tf.constant(PHI_0.T, dtype=tf.float32)
PHI_1_T = tf.constant(PHI_1.T, dtype=tf.float32)

COVARIANCE_MATRIX_TRANSPOSE = tf.constant(COVARIANCE_MATRIX.T, dtype=tf.float32)

HETA_R_TRANSPOSE = tf.constant(HETA_R.T, dtype=tf.float32)
HETA_RF_TRANPOSE = tf.constant(HETA_RF.T, dtype=tf.float32)

color_palatte_NN = ['tab:red', 'tab:blue', 'xkcd:orchid', 'tab:orange', 'tab:gray']
color_palatte_JV = ['navy', 'darkred', 'darkgreen', 'darkorange', 'black']
YLIM = [[0.0, 0.5, 0.0], [1.0, 1.5, 1.0]]
plt.rcParams.update({'font.size': 8})


@tf.function(reduce_retracing=True)
def value_prime_repeated_fn(z, epsilon, value_prime_func):
    simulated_states_matrix = tf.add(tf.matmul(tf.expand_dims(z, axis=1), PHI_1_T), PHI_0_T)

    states_prime = simulated_states_matrix + tf.matmul(epsilon, COVARIANCE_MATRIX_TRANSPOSE)
    value_prime_array = value_prime_func(tf.reshape(states_prime, PRIME_ARRAY_SHAPE))
    value_prime_repeated = tf.reshape(value_prime_array, PRIME_REPEATED_SHAPE)

    return states_prime, value_prime_repeated


@tf.function(reduce_retracing=True)
def value_function(states_prime, value_prime, alpha):
    # TODO: all the names after r, and rf
    risk = tf.matmul(states_prime, HETA_R_TRANSPOSE)
    riskFree = tf.matmul(states_prime, HETA_RF_TRANPOSE)

    riskFreeExponential = tf.exp(riskFree)
    re = riskFreeExponential * (tf.exp(risk) - 1)
    omega = riskFreeExponential + tf.matmul(re, tf.expand_dims(alpha, axis=2))

    drawdown = tf.cast(omega < (1-DRAWDOWN_LIMIT), tf.float32)

    return GAMMA_INVERSE*tf.pow(omega*value_prime, GAMMA_MINUS), drawdown


@tf.function(reduce_retracing=True)
def get_loss(states_prime, value_prime, alpha):
    value, _ = value_function(states_prime, value_prime, alpha)
    return -tf.reduce_sum(value)*INVERSE_BATCH_SIZE


@tf.function(reduce_retracing=True)
def get_expected_utility(states_prime, value_prime, alpha):
    value, drawdown = value_function(states_prime, value_prime, alpha)
    return tf.reduce_mean(value, axis=1), tf.reduce_sum(drawdown, axis=1)/BATCH_SIZE_OPTIM


@tf.function
def optimal_alpha_step(states_prime, value_prime, alpha, optimizer):
    loss = get_loss(states_prime, value_prime, alpha)
    grads_eu = tf.gradients(loss, alpha)
    grads, _ = tf.clip_by_global_norm(grads_eu, clip_norm=0.5)
    optimizer.apply_gradients(zip(grads, [alpha]))
    return loss


@tf.function
def find_optimal_alpha(z, alpha, value_prime_func, num_samples, optimizer):
    loss_epoch = .0
    eu_epoch = tf.zeros((num_samples, 1))
    drawdown_epoch = tf.zeros((num_samples, 1))
    alpha_epoch = tf.zeros(((num_samples, NUM_ASSETS)))
    for _ in tf.range(ITER_PER_EPOCH_OPTIM):
        epsilon = tf.random.normal(EPSILON_SHAPE)
        states_prime, value_prime = value_prime_repeated_fn(z, epsilon, value_prime_func)

        loss = optimal_alpha_step(states_prime, value_prime, alpha, optimizer)
        alpha_clipped = clip_alpha(alpha)

        loss_epoch += loss
        expected_utility, drawdown_prob = get_expected_utility(states_prime, value_prime, alpha_clipped)
        eu_epoch += expected_utility
        alpha_epoch += alpha_clipped
        drawdown_epoch += drawdown_prob

    return loss_epoch, eu_epoch, alpha_epoch, drawdown_epoch


@tf.function
def optimal_alpha_sgd(z, value_prime_func, alpha, num_samples, number_epochs, optimizer):
    losses = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    expected_utilities = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    alphas = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    drawdowns = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)

    start_time = tf.timestamp()
    for iter_alpha in tf.range(number_epochs):
        approx_time = tf.math.ceil((tf.timestamp()-start_time) * tf.cast((number_epochs-iter_alpha), tf.float64))
        start_time = tf.timestamp()
        if iter_alpha % 32 == 0:
            tf.print(iter_alpha, '/', number_epochs, "(", approx_time, "secs )", summarize=1, output_stream=sys.stdout)

        loss_epoch, EU_epoch, alpha_epoch, drawdown_epoch = find_optimal_alpha(z, alpha, value_prime_func, num_samples, optimizer)

        losses = losses.write(iter_alpha, loss_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)
        expected_utilities = expected_utilities.write(iter_alpha, EU_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)
        alphas = alphas.write(iter_alpha, alpha_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)
        drawdowns = drawdowns.write(iter_alpha, drawdown_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)

    return alphas.stack()[-1], expected_utilities.stack()[-1], losses.stack(), drawdowns.stack()[-1]


@tf.function
def optimal_alpha_start(states_prime, value_prime, alpha, optimizer):
    loss = get_loss(states_prime, value_prime, alpha)
    grads_eu = tf.gradients(loss, alpha)
    grads, _ = tf.clip_by_global_norm(grads_eu, clip_norm=0.5)
    return grads


def find_optimal_alpha_start(z, alpha, value_prime_func, optimizer):
    epsilon = tf.random.normal(EPSILON_SHAPE)
    states_prime, value_prime = value_prime_repeated_fn(z, epsilon, value_prime_func)

    grads = optimal_alpha_start(states_prime, value_prime, alpha, optimizer)
    optimizer.apply_gradients(zip(grads, [alpha]))

    return


@tf.function
def clip_alpha(alpha):
    return tf.clip_by_value(alpha, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])

# TODO: better name for j


@tf.function
def calculate_V(j):
    return (GAMMA_MINUS * j) ** GAMMA_INVERSE


@tf.function
# TODO: better name for jv
def jv_allocation_period(period, simulated_states):
    return tf.expand_dims(A0[:, period], axis=0) + tf.matmul(simulated_states, A1[:, :, period], transpose_b=True)


@tf.function
def initial_prime_function(z): return tf.ones((z.shape[0], 1))


# ---------------------------------------------------------------------------- #
#                                  Get alphas                                  #
# ---------------------------------------------------------------------------- #
load_options = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)

prime_functions = [initial_prime_function]
for period in range(NUM_PERIODS-1):
    print(f'Loading value NN for period: {period}')
    value_neuralnet_fn = tf.keras.models.load_model(f"{FILES_PATH + RESULTS_PATH}/value_{period}", options=load_options, compile=False)
    prime_functions.append(value_neuralnet_fn)


# ---------------------------------------------------------------------------- #
#                                  BY HORIZON                                  #
# ---------------------------------------------------------------------------- #
# num_samples = 1
# PRIME_ARRAY_SHAPE = tf.constant([num_samples * BATCH_SIZE_OPTIM, NUM_STATES], dtype=tf.int32)
# PRIME_REPEATED_SHAPE = tf.constant([num_samples, BATCH_SIZE_OPTIM, 1], dtype=tf.int32)
# INDEXES = tf.constant(tf.range(num_samples))
# EPSILON_SHAPE = tf.constant((num_samples, BATCH_SIZE_OPTIM, NUM_VARS))

# alpha_init_horizon = tf.Variable(tf.constant([[0.5,0.4,0.1]]),name = 'alpha_z',trainable=True,dtype = tf.float32)
# z_init = tf.cast(UNCONDITIONAL_MEAN,tf.float32)
# z_t = tf.expand_dims(tf.cast(z_init,tf.float32),axis = 0)
# # Simulate Z
# alphas_neuralnet = np.zeros((NUM_PERIODS,NUM_ASSETS))
# alphas_JV = np.zeros((NUM_PERIODS,NUM_ASSETS))


# for h in range(NUM_PERIODS):
#     print(f'n = {h}')
#     alpha_JV= jv_allocation_period(h,z_t)
#     alpha_init_horizon.assign(tf.constant(alpha_JV))
#     lr_optim = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_OPTIM, LEARNING_RATE_STEP_FIRST_OPTIM, LEARNING_RATE_DECAY_OPTIM, staircase = LEARNING_RATE_STAIRCASE_OPTIM)
#     optimizer_alpha = tfk.optimizers.Adam(lr_optim)

#     find_optimal_alpha_start(z_t, alpha_init_horizon, prime_functions[0], optimizer_alpha)
#     alpha_neuralnet, J, loss, _ = optimal_alpha_sgd(z_t,prime_functions[h], alpha_init_horizon, num_samples, NUM_EPOCHS_FIRST_OPTIM, optimizer_alpha)

#     alphas_neuralnet[h,:] = alpha_neuralnet
#     alphas_JV[h,:] = alpha_JV

# names = ["Equity","Bond","Commodity"]
# YLIM = [[0.2,0.0,0.0],[0.8,2.0,0.5]]
# plt.figure(figsize = (10,4))
# for j in range(NUM_ASSETS):
#     plt.subplot(1,3,j+1)
#     plt.plot(np.arange(NUM_PERIODS)+1,alphas_neuralnet[:,j], label = 'Neural Network',color = 'tab:green')
#     plt.plot(np.arange(NUM_PERIODS)+1,alphas_JV[:,j], label = 'Jurek-Viceira',color = 'black',linestyle = '--')
#     plt.xlabel('Investment Horizon')
#     plt.title(f'Allocation to asset {names[j]}')
#     plt.legend()
#     plt.ylim(YLIM[0][j],YLIM[1][j])
# plt.savefig(f'{FILES_PATH + FIGURES_PATH}/allocations_by_horizon_{NUM_PERIODS}_JV_init.png')

# --------------------------- REALIZED ALLOCATIONS --------------------------- #
H_TO_PLOT = 0

z_data_sample = Z_DATA[553:, :]
num_samples = z_data_sample.shape[0]
PRIME_ARRAY_SHAPE = tf.constant([num_samples * BATCH_SIZE_OPTIM, NUM_STATES], dtype=tf.int32)
PRIME_REPEATED_SHAPE = tf.constant([num_samples, BATCH_SIZE_OPTIM, 1], dtype=tf.int32)
INDEXES = tf.constant(tf.range(num_samples))
EPSILON_SHAPE = tf.constant((num_samples, BATCH_SIZE_OPTIM, NUM_VARS))

alpha_JV = jv_allocation_period(H_TO_PLOT, z_data_sample)
alpha_init_realized_data = tf.Variable(tf.constant([[0.5, 1.0, 0.6]])*tf.ones((num_samples, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)
lr_optim = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_OPTIM, LEARNING_RATE_STEP_FIRST_OPTIM, LEARNING_RATE_DECAY_OPTIM, staircase=LEARNING_RATE_STAIRCASE_OPTIM)
optimizer_alpha = tfk.optimizers.Adam(lr_optim)

find_optimal_alpha_start(z_data_sample, alpha_init_realized_data, prime_functions[0], optimizer_alpha)
alpha_neuralnet, J, loss, drawdown_probabilities = optimal_alpha_sgd(z_data_sample, prime_functions[H_TO_PLOT], alpha_init_realized_data, num_samples, NUM_EPOCHS_FIRST_OPTIM, optimizer_alpha)

re = tf.matmul(z_data_sample, HETA_R.T)
r_f = tf.matmul(z_data_sample, HETA_RF.T)
R_f = tf.exp(r_f)
Re = tf.exp(r_f)*(tf.exp(re) - 1)

w_nn = np.zeros(num_samples)
w_nn[0] = 1
w_JV = np.zeros(num_samples)
w_JV[0] = 1
w_JV_unc = np.zeros(num_samples)
w_JV_unc[0] = 1

for t in range(num_samples-1):
    w_nn[t+1] = R_f[t+1, 0] + tf.matmul(tf.expand_dims(Re[t+1, :], axis=0), tf.expand_dims(alpha_neuralnet[t, :], axis=1))
    w_JV[t+1] = R_f[t+1, 0] + tf.matmul(tf.expand_dims(Re[t+1, :], axis=0), tf.expand_dims(alpha_JV[t, :], axis=1))

wT_nn = np.cumprod(w_nn)
wT_JV = np.cumprod(w_JV)


YLIM = [[30, 0, -20], [70, 100, 25]]
names = ["Equity", "Bond", "Commodity"]
plt.figure(figsize=(11, 5))
for j in range(NUM_ASSETS):
    plt.subplot(2, 3, j+1)
    plt.plot(alpha_neuralnet[:, j]*100, color=color_palatte_NN[j], label='NN')
    plt.plot(alpha_JV[:, j]*100, color='black', linestyle='--', linewidth=0.8, label='Jurek-Viceira')
    plt.ylim(YLIM[0][j], YLIM[1][j])
    plt.title(f'{names[j]}, H = {H_TO_PLOT+1}')
    if j == 0:
        plt.legend()
plt.subplot(2, 3, 4)
plt.plot(tf.reduce_sum(alpha_neuralnet, axis=1)*100, color=color_palatte_NN[j+1], label='NN')
plt.plot(tf.reduce_sum(alpha_JV, axis=1)*100, color='black', linestyle='--', linewidth=0.8, label='Jurek-Viceira')
plt.title('Total Allocation')

plt.subplot(2, 3, 5)
plt.plot(drawdown_probabilities*100, color=color_palatte_NN[4], label='NN')
plt.legend()
plt.title(f'Probability of {DRAWDOWN_LIMIT*100}% drawdown')
plt.ylabel('Probability (%)')

plt.subplot(2, 3, 6)
plt.plot(wT_nn, color=color_palatte_NN[0], label='NN')
plt.plot(wT_JV, color='black', label='JV', linestyle='--')
plt.legend()
plt.title(f'Total Return, NN:{np.round(wT_nn[-1]*100)}%, JV: {np.round(wT_JV[-1]*100)}%')
plt.ylabel('Wealth ($W_0 = 1$)')

plt.tight_layout()
plt.savefig(f'{FILES_PATH + FIGURES_PATH}/allocation_realized_period_{H_TO_PLOT}_diff.png')

# ----------------------------- Lifetime testins ----------------------------- #
z_lifetime = Z_DATA[-NUM_PERIODS-1:, :]
z_init = tf.expand_dims(tf.cast(UNCONDITIONAL_MEAN, tf.float32), axis=0)

num_samples = 1
PRIME_ARRAY_SHAPE = tf.constant([num_samples * BATCH_SIZE_OPTIM, NUM_STATES], dtype=tf.int32)
PRIME_REPEATED_SHAPE = tf.constant([num_samples, BATCH_SIZE_OPTIM, 1], dtype=tf.int32)
INDEXES = tf.constant(tf.range(num_samples))
EPSILON_SHAPE = tf.constant((num_samples, BATCH_SIZE_OPTIM, NUM_VARS))


re = tf.matmul(z_lifetime, HETA_R.T)
r_f = tf.matmul(z_lifetime, HETA_RF.T)
R_f = tf.exp(r_f)
Re = tf.exp(r_f)*(tf.exp(re) - 1)

w_nn = np.zeros(NUM_PERIODS)
w_nn[0] = 1
w_JV = np.zeros(NUM_PERIODS)
w_JV[0] = 1

alpha_init_lifetime = tf.Variable(tf.constant([[0.5, 1.0, 0.6]])*tf.ones((1, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)


for t in range(NUM_PERIODS-1):
    z_t = tf.expand_dims(z_lifetime[t, :], axis=0)
    lr_optim = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_OPTIM, LEARNING_RATE_STEP_FIRST_OPTIM, LEARNING_RATE_DECAY_OPTIM, staircase=LEARNING_RATE_STAIRCASE_OPTIM)
    optimizer_alpha = tfk.optimizers.Adam(lr_optim)

    find_optimal_alpha_start(z_t, alpha_init_lifetime, prime_functions[NUM_PERIODS-(t+1)], optimizer_alpha)

    alpha_neuralnet, J, loss, drawdown_probabilities = optimal_alpha_sgd(z_t, prime_functions[NUM_PERIODS-(t+1)], alpha_init_lifetime, num_samples, NUM_EPOCHS_FIRST_OPTIM, optimizer_alpha)

    alpha_JV = jv_allocation_period(NUM_PERIODS-(t+1), z_t)

    w_nn[t+1] = R_f[t+1, 0] + tf.matmul(tf.expand_dims(Re[t+1, :], axis=0), tf.transpose(alpha_neuralnet))
    w_JV[t+1] = R_f[t+1, 0] + tf.matmul(tf.expand_dims(Re[t+1, :], axis=0), tf.transpose(alpha_JV))


wT_nn = np.cumprod(w_nn)
wT_JV = np.cumprod(w_JV)

plt.figure()
plt.plot(wT_nn, color=color_palatte_NN[0], label='NN')
plt.plot(wT_JV, color='black', label='JV')
plt.legend()
plt.title(f'Total lifetime return, NN:{np.round(wT_nn[-1]*100)}%, JV: {np.round(wT_JV[-1]*100)}%')
plt.ylabel('Wealth ($W_0 = 1$)')
plt.savefig(f'{FILES_PATH + FIGURES_PATH}/lifetime_weatlh_{NUM_PERIODS}.png')
