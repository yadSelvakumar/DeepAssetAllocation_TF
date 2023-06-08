from tensorflow import keras as tfk
from tqdm import trange
from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy as sp
import sys
import os

# TODO: add correct types to everything

# ------------------------------- TENSORFLOW SETUP ------------------------------- #
# WARNING: May slow down training, depends on hardware
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

DEVICE = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
print('Using device', DEVICE)

# ------------------------------- FILE LOCATIONS ------------------------------- #

FILES_PATH = '.'  # '/run/user/12406999/gvfs/smb-share:server=fam-ldn-nas01.local,share=fulcrum/Macro Research/yad/deep_dynamic_programming/model_checks/MARS_v1NN_v3'
FIGURES_PATH = '/figures'
RESULTS_PATH = '/results'
SAVE_OPTIONS = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
# LOAD_OPTIONS = tf.saved_model.LoadOptions(allow_partial_checkpoint=False, experimental_io_device="/job:localhost", experimental_skip_checkpoint=False)

os.makedirs(FILES_PATH + FIGURES_PATH, exist_ok=True)
os.makedirs(FILES_PATH + RESULTS_PATH, exist_ok=True)

# -------------------------- FULCRUM MODEL SETTINGS -------------------------- #


# TODO: move util functions to utils.py
def unpack_mars_settings(MARS_FILE):
    settings, parameters = MARS_FILE["settings"], MARS_FILE["parameters"]
    GAMMA = np.float32(settings["allocationSettings"][0][0][0][0][0][0][0])

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

    return GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


def get_model_parameters(settings, MARS_FILE):
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


MARS_FILE = sp.io.loadmat(f'{FILES_PATH}/settings/ResultsForYad.mat')
SETTINGS = unpack_mars_settings(MARS_FILE)
PARAMETERS = get_model_parameters(SETTINGS, MARS_FILE)

# ---------------------------- SIMULATION SETTINGS --------------------------- #

NUM_SAMPLES = 16 ** 3

# ------------------------------ OPTIM SETTINGS ------------------------------ #

ALPHA_BOUNDS = ([-5, -5, -5, -5], [5, 5, 5, 5])

NUM_EPOCHS_OPTIM = 40
NUM_EPOCHS_FIRST_OPTIM = 40  # 100

ITER_PER_EPOCH_OPTIM = 50
INVERSE_ITER_PER_EPOCH_OPTIM = tf.constant(1/ITER_PER_EPOCH_OPTIM, dtype=tf.float32)

BATCH_SIZE_OPTIM = 1024
INVERSE_BATCH_SIZE = 1/BATCH_SIZE_OPTIM

INITIAL_LEARNING_RATE_OPTIM = 1e-3
LEARNING_RATE_DECAY_OPTIM = 0.5
LEARNING_RATE_STEP_FIRST_OPTIM = 400  # 600
LEARNING_RATE_STEP_OPTIM = 500
LEARNING_RATE_STAIRCASE_OPTIM = True


# ------------------------------ TRAINING SETTINGS ------------------------------ #

MODEL_OUTPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 2
NUM_NEURONS = 16
ACTIVATION_FUNCTION = "tanh"
ACTIVATION_FUNCTION_OUTPUT = "linear"
INITIAL_GUESS = 1

BATCH_SIZE_NEURALNET = 1024
NUM_EPOCHS_FIRST_NEURALNET = 100_000
NUM_EPOCHS_NEURALNET = 100_000

INITIAL_LEARNING_RATE_NEURALNET = 1e-4
LEARNING_RATE_DECAY_NEURALNET = 0.5
LEARNING_RATE_STEP_FIRST_NEURALNET = 20_000
LEARNING_RATE_STEP_NEURALNET = 25_000
LEARNING_RATE_STAIRCASE_NEURALNET = True

# ------------------------------ CONSTANTS ------------------------------ #

_, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, _, _, NUM_PERIODS = SETTINGS
GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, _, HETA_RF, HETA_R = PARAMETERS

NUM_ASSETS = NUM_ASSETS+1

PHI_0_T = tf.constant(PHI_0.T, dtype=tf.float32)
PHI_1_T = tf.constant(PHI_1.T, dtype=tf.float32)

COVARIANCE_MATRIX_TRANSPOSE = tf.constant(COVARIANCE_MATRIX.T, dtype=tf.float32)

HETA_R_TRANSPOSE = tf.constant(HETA_R.T, dtype=tf.float32)
HETA_RF_TRANPOSE = tf.constant(HETA_RF.T, dtype=tf.float32)

PRIME_ARRAY_SHAPE = tf.constant([NUM_SAMPLES * BATCH_SIZE_OPTIM, NUM_STATES], dtype=tf.int32)
PRIME_REPEATED_SHAPE = tf.constant([NUM_SAMPLES, BATCH_SIZE_OPTIM, 1], dtype=tf.int32)

INDEXES = tf.constant(tf.range(NUM_SAMPLES))
EPSILON_SHAPE = tf.constant((NUM_SAMPLES, BATCH_SIZE_OPTIM, NUM_VARS))

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #


def initialize_neuralnet(weights):
    model = tfk.models.Sequential()

    if weights:
        # ,weights[14].numpy(),weights[16].numpy(),weights[18].numpy(),weights[20].numpy(),weights[22].numpy(),weights[24].numpy(),weights[26].numpy(),weights[28].numpy(),weights[30].numpy(),weights[32].numpy()
        kernel_weights = weights[0].numpy(), weights[2].numpy(), weights[4].numpy()  # , weights[6].numpy(), weights[8].numpy(), weights[10].numpy(), weights[12].numpy()
        # ,weights[13].numpy(),weights[15].numpy(),weights[17].numpy(),weights[19].numpy(),weights[21].numpy(),weights[23].numpy(),weights[25].numpy(),weights[27].numpy(),weights[29].numpy(),weights[31].numpy()
        bias_weights = weights[1].numpy(), weights[3].numpy(), weights[5].numpy()  # , weights[7].numpy(), weights[9].numpy(), weights[11].numpy(), weights[13].numpy()

        # Hidden layers
        for layer in range(NUM_HIDDEN_LAYERS):
            model.add(tfk.layers.Dense(NUM_NEURONS, activation=ACTIVATION_FUNCTION, input_dim=NUM_STATES, kernel_initializer=tfk.initializers.Constant(value=kernel_weights[layer]),
                                       bias_initializer=tfk.initializers.Constant(value=bias_weights[layer])))

        # Output layer
        model.add(tfk.layers.Dense(MODEL_OUTPUT_SIZE, kernel_initializer=tfk.initializers.Constant(
            value=kernel_weights[NUM_HIDDEN_LAYERS]), bias_initializer=tfk.initializers.Constant(value=bias_weights[NUM_HIDDEN_LAYERS]), activation=ACTIVATION_FUNCTION_OUTPUT))

    else:
        # Hidden layers
        for layer in range(NUM_HIDDEN_LAYERS):
            model.add(tfk.layers.Dense(NUM_NEURONS, activation=ACTIVATION_FUNCTION, input_dim=NUM_STATES))

        # Output layer
        model.add(tfk.layers.Dense(MODEL_OUTPUT_SIZE, bias_initializer=tfk.initializers.Constant(value=INITIAL_GUESS), activation=ACTIVATION_FUNCTION_OUTPUT))

    return model

# TODO: Create test class after optimizing


@tf.function(reduce_retracing=True)
def value_prime_repeated_fn(epsilon, value_prime_func):
    states_prime = SIMULATED_STATES_MATRIX + tf.matmul(epsilon, COVARIANCE_MATRIX_TRANSPOSE)
    value_prime_array = value_prime_func(tf.reshape(states_prime, PRIME_ARRAY_SHAPE))
    value_prime_repeated = tf.reshape(value_prime_array, PRIME_REPEATED_SHAPE)

    return states_prime, value_prime_repeated


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, PHI_1_T.shape[0]), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, 1), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, NUM_ASSETS), dtype=tf.float32)])
def value_function_MC_V(states_prime, value_prime, alpha):
    Rf = tf.expand_dims(tf.exp(states_prime[:, :, 0]), 2)
    R = Rf * tf.exp(states_prime[:, :, 1:NUM_ASSETS])
    omega = tf.matmul(tf.concat((Rf, R), 2), tf.expand_dims(alpha, -1))

    return GAMMA_INVERSE*tf.pow(omega*value_prime, GAMMA_MINUS)


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, PHI_1_T.shape[0]), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, 1), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, NUM_ASSETS), dtype=tf.float32)])
def get_loss(states_prime, value_prime, alpha):
    return -tf.reduce_sum(value_function_MC_V(states_prime, value_prime, alpha))*INVERSE_BATCH_SIZE


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, PHI_1_T.shape[0]), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, BATCH_SIZE_OPTIM, 1), dtype=tf.float32), tf.TensorSpec(shape=(NUM_SAMPLES, NUM_ASSETS), dtype=tf.float32)])
def get_eu(states_prime, value_prime, alpha):
    return tf.reduce_mean(value_function_MC_V(states_prime, value_prime, alpha), axis=1)


@tf.function
def optimal_alpha_step(states_prime, value_prime, alpha, learning_rate, optimizer):
    alpha.assign(normalize(alpha))
    with tf.GradientTape() as tape:
        loss = get_loss(states_prime, value_prime, alpha)

    grads_eu = tape.gradient(loss, alpha)
    alpha_grad = gradient_projection(alpha, grads_eu)
    optimizer.apply_gradients(zip([alpha_grad], [alpha]))

    return loss


@tf.function
def find_optimal_alpha(alpha, value_prime_func, current_learning_rate, optimizer):
    loss_epoch = .0
    eu_epoch = tf.zeros((NUM_SAMPLES, 1))
    alpha_epoch = tf.zeros(((NUM_SAMPLES, NUM_ASSETS)))
    for _ in tf.range(ITER_PER_EPOCH_OPTIM):
        epsilon = tf.random.normal(EPSILON_SHAPE)
        states_prime, value_prime = value_prime_repeated_fn(epsilon, value_prime_func)

        loss = optimal_alpha_step(states_prime, value_prime, alpha, current_learning_rate,  optimizer)
        alpha_clipped = clip_alpha(alpha)

        loss_epoch += loss
        eu_epoch += get_eu(states_prime, value_prime, alpha_clipped)
        alpha_epoch += alpha_clipped
    return loss_epoch, eu_epoch, alpha_epoch


@tf.function
def optimal_alpha_sgd(value_prime_func, alpha, number_epochs, optimizer):
    losses = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    EUs = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    alphas = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
    current_learning_rate = INITIAL_LEARNING_RATE_OPTIM
    start_time = tf.timestamp()
    it = 0
    for iter_alpha in tf.range(number_epochs):
        approx_time = tf.math.ceil((tf.timestamp()-start_time) * tf.cast((number_epochs-iter_alpha), tf.float64))
        start_time = tf.timestamp()

        # TODO: minimize printing, by only printing steps
        if iter_alpha % 4 == 0:
            tf.print(iter_alpha, '/', number_epochs, "(", approx_time, "secs )", summarize=1, output_stream=sys.stdout)
        if it % LEARNING_RATE_STEP_FIRST_OPTIM == 0 and it > 0:
            current_learning_rate = current_learning_rate*LEARNING_RATE_DECAY_OPTIM
            print(f'it:{it},lr:{current_learning_rate}')
        loss_epoch, EU_epoch, alpha_epoch = find_optimal_alpha(alpha, value_prime_func, current_learning_rate, optimizer)
        it += ITER_PER_EPOCH_OPTIM

        losses = losses.write(iter_alpha, loss_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)
        EUs = EUs.write(iter_alpha, EU_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)
        alphas = alphas.write(iter_alpha, alpha_epoch * INVERSE_ITER_PER_EPOCH_OPTIM)

    return alphas.stack()[-1], EUs.stack()[-1], losses.stack()


@tf.function
def optimal_alpha_start(states_prime, value_prime, alpha, optimizer):

    alpha.assign(normalize(alpha))
    with tf.GradientTape() as tape:
        loss = get_loss(states_prime, value_prime, alpha)

    grads_eu = tape.gradient(loss, alpha)
    alpha_grad = gradient_projection(alpha, grads_eu)
    return alpha_grad


def find_optimal_alpha_start(alpha, value_prime_func, learning_rate, optimizer):
    epsilon = tf.random.normal(EPSILON_SHAPE)
    states_prime, value_prime = value_prime_repeated_fn(epsilon, value_prime_func)

    alpha_grad = optimal_alpha_start(states_prime, value_prime, alpha, optimizer)
    optimizer.apply_gradients(zip([alpha_grad], [alpha]))


@tf.function
def clip_alpha(alpha):
    return tf.clip_by_value(alpha, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])


@tf.function
def calculate_V(j):
    return (GAMMA_MINUS * j) ** GAMMA_INVERSE


def get_train_data(period, alpha_JV_unc, value_prime_func, number_epochs, alpha, optimizer):
    data = np.zeros((NUM_SAMPLES, NUM_STATES+1))

    start_time = time()
    alpha_neuralnet, J, loss = optimal_alpha_sgd(value_prime_func, alpha, number_epochs, optimizer)

    alpha_neuralnet = clip_alpha(alpha_neuralnet)
    alpha_JV = clip_alpha(alpha_JV_unc)

    print(f'Done...took: {(time() - start_time)/60} mins')
    print(
        f'Mean abs diff (ppts): {100*np.mean(np.abs(alpha_neuralnet-alpha_JV),axis = 0)}, Max alpha difference (ppts): {100*np.max(np.abs(alpha_neuralnet-alpha_JV),axis = 0)}, Mean diff (ppts): {100*np.mean(alpha_neuralnet-alpha_JV,axis = 0)}, Loss = {loss[-1]}, Total mean abs error: {100*np.mean(np.abs(alpha_neuralnet-alpha_JV))}')

    V = calculate_V(J)

    data[:, :NUM_STATES] = SIMULATED_STATES
    data[:, -1] = V[:, 0]

    data = tf.cast(data[:NUM_SAMPLES, :], tf.float32)

    # ------------------- Plotting -------------------

    plot_loss(loss, f'Optim loss, period {period}', f'losses_period_{period}')

    assets = ["Cash", "Equity", "Bond", "Commodity"]
    plt.figure(figsize=(12, 10))
    for j in range(NUM_ASSETS):
        plt.subplot(2, 2, j+1)
        plt.plot(alpha_neuralnet[:, j], color='tab:green', label='NN', linewidth=1.0)
        plt.plot(alpha_JV[:, j], color='black', linestyle='--', label='JV', linewidth=0.8)

        plt.title(f'{assets[j]}')
        if j == 0:
            plt.legend()
    plt.savefig(f'{FILES_PATH + FIGURES_PATH}/allocations_period_{period}.png')

    return data, alpha_neuralnet


@ tf.function(reduce_retracing=True)
def objective_neuralnet(data, value_prime_neuralnet):
    random_indexes = tf.random.shuffle(INDEXES)[:BATCH_SIZE_NEURALNET]
    data_batch = tf.gather(data, random_indexes)

    states_batch = data_batch[:, :NUM_STATES]
    value_prime_optimal_batch = tf.expand_dims(data_batch[:, -1], axis=1)
    error_value_prime_neuralnet = tf.reduce_mean(tf.square(value_prime_optimal_batch - value_prime_neuralnet(states_batch)))

    return error_value_prime_neuralnet


@tf.function
def training_step(value_prime_neuralnet, weights, data, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        value_prime_loss = objective_neuralnet(data, value_prime_neuralnet)
    value_prime_gradients = tape.gradient(value_prime_loss, weights)
    optimizer.apply_gradients(zip(value_prime_gradients, weights))
    return value_prime_loss


@ tf.function()
def training_start(value_prime_neuralnet, weights, data):
    with tf.GradientTape(persistent=True) as tape:
        value_prime_loss = objective_neuralnet(data, value_prime_neuralnet)
    value_prime_gradients = tape.gradient(value_prime_loss, weights)
    return value_prime_gradients


def train_neuralnet(value_prime_neuralnet, train_data, number_epochs_neuralnet, optimizer):
    losses_primes = []
    weights = value_prime_neuralnet.trainable_variables

    gradients_of_primes = training_start(value_prime_neuralnet, weights, train_data)
    optimizer.apply_gradients(zip(gradients_of_primes, weights))

    for _ in trange(number_epochs_neuralnet):
        mean_loss_prime = training_step(value_prime_neuralnet, weights, train_data, optimizer)
        losses_primes.append(mean_loss_prime)

    print(f'Done...\nTrain mean loss: {np.mean(np.array(losses_primes)[-2000:])}')
    return losses_primes


def plot_loss(losses, title, filename):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.savefig(f'{FILES_PATH + FIGURES_PATH}/{filename}.png')
    plt.close()


@tf.function
def normalize(alpha: tf.Variable) -> tf.Tensor:
    num_assets = alpha.shape[1]
    alpha_normalized = alpha - (tf.reduce_sum(alpha, axis=1, keepdims=True) - 1) / num_assets
    return alpha_normalized


@tf.function
def gradient_projection(alpha: tf.Variable, grad_alpha: tf.Tensor) -> tf.Tensor:
    num_assets = alpha.shape[1]
    return grad_alpha - tf.reduce_sum(grad_alpha, axis=1, keepdims=True) / num_assets


def train_period(period, alpha_JV_unc, start_alpha, number_epochs_optim, number_epochs_neuralnet, alpha, prime_functions, weights):
    print(f'PERIOD:{period}/{NUM_PERIODS}')
    alpha.assign(start_alpha)

    # Setup optimizer
    if period == 0:
        lr_optim = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_OPTIM, LEARNING_RATE_STEP_FIRST_OPTIM, LEARNING_RATE_DECAY_OPTIM, staircase=LEARNING_RATE_STAIRCASE_OPTIM)
        lr_neural_net = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_NEURALNET, LEARNING_RATE_STEP_FIRST_NEURALNET, LEARNING_RATE_DECAY_NEURALNET, staircase=LEARNING_RATE_STAIRCASE_NEURALNET)
    else:
        lr_optim = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_OPTIM, LEARNING_RATE_STEP_OPTIM, LEARNING_RATE_DECAY_OPTIM, staircase=LEARNING_RATE_STAIRCASE_OPTIM)
        lr_neural_net = tfk.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE_NEURALNET, LEARNING_RATE_STEP_NEURALNET, LEARNING_RATE_DECAY_NEURALNET, staircase=LEARNING_RATE_STAIRCASE_NEURALNET)
    optimizer_alpha = tfk.optimizers.Adam(lr_optim)
    optimizer_neuralnet = tfk.optimizers.Adam(lr_neural_net)

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
    find_optimal_alpha_start(alpha, prime_functions[period], INITIAL_LEARNING_RATE_OPTIM, optimizer_alpha)
    train_data, last_alphas = get_train_data(period, alpha_JV_unc, prime_functions[period], number_epochs_optim, alpha, optimizer_alpha)

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})
    value_prime_neuralnet = initialize_neuralnet(weights)

    value_prime_neuralnet.compile(optimizer=optimizer_neuralnet, loss='mse')
    losses_prime = train_neuralnet(value_prime_neuralnet, train_data, number_epochs_neuralnet, optimizer_neuralnet)
    prime_functions.append(value_prime_neuralnet)

    plot_loss(losses_prime[-20000:], f'Optim loss, period {period}', f'NN_losses_period_{period}')
    value_prime_neuralnet.save(f"{FILES_PATH + RESULTS_PATH}/value_{period}", options=SAVE_OPTIONS)

    return last_alphas, value_prime_neuralnet.trainable_variables


@tf.function
def jv_allocation_period(period, simulated_states):
    JV_original = tf.expand_dims(A0[:, period], axis=0) + tf.matmul(simulated_states, A1[:, :, period], transpose_b=True)
    cash = 1 - tf.expand_dims(tf.reduce_sum(JV_original, axis=1), axis=1)
    return tf.concat([cash, JV_original], 1)


def get_states_simulation(periods, initial_state):
    state_simulations = np.zeros((periods, NUM_STATES))
    state_simulations[0, :] = initial_state
    error_epsilon = np.random.multivariate_normal(np.zeros(NUM_VARS), np.eye(NUM_VARS), size=periods)
    for n in range(periods-1):
        state_simulations[n+1, :] = PHI_0.T + PHI_1@state_simulations[n, :] + COVARIANCE_MATRIX@error_epsilon[n, :]
    states = tf.constant(state_simulations, tf.float32)
    return states, tf.constant(tf.expand_dims(states, axis=1) @ PHI_1_T + PHI_0_T, tf.float32)


@tf.function
def initial_prime_function(z): return tf.ones((z.shape[0], 1))


def run_model():
    global SIMULATED_STATES, SIMULATED_STATES_MATRIX
    prime_functions = [initial_prime_function]

    SIMULATED_STATES, SIMULATED_STATES_MATRIX = get_states_simulation(NUM_SAMPLES, UNCONDITIONAL_MEAN)
    initial_alpha = tf.Variable(1/(1+NUM_ASSETS)*tf.ones((NUM_SAMPLES, NUM_ASSETS)), name='alpha_z', trainable=True, dtype=tf.float32)

    # # This is the first training period so it's set already
    alpha_JV_unc = jv_allocation_period(0, SIMULATED_STATES)
    last_alphas, weights = train_period(0, alpha_JV_unc, alpha_JV_unc, NUM_EPOCHS_FIRST_OPTIM, NUM_EPOCHS_FIRST_NEURALNET,
                                        initial_alpha, prime_functions, [])

    for period in range(1, NUM_PERIODS):
        start_time = time()

        alpha_JV_unc = jv_allocation_period(period, SIMULATED_STATES)
        last_alphas, weights = train_period(period, alpha_JV_unc, last_alphas, NUM_EPOCHS_OPTIM, NUM_EPOCHS_NEURALNET,
                                            initial_alpha, prime_functions, weights)
        end_time = time()
        print(f'Period {period} took {(end_time - start_time)/60} minutes')
        tfk.backend.clear_session()


if __name__ == '__main__':
    # TODO: Add expected time of the whole training
    with tf.device(DEVICE):
        start_time = time()
        run_model()
        end_time = time()
        print((end_time - start_time)/60)
