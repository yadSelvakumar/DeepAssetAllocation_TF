from logging import Logger
from typing import Callable
from matplotlib import pyplot as plt
from tensorflow import keras as K
from argparse import Namespace
from scipy.io import loadmat
from tqdm import trange
from time import time
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

# TODO: Change parameters to get settings


class AlphaModel(K.Model):
    def __init__(self, alpha, alpha_bounds, iter_per_epoch, num_samples, num_assets, gamma, batch_size, sim_states_matrix, cov_matrix, epsilon_shape, prime_shape, prime_rep_shape):
        super().__init__()
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds

        self.num_samples = num_samples
        self.num_assets = num_assets
        self.inverse_batch_size = 1 / batch_size

        self.gamma_minus = 1 - gamma
        self.gamma_minus_inverse = 1 / self.gamma_minus

        self.iter_per_epoch = iter_per_epoch
        self.inverse_iter_per_epoch = 1 / self.iter_per_epoch

        self.epsilon_shape = epsilon_shape
        self.prime_shape = prime_shape
        self.prime_rep_shape = prime_rep_shape

        self.transposed_cov_matrix = tf.transpose(cov_matrix)
        self.sim_states_matrix = sim_states_matrix

    def initialize(self, value_prime_func, initial_alpha, lr_optim):
        self.alpha.assign(initial_alpha)
        self.optimizer = K.optimizers.Adam(lr_optim)  # type: ignore

        epsilon = tf.random.normal(self.epsilon_shape)
        states_prime, value_prime = self.value_prime_repeated_fn(epsilon, value_prime_func)

        alpha_grad = self.initialize_optimal_alpha(states_prime, value_prime)
        self.optimizer.apply_gradients(zip([alpha_grad], [self.alpha]))

    @tf.function
    def call(self, value_prime_func, number_epochs) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        losses = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
        EUs = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
        alphas = tf.TensorArray(tf.float32, size=number_epochs, clear_after_read=False, dynamic_size=False)
        start_time = tf.timestamp()
        for iter_alpha in tf.range(number_epochs):
            approx_time = tf.math.ceil((tf.timestamp()-start_time) * tf.cast((number_epochs-iter_alpha), tf.float64))
            start_time = tf.timestamp()

            # TODO: minimize calculations of printing, by only printing steps
            if iter_alpha % 4 == 0:
                tf.print(iter_alpha, '/', number_epochs, "(", approx_time, "secs )", summarize=1)
            loss_epoch, EU_epoch, alpha_epoch = self.find_optimal_alpha(value_prime_func)

            losses = losses.write(iter_alpha, loss_epoch * self.inverse_iter_per_epoch)
            EUs = EUs.write(iter_alpha, EU_epoch * self.inverse_iter_per_epoch)
            alphas = alphas.write(iter_alpha, alpha_epoch * self.inverse_iter_per_epoch)

        return alphas.stack()[-1], EUs.stack()[-1], losses.stack()

    @tf.function(reduce_retracing=True)
    def get_eu(self, states_prime, value_prime, alpha_clipped):
        return tf.reduce_mean(self.value_function_MC_V(states_prime, value_prime, alpha_clipped), axis=1)

    # TODO: This should be the call
    @tf.function
    def optimal_alpha_step(self, states_prime, value_prime):
        self.normalize_alpha()
        with tf.GradientTape() as tape:
            loss = self.get_loss(states_prime, value_prime)

        grads_eu = tape.gradient(loss, self.alpha)
        alpha_grad = self.gradient_projection(grads_eu)
        self.optimizer.apply_gradients(zip([alpha_grad], [self.alpha]))

        return loss

    @tf.function
    def find_optimal_alpha(self, value_prime_func):
        loss_epoch = .0
        eu_epoch = tf.zeros((self.num_samples, 1))
        alpha_epoch = tf.zeros(((self.num_samples, self.num_assets)))
        for _ in tf.range(self.iter_per_epoch):
            epsilon = tf.random.normal(self.epsilon_shape)
            states_prime, value_prime = self.value_prime_repeated_fn(epsilon, value_prime_func)

            loss = self.optimal_alpha_step(states_prime, value_prime)
            alpha_clipped = clip_alpha(self.alpha, self.alpha_bounds)

            loss_epoch += loss
            eu_epoch += self.get_eu(states_prime, value_prime, alpha_clipped)
            alpha_epoch += alpha_clipped
        return loss_epoch, eu_epoch, alpha_epoch

    @tf.function
    def normalize_alpha(self) -> None:
        num_assets = self.alpha.shape[1]
        alpha_normalized = self.alpha - (tf.reduce_sum(self.alpha, axis=1, keepdims=True) - 1) / num_assets
        self.alpha.assign(alpha_normalized)

    @tf.function
    def gradient_projection(self, grad_alpha: tf.Tensor) -> tf.Tensor:
        num_assets = self.alpha.shape[1]
        return grad_alpha - tf.reduce_sum(grad_alpha, axis=1, keepdims=True) / num_assets

    @tf.function(reduce_retracing=True)
    def value_function_MC_V(self, states_prime, value_prime, alpha):
        Rf = tf.expand_dims(tf.exp(states_prime[:, :, 0]), 2)
        R = Rf * tf.exp(states_prime[:, :, 1:self.num_assets])
        omega = tf.matmul(tf.concat((Rf, R), 2), tf.expand_dims(alpha, -1))

        return self.gamma_minus_inverse*tf.pow(omega*value_prime, self.gamma_minus)

    @tf.function(reduce_retracing=True)
    def get_loss(self, states_prime, value_prime) -> tf.Tensor:
        return -tf.reduce_sum(self.value_function_MC_V(states_prime, value_prime, self.alpha))*self.inverse_batch_size

    @tf.function
    def initialize_optimal_alpha(self, states_prime, value_prime):
        self.normalize_alpha()
        with tf.GradientTape() as tape:
            loss = self.get_loss(states_prime, value_prime)

        grads_eu = tape.gradient(loss, self.alpha)
        alpha_grad = self.gradient_projection(grads_eu)
        return alpha_grad

    @tf.function(reduce_retracing=True)
    def value_prime_repeated_fn(self, epsilon, value_prime_func):
        states_prime = self.sim_states_matrix + tf.matmul(epsilon, self.transposed_cov_matrix)
        value_prime_array = value_prime_func(tf.reshape(states_prime, self.prime_shape))
        value_prime_repeated = tf.reshape(value_prime_array, self.prime_rep_shape)

        return states_prime, value_prime_repeated


class TrainingModel(K.Sequential):
    def __init__(self, weights, args, num_states, lr_optim):
        super().__init__()
        self.optimizer = K.optimizers.Adam(lr_optim)  # type: ignore
        self.num_states = num_states
        self.batch_size = args.batch_size
        self.indexes = tf.range(args.num_samples)

        if weights:
            # ,weights[14].numpy(),weights[16].numpy(),weights[18].numpy(),weights[20].numpy(),weights[22].numpy(),weights[24].numpy(),weights[26].numpy(),weights[28].numpy(),weights[30].numpy(),weights[32].numpy()
            kernel_weights = weights[0].numpy(), weights[2].numpy(), weights[4].numpy()  # , weights[6].numpy(), weights[8].numpy(), weights[10].numpy(), weights[12].numpy()
            # ,weights[13].numpy(),weights[15].numpy(),weights[17].numpy(),weights[19].numpy(),weights[21].numpy(),weights[23].numpy(),weights[25].numpy(),weights[27].numpy(),weights[29].numpy(),weights[31].numpy()
            bias_weights = weights[1].numpy(), weights[3].numpy(), weights[5].numpy()  # , weights[7].numpy(), weights[9].numpy(), weights[11].numpy(), weights[13].numpy()

            # Hidden layers
            for layer in range(args.num_hidden_layers):
                self.add(K.layers.Dense(args.num_neurons, activation=args.activation_function, input_dim=num_states, kernel_initializer=K.initializers.Constant(value=kernel_weights[layer]),
                                        bias_initializer=K.initializers.Constant(value=bias_weights[layer])))

            # Output layer
            self.add(K.layers.Dense(args.model_output_size, kernel_initializer=K.initializers.Constant(
                value=kernel_weights[args.num_hidden_layers]), bias_initializer=K.initializers.Constant(value=bias_weights[args.num_hidden_layers]), activation=args.activation_function_output))

        else:
            # Hidden layers
            for layer in range(args.num_hidden_layers):
                self.add(K.layers.Dense(args.num_neurons, activation=args.activation_function, input_dim=num_states))

            # Output layer
            self.add(K.layers.Dense(args.model_output_size, bias_initializer=K.initializers.Constant(value=args.initial_guess), activation=args.activation_function_output))

    def train(self, train_data, number_epochs):
        losses_primes = []
        weights = self.trainable_variables

        gradients_of_primes = self.training_start(train_data)
        self.optimizer.apply_gradients(zip(gradients_of_primes, weights))

        for _ in trange(number_epochs):
            mean_loss_prime = self.training_step(train_data)
            losses_primes.append(mean_loss_prime)

        print(f'Done...\nTrain mean loss: {np.mean(np.array(losses_primes)[-2000:])}')
        return losses_primes

    @tf.function(reduce_retracing=True)
    def objective_neuralnet(self, data):
        random_indexes = tf.random.shuffle(self.indexes)[:self.batch_size]
        data_batch = tf.gather(data, random_indexes)

        states_batch = data_batch[:, :self.num_states]
        value_prime_optimal_batch = tf.expand_dims(data_batch[:, -1], axis=1)
        error_value_prime_neuralnet = tf.reduce_mean(tf.square(value_prime_optimal_batch - self(states_batch)))

        return error_value_prime_neuralnet

    @tf.function
    def training_step(self, data):
        with tf.GradientTape() as tape:
            value_prime_loss = self.objective_neuralnet(data)
        value_prime_gradients = tape.gradient(value_prime_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(value_prime_gradients, self.trainable_variables))
        return value_prime_loss

    @tf.function()
    def training_start(self, data):
        with tf.GradientTape() as tape:
            value_prime_loss = self.objective_neuralnet(data)
        value_prime_gradients = tape.gradient(value_prime_loss, self.trainable_variables)
        return value_prime_gradients


@tf.function
def clip_alpha(alpha, bounds):
    return tf.clip_by_value(alpha, *bounds)

def plot_loss(losses, title, filepath):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.savefig(filepath)
    plt.close()

# TODO: also reduce parameters with passing settings
# Notice that alpha_JV is only for logging purposes

def train_period_model(period, log: Logger, args: Namespace, prime_function: Callable, alpha_JV: tf.Tensor, initial_alpha: tf.Tensor, alpha_model: AlphaModel, simulated_states: tf.Tensor, num_states: int, alpha_decay_steps: int, model_decay_steps: int, weights: list[tf.Tensor]):
    log.info('Initializing alpha optimizer')
    log.info(f'PERIOD:{period}/50')

    NUM_SAMPLES = args.num_samples

    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

    lr_optim_alpha = K.optimizers.schedules.ExponentialDecay(args.learning_rate_alpha, alpha_decay_steps, args.decay_rate_alpha, staircase=True)
    alpha_model.initialize(prime_function, initial_alpha, lr_optim_alpha)

    log.info('Training alpha')

    data = np.zeros((NUM_SAMPLES, num_states+1))

    start_time = time()
    alpha_neuralnet, J, loss = alpha_model(prime_function, args.num_epochs_alpha)

    alpha_neuralnet = clip_alpha(alpha_neuralnet, alpha_model.alpha_bounds)
    alpha_JV = clip_alpha(alpha_JV, alpha_model.alpha_bounds)

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

    # TODO: Add plotting here

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

    alpha_optm = AlphaModel(alpha, ALPHA_BOUNDS, args.iter_per_epoch, NUM_SAMPLES, NUM_ASSETS, GAMMA, BATCH_SIZE, SIMULATED_STATES_MATRIX, COVARIANCE_MATRIX, EPSILON_SHAPE, PRIME_ARRAY_SHAPE, PRIME_REPEATED_SHAPE)

    model, last_alpha = train_period_model(0, log, args, prime_functions[0], alpha_JV_unc, alpha_JV_unc, alpha_optm, SIMULATED_STATES, NUM_STATES, args.first_decay_steps_alpha, args.first_decay_steps, [])
    prime_functions.append(model)

    for period in range(1, NUM_PERIODS):
        weights = model.trainable_variables

        start_time = time()
        model, last_alpha = train_period_model(period, log, args, prime_functions[period], alpha_JV_unc, last_alpha, alpha_optm, SIMULATED_STATES, NUM_STATES, args.decay_steps_alpha, args.decay_steps, weights)
        time_taken = time() - start_time

        expected_time = time_taken * (NUM_PERIODS - period) / 60
        log.info(f'Period {period} took {time_taken/60} minutes')
        log.info(f'Expected time remaining: {expected_time} minutes')

        prime_functions.append(model)

        K.backend.clear_session()

    log.info('Training complete')
