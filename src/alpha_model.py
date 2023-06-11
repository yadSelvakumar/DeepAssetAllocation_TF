import tensorflow as tf
from tensorflow import keras as K

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
            alpha_clipped = tf.clip_by_value(self.alpha, *self.alpha_bounds)

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
