import numpy as np
import tensorflow as tf


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