
import numpy as np
import tensorflow as tf


@tf.function
def old_jv_allocation_period(period, simulated_states, a0, a1):
    JV_original = tf.expand_dims(a0[:, period], axis=0) + tf.matmul(simulated_states, a1[:, :, period], transpose_b=True)
    cash = 1 - tf.expand_dims(tf.reduce_sum(JV_original, axis=1), axis=1)
    return tf.concat([cash, JV_original], 1)


def old_get_states_simulation(num_samples, initial_state, phi0, phi1, covariance_matrix, num_states, num_vars):
    state_simulations = np.zeros((num_samples, num_states))
    state_simulations[0] = initial_state
    error_epsilon = np.random.multivariate_normal(np.zeros(num_vars), np.eye(num_vars), size=num_samples)
    for n in range(num_samples-1):
        w1 = phi1@state_simulations[n]
        w2 = covariance_matrix@error_epsilon[n]
        state_simulations[n+1] = phi0.T + w1 + w2
    states = tf.constant(state_simulations, tf.float32)
    return states, tf.constant(tf.expand_dims(states, axis=1) @ phi1.T + phi0.T, tf.float32)