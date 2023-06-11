import numpy as np
import tensorflow as tf
from src.training import TrainingInitializer

def test_get_states_simulation(simulated_states: tuple[tf.Tensor, tf.Tensor], initializer: TrainingInitializer, tf_test: tf.test.TestCase):
    np.random.seed(0)

    states, matrix = simulated_states
    u_states, u_matrix = initializer.get_states_simulation()
    
    tf_test.assertAllClose(states, u_states, rtol=1e-5, atol=1e-5)
    tf_test.assertAllClose(matrix, u_matrix, rtol=1e-5, atol=1e-5)

def test_jv_allocation_period(alpha_allocation: tf.Variable, initializer: TrainingInitializer, tf_test: tf.test.TestCase):
    np.random.seed(0)

    alpha, states = alpha_allocation
    u_alpha = initializer.jv_allocation_period(0, states)

    tf_test.assertAllClose(alpha, u_alpha, rtol=1e-5, atol=1e-5)

 