import numpy as np
import tensorflow as tf
from src.training import TrainingInitializer

def test_get_states_simulation(simulated_states: tuple[tf.Tensor, tf.Tensor], initializer: TrainingInitializer):
    np.random.seed(0)

    states, matrix = simulated_states
    u_states, u_matrix = initializer.get_states_simulation()
    
    tft = tf.test.TestCase()
    tft.assertAllClose(states, u_states, rtol=1e-5, atol=1e-5)
    tft.assertAllClose(matrix, u_matrix, rtol=1e-5, atol=1e-5)

def test_jv_allocation_period(alpha_allocation: tf.Variable, initializer: TrainingInitializer):
    np.random.seed(0)

    alpha, states = alpha_allocation
    u_alpha = initializer.jv_allocation_period(0, states)

    tf.test.TestCase().assertAllClose(alpha, u_alpha, rtol=1e-5, atol=1e-5)

 