import numpy as np
from src.training import Training
import tensorflow as tf

def test_get_states_simulation(states_and_parameters):
    states, matrix, NUM_SAMPLES, UNCONDITIONAL_MEAN, PHI_0, PHI_1, SIGMA_VARS, NUM_STATES, NUM_VARS = states_and_parameters
    np.random.seed(0)
    u_states, u_matrix = Training().get_states_simulation(NUM_SAMPLES, UNCONDITIONAL_MEAN, PHI_0, PHI_1, SIGMA_VARS, NUM_STATES, NUM_VARS)
    
    tft = tf.test.TestCase()
    tft.assertAllClose(states, u_states)
    tft.assertAllClose(matrix, u_matrix)

    



    