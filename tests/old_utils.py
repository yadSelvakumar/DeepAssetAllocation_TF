
import numpy as np
import tensorflow as tf


def old_unpack_mars_settings(mars_file: dict):
    settings, parameters = mars_file["settings"], mars_file["parameters"]
    GAMMA: float = settings["allocationSettings"][0][0][0][0][0][0][0]

    NUM_VARS = settings["nstates"][0][0][0][0]
    NUM_ASSETS = settings["allocationSettings"][0][0][0][0][2][0][0]
    NUM_STATES = NUM_ASSETS + settings["allocationSettings"][0][0][0][0][3][0][0] + 1

    PHI_0 = parameters["Phi0_C"][0][0]
    PHI_1 = parameters["Phi1_C"][0][0]
    SIGMA_VARS = parameters["Sigma"][0][0]

    P = settings["p"][0][0][0][0]
    NUM_PERIODS = 120  # 320 #settings["allocationSettings"][0][0][0][0][1][0][0]

    A0 = tf.cast(mars_file["A0"], tf.float32)
    A1 = tf.cast(mars_file["A1"], tf.float32)

    return GAMMA, NUM_VARS, NUM_ASSETS + 1, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS


def old_get_model_parameters(settings, mars_file: dict):
    _, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _ = settings
    COVARIANCE_MATRIX = np.vstack((np.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), np.zeros((NUM_STATES-NUM_VARS, NUM_VARS))))
    assert (np.isclose(COVARIANCE_MATRIX@COVARIANCE_MATRIX.T, mars_file["parameters"]["Sigma_vC"][0][0]).all())

    UNCONDITIONAL_MEAN = tf.transpose((np.linalg.inv(np.eye(NUM_STATES) - PHI_1)@PHI_0))[0, :]

    SIGMA_DIAGONAL_SQRT_VARS = np.sqrt(np.diag(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL = np.tile(SIGMA_DIAGONAL_SQRT_VARS, P)

    HETA_RF = np.zeros((1, NUM_STATES))
    HETA_RF[0, 0] = 1

    HETA_R = np.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R[np.arange(NUM_ASSETS), np.arange(NUM_ASSETS)+1] = np.ones(NUM_ASSETS)

    return COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R
