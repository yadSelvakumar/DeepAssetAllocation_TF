import tensorflow as tf
from src.utils import unpack_mars_settings, get_model_settings

def test_unpack_mars(mars_settings, mars_file):
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS = mars_settings
    U_GAMMA, U_NUM_VARS, U_NUM_ASSETS, U_NUM_STATES, U_A0, U_A1, U_PHI_0, U_PHI_1, U_SIGMA_VARS, U_P, U_NUM_PERIODS = unpack_mars_settings(mars_file)

    assert GAMMA == U_GAMMA
    assert NUM_VARS == U_NUM_VARS
    assert NUM_ASSETS == U_NUM_ASSETS
    assert NUM_STATES == U_NUM_STATES
    assert tf.reduce_all(tf.equal(A0, U_A0))
    assert tf.reduce_all(tf.equal(A1, U_A1))
    assert tf.reduce_all(tf.equal(PHI_0, U_PHI_0))
    assert tf.reduce_all(tf.equal(PHI_1, U_PHI_1))
    assert tf.reduce_all(tf.equal(SIGMA_VARS, U_SIGMA_VARS))
    assert P == U_P
    assert NUM_PERIODS == U_NUM_PERIODS

def test_get_model_settings(model_settings, mars_settings, mars_file, tf_test):
    COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R = model_settings
    U_COVARIANCE_MATRIX, U_UNCONDITIONAL_MEAN, U_SIGMA_DIAGONAL, U_HETA_RF, U_HETA_R = get_model_settings(mars_settings, mars_file)

    tf_test.assertAllClose(COVARIANCE_MATRIX, U_COVARIANCE_MATRIX)
    tf_test.assertAllClose(UNCONDITIONAL_MEAN, U_UNCONDITIONAL_MEAN, atol=1e-5, rtol=1e-5)
    tf_test.assertAllClose(SIGMA_DIAGONAL, U_SIGMA_DIAGONAL)
    tf_test.assertAllClose(HETA_RF, U_HETA_RF)
    tf_test.assertAllClose(HETA_R, U_HETA_R)

