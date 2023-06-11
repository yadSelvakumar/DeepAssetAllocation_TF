from typing import Callable
from src.alpha_model import AlphaModel
from tensorflow import keras as K
import tensorflow as tf

VPrimeType = tuple[tf.Tensor, tf.Tensor]
ShapesType = tuple[tuple[int, int, int], list[int], list[int]]


def test_normalize(normalized_alpha: tf.Variable, alpha_model: AlphaModel, tf_test: tf.test.TestCase):
    tf_test.assertAllClose(normalized_alpha, alpha_model.normalize_alpha(alpha_model.alpha))


def test_get_loss(loss: tf.Tensor, alpha_model: AlphaModel, value_prime_repeat: VPrimeType, tf_test: tf.test.TestCase):
    tf_test.assertAllClose(loss, alpha_model.get_loss(*value_prime_repeat))


def test_value_prime_repeated_fn(value_prime_repeat: VPrimeType, alpha_model: AlphaModel, initial_prime_fn: Callable[[], tf.Tensor],  shapes: ShapesType, tf_test: tf.test.TestCase):
    tf.random.set_seed(0)
    epsilon = tf.random.normal(shapes[0])
    states_prime, value_prime = value_prime_repeat
    u_states_prime, u_value_prime = alpha_model.value_prime_repeated_fn(epsilon, initial_prime_fn)

    tf_test.assertAllClose(states_prime, u_states_prime)
    tf_test.assertAllClose(value_prime, u_value_prime)


def test_value_function_MC_V(value_fn_result: tf.Tensor, alpha_model: AlphaModel, value_prime_repeat: VPrimeType, tf_test: tf.test.TestCase):
    u_value_fn_result = alpha_model.value_function_MC_V(value_prime_repeat[0], value_prime_repeat[1])
    tf_test.assertAllClose(value_fn_result, u_value_fn_result)


def test_gradient_projection(gradient_projection: tf.Tensor, alpha_model: AlphaModel, grad_alpha: tf.Tensor, tf_test: tf.test.TestCase):
    tf_test.assertAllClose(gradient_projection, alpha_model.gradient_projection(grad_alpha))


def test_optimal_alpha_start(optimal_alpha_start: K.optimizers.Adam, alpha_model: AlphaModel, initial_prime_fn: Callable[[], tf.Tensor], tf_test: tf.test.TestCase):
    tf.random.set_seed(0)
    alpha_model(initial_prime_fn)
    model_optimizer = alpha_model.optimizer
    tf_test.assertAllClose(optimal_alpha_start.get_weights(), model_optimizer.get_weights())
