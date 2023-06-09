from pathlib import Path
from typing import Union, cast
import tensorflow as tf
import logging

MarsReturnType = tuple[float, int, int, int, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int, int]


def unpack_mars_settings(MARS_FILE: dict, dtype=tf.float32) -> MarsReturnType:
    settings, parameters = MARS_FILE["settings"], MARS_FILE["parameters"]
    alloc_settings: dict = settings["allocationSettings"]
    GAMMA: float = alloc_settings[0][0][0][0][0][0][0]

    NUM_VARS: int = settings["nstates"][0][0][0][0]
    NUM_ASSETS: int = alloc_settings[0][0][0][0][2][0][0]
    NUM_STATES: int = NUM_ASSETS + alloc_settings[0][0][0][0][3][0][0] + 1

    PHI_0: tf.Tensor = tf.convert_to_tensor(parameters["Phi0_C"][0][0], dtype)
    PHI_1: tf.Tensor = tf.convert_to_tensor(parameters["Phi1_C"][0][0], dtype)
    SIGMA_VARS: tf.Tensor = tf.convert_to_tensor(parameters["Sigma"][0][0], dtype)

    P: int = settings["p"][0][0][0][0]
    NUM_PERIODS = 50  # settings["allocationSettings"][0][0][0][0][1][0][0]

    A0: tf.Tensor = cast(tf.Tensor, tf.cast(MARS_FILE["A0"], tf.float32))
    A1: tf.Tensor = cast(tf.Tensor, tf.cast(MARS_FILE["A1"], tf.float32))

    return GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, A0, A1, PHI_0, PHI_1, SIGMA_VARS, P, NUM_PERIODS

# FIX: type warnings
def get_model_settings(settings: MarsReturnType, MARS_FILE: dict) -> tuple[float, float, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    GAMMA, NUM_VARS, NUM_ASSETS, NUM_STATES, _, _, PHI_0, PHI_1, SIGMA_VARS, P, _ = settings
    GAMMA_MINUS: float = 1 - GAMMA
    GAMMA_INVERSE: float = 1 / GAMMA_MINUS
    COVARIANCE_MATRIX: tf.Tensor = tf.concat([tf.cast(tf.linalg.cholesky(SIGMA_VARS[:NUM_VARS, :NUM_VARS]), tf.float32), tf.zeros((NUM_STATES-NUM_VARS, NUM_VARS), tf.float32)], axis=0)

    tf.test.TestCase().assertAllClose(tf.matmul(COVARIANCE_MATRIX, COVARIANCE_MATRIX, transpose_b=True), MARS_FILE["parameters"]["Sigma_vC"][0][0])

    UNCONDITIONAL_MEAN: tf.Tensor = tf.transpose(tf.linalg.inv(tf.eye(NUM_STATES) - PHI_1) @ PHI_0)[0, :]

    SIGMA_DIAGONAL_SQRT_VARS: tf.Tensor = tf.sqrt(tf.linalg.diag_part(SIGMA_VARS)[:NUM_VARS])
    SIGMA_DIAGONAL: tf.Tensor = tf.tile(SIGMA_DIAGONAL_SQRT_VARS, [P])

    HETA_RF: tf.Tensor = tf.zeros((1, NUM_STATES))
    HETA_RF = tf.tensor_scatter_nd_update(HETA_RF, [[0, 0]], tf.ones(1))
    
    HETA_R: tf.Tensor = tf.zeros((NUM_ASSETS, NUM_STATES))
    HETA_R = tf.tensor_scatter_nd_update(HETA_R, [[i, i+1] for i in range(NUM_ASSETS)], tf.ones(NUM_ASSETS))

    return GAMMA_MINUS, GAMMA_INVERSE, COVARIANCE_MATRIX, UNCONDITIONAL_MEAN, SIGMA_DIAGONAL, HETA_RF, HETA_R


def add_handler_logger(logger: logging.Logger, handler: Union[logging.FileHandler, logging.StreamHandler]) -> None:
    handler.setLevel(logger.level)
    logger.addHandler(handler)


def create_logger(logpath: str, filename: str, package_files: list[str] = [], displaying: bool = True, saving: bool = True, debug: bool = False) -> logging.Logger:
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if saving:
        logs_count = len(list(Path(logpath).glob('*.log')))
        add_handler_logger(logger, logging.FileHandler(f"{logpath}/{filename}_{logs_count}.log", "a"))
    if displaying:
        add_handler_logger(logger, logging.StreamHandler())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def create_dir_if_missing(*dirs: str) -> None:
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)

def print_tensor(x: tf.Tensor, mean=True, min=False, max=False) -> None:
    print(x, x.shape, tf.reduce_mean(x) if mean else "", tf.reduce_min(x) if min else "", tf.reduce_max(x) if max else "")


def print_tensors(*args, mean=True, min=False, max=False) -> None:
    for x in args:
        print_tensor(x, mean, min, max)
