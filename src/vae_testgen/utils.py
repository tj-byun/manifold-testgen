import logging

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import stats as stats
import tensorflow.compat.v1 as tf1

P_STANDARD_NORMAL = 0.3989422804014327


def plot_images(x_hats: np.array, dim: tuple, img_path: str = "xhats.png"):
    """ Plot synthesized inputs

    :param x_hats: List of synthesized inputs in numpy format
    :param dim: dimension (width, height)
    :param img_path: Path to save the image
    """
    _x_hats = x_hats * 255
    _x_hats = x_hats.reshape(len(x_hats), dim[0], dim[1], dim[2])
    if x_hats.shape[1] > 500:
        _x_hats = x_hats.reshape((x_hats.shape[0], dim[0], dim[1]))
    m = math.ceil(math.sqrt(len(_x_hats)))
    figure = np.zeros((dim[0] * m, dim[1] * m))
    reached_the_end = False
    for i in range(m):
        for j in range(m):
            if i * m + j >= len(_x_hats):
                reached_the_end = True
                break
            figure[i * dim[0]: (i + 1) * dim[0],
            j * dim[1]: (j + 1) * dim[1]] = _x_hats[i * m + j]
        if reached_the_end:
            break

    plt.figure(figsize=(10, 10))
    #plt.imshow(figure, cmap='Greys_r')
    #plt.savefig(img_path)
    #plt.show()
    figures_scaled = np.around(figure / 255.).astype(int)
    cv2.imwrite(img_path, figures_scaled)


# TODO (7/28): remove
def plot_digits(x_hats, img_path="xhats.png"):
    """ Plot synthesized inputs

    :param x_hats: List of synthesized inputs in numpy format
    :param img_path: Path to save the image
    """
    digit_size = 28

    _x_hats = x_hats * 255
    _x_hats = x_hats.reshape(len(x_hats), 28, 28)
    if x_hats.shape[1] == 784:
        _x_hats = x_hats.reshape((x_hats.shape[0], 28, 28))
    m = math.ceil(math.sqrt(len(_x_hats)))
    figure = np.zeros((digit_size * m, digit_size * m))
    reached_the_end = False
    for i in range(m):
        for j in range(m):
            if i * m + j >= len(_x_hats):
                reached_the_end = True
                break
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = \
                _x_hats[i * m + j]
        if reached_the_end:
            break

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(img_path)
    plt.show()


def setup_logger(debug=False):
    # Logging
    logger = logging.getLogger("vae_testgen.testgen")
    handler = logging.StreamHandler()
    # %(asctime)s
    formatter = logging.Formatter("  [%(levelname)-3.3s] %(message)s")
    handler.setFormatter(formatter)
    # logger.addHandler(handler)
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.info("DEBUG flag is set.")
    else:
        logger.info("Logging at INFO level")
        logger.setLevel(logging.INFO)

    # Turn off Tensorflow & Pyswarms logging
    debug_level = logging.DEBUG if debug else logging.WARNING
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    logging.getLogger("pyswarms").setLevel(debug_level)
    logging.getLogger("pyswarms.single.general_optimizer").setLevel(debug_level)
    return logger


def unit_gauss_pdf(x: np.ndarray) -> np.ndarray:
    """ The PDF of the unit Gaussian distribution """
    # 0.39894 ~= 1 / sqrt(2 * pi)
    return 0.3989422804014327 * np.exp(-.5 * np.power(x, 2))


def normal_pdf(m: float, s: float, x: float) -> float:
    """ Probability density function of a normal distribution

    :param m: mu
    :param s: sigma
    :param x: value to predict probability
    :return: N(m, s).pdf(x)
    """
    a = (x - m) / s
    return (P_STANDARD_NORMAL / s) * np.exp(-.5 * a * a)


def normal_pval(m: float, s: float, x: float) -> float:
    p = stats.norm(m, s).cdf(x)
    if p > 0.5:
        p = 1 - p
    return 2 * p


def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)