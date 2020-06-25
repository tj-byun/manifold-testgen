import argparse
import pprint

import numpy as np
import os
from typing import List
import tensorflow.compat.v1 as tf1

__root_dir = '.'


def parse_and_process_args(parser: argparse.ArgumentParser,
                           args: List[str] = None) -> argparse.Namespace:
    """ parse the argument and apply some post-processing

    :param parser: ArgumentParser instance
    :param args: Command-line arguments
    :return: Argument namespace
    """
    _args = parser.parse_args(args)

    if _args.exp_dir[-1] == '/' or _args.exp_dir[-1] == '\\':
        _args.exp_dir = _args.exp_dir[:-1]
    t1, t2 = os.path.split(_args.exp_dir)
    (_args.output_path, _args.dataset), _args.exp_name = os.path.split(t1), t2
    if _args.dataset == 'mnist-old':
        _args.dataset = 'mnist'

    print("Command-line arguments:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(_args))

    if hasattr(_args, 'limit_gpu'):
        limit_keras_gpu_usage(_args.limit_gpu)

    return _args


def limit_keras_gpu_usage(fraction: float):
    assert 0. < fraction <= 1.
    tf_config = tf1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = fraction
    tf1.keras.backend.set_session(tf1.Session(config=tf_config))


def set_root_dir(root_dir: str):
    global __root_dir
    __root_dir = root_dir


def get_root_dir() -> str:
    return __root_dir


def one_hot(targets: np.array, n_class=None) -> np.array:
    if type(targets[0]) in [np.array, np.ndarray, list]:
        # the target is already encoded
        return targets
    if n_class is None:
        n_class = np.max(targets) + 1
    res = np.eye(n_class)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [n_class])