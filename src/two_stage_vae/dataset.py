from typing import Tuple, List

import numpy as np
import os
import tensorflow_datasets as tfds
from PIL import Image
from utility import get_root_dir


def load_attribute(name, root_dir: str = os.getcwd(), category='train',
                   classes: List[int] = None) -> Tuple[np.ndarray, np.array]:
    """
    Load (CelebA) attributes

    :param name: Name of the dataset
    :param root_dir: Root folder
    :param category: train, val, test
    :param classes: List of class label indices to select
    :return: ([cnt, attributes], [labels])
    """
    data_folder = os.path.join(root_dir, 'data', name)
    attr = np.load(os.path.join(data_folder, 'attr_{}.npy'.format(category)))
    with open(os.path.join(data_folder, 'labels.txt'), 'r') as f:
        labels = np.array(f.readline().strip().split(' '))
    # select = np.array([2, 18, 19, 21, 24, 31, 36, 39])
    # select = np.array([24, 31, 36])
    if classes is None:
        return attr, labels
    else:
        return attr[:, classes], labels[classes]


def load_dataset(name, category: str, root_dir: str = get_root_dir(),
                 label: bool = False, normalize: bool = False):
    """ Load dataset (x only, unless label=True) by name.

    :param name: Name of a dataset
    :param category: {train, val, test}
    :param root_dir:
    :param label: True to load label (y)
    :param normalize: Normalize input images, x /= 255.0
    :return: (xs, shape) if not label else (ys, n_classes)
    """
    if 'svhn' == name.lower()[:4]:
        x, y = tfds.as_numpy(tfds.load('svhn_cropped',
                                       split=category,
                                       batch_size=-1,
                                       as_supervised=True, ))
        if name.lower()[5:] == 'gray':
            x = np.array([np.array(Image.fromarray(_x).convert('L'))
                          for _x in x])
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        if label:
            return y, 10
        else:
            x = x.astype(np.float32) / 255. if normalize else x
            return x, x.shape[1:]

    data_folder = os.path.join(root_dir, 'data', name)
    filename = category + '.npy' if not label else category + '_y.npy'
    path = os.path.join(data_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("Cannot load file {} for dataset {}.".format(
            path, name))

    if name.lower() in {'mnist', 'emnist', 'fashion'}:
        x = np.load(path)
        n_classes = 10
        shape = (28, 28, 1)
    elif name.lower() in {'lecs', 'lecs.1'}:
        n_classes = 10
        x = np.load(path)
        shape = (32, 32, 3)
    elif 'celeba' in name.lower():
        x = np.load(path)
        n_classes = 2
        shape = (64, 64, 3)
    elif name.lower() == 'taxinet':
        x = np.load(path)
        n_classes = 5
        shape = (100, 180, 3)
    else:
        raise Exception('No such dataset called {}.'.format(name))

    if normalize:
        x = x.astype(np.float) / 255.0
    if not label:
        return x, shape
    else:
        return x, n_classes


def get_subset(xs: np.ndarray, ys: np.ndarray, i_from: int, i_to: int, n: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """ Split the dataset into `n` consecutive subsets of the same size and
    return the ith subset.

    :param xs: Xs
    :param ys: Ys
    :param i_from: starting subset ID, 1 <= i <= n
    :param i_to: ending subset ID, 1 <= i <= n
    :param n: Number of subsets
    :return: i-th subset of (xs, ys).
    """
    assert 1 <= i_from <= n and 1 <= i_to <= n and i_from <= i_to,\
        "Invalid i_from: {}, i_to: {}".format(i_from, i_to)
    cnt = len(xs) // n
    return xs[cnt * (i_from - 1): cnt * i_to], \
           ys[cnt * (i_from - 1): cnt * i_to]


def drop_class(xs: np.ndarray, ys: np.ndarray, label: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """ For a given dataset, drop data points of the given label.

    :param xs: Xs
    :param ys: Ys
    :param label: class label to drop (int)
    :return: Filtered (xs, ys)
    """
    if label is None:
        return xs, ys
    x_filtered = np.array([x for x, y in zip(xs, ys) if y != label])
    y_filtered = np.array([y for x, y in zip(xs, ys) if y != label])
    return x_filtered, y_filtered


def drop_class_except(xs: np.ndarray, ys: np.ndarray, label: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """ For a given dataset, drop any data point except the given label.

    :param xs: Xs
    :param ys: Ys
    :param label: class label to drop (int)
    :return: Filtered (xs, ys)
    """
    x_filtered = np.array([x for x, y in zip(xs, ys) if y == label])
    y_filtered = np.array([y for x, y in zip(xs, ys) if y == label])
    return x_filtered, y_filtered


def get_test_dataset(dataset, val=False) -> Tuple[np.array, np.array]:
    """ Get a test dataset

    :param dataset: {mnist, cifar10, svhn, fashion, celeba}
    :param val: True to load validation data. False to load the test data.
    :return: (xs, ys)
    """
    if dataset == 'mnist':
        if val:
            xs, _ = load_dataset('mnist', 'test', normalize=True)
            ys, _ = load_dataset('mnist', 'test', label=True)
        else:
            x1, _ = load_dataset('emnist', 'train', normalize=True)
            x2, _ = load_dataset('emnist', 'test', normalize=True)
            y1, _ = load_dataset('emnist', 'train', label=True)
            y2, _ = load_dataset('emnist', 'test', label=True)
            xs = np.concatenate([x1, x2], axis=0)
            ys = np.concatenate([y1, y2], axis=0)
    elif dataset == 'cifar10':
        if val:
            xs, _ = load_dataset('cifar10', 'test', normalize=True)
            ys, _ = load_dataset('cifar10', 'test', label=True)
        else:
            xs, _ = load_dataset('cifar10', '10.1_v6', normalize=True)
            ys, _ = load_dataset('cifar10', '10.1_v6', label=True)
    elif dataset == 'svhn':
        if val:
            xs, _ = load_dataset('svhn', 'test', normalize=True)
            ys, _ = load_dataset('svhn', 'test', label=True)
        else:
            xs, _ = load_dataset('svhn', 'extra', normalize=True)
            ys, _ = load_dataset('svhn', 'extra', label=True)
    elif dataset == 'fashion':
        if val:
            xs, _ = load_dataset('fashion', 'test', normalize=True)
            ys, _ = load_dataset('fashion', 'test', label=True)
        else:
            raise Exception("Test dataset for fashion MNIST doesn't exist.")
    elif dataset == 'celeba':
        if val:
            xs, _ = load_dataset('celeba', 'val', normalize=True)
            ys, _ = load_dataset('celeba', 'val', label=True)
        else:
            xs, _ = load_dataset('celeba', 'test', normalize=True)
            ys, _ = load_dataset('celeba', 'test', label=True)
    else:
        raise Exception('{} dataset not supported'.format(dataset))
    return xs, ys


def load_dataset_x(name, root_dir: str = os.getcwd(), test: bool = False) -> \
        Tuple[np.array, np.array]:
    """ Load dataset input by name.

    :param name: Name of a dataset
    :param root_dir:
    :param test: True to load the test dataset
    :return: (xs, side_length, channels)
    """
    return load_dataset(name, root_dir, test, False)


def load_dataset_y(name, root_dir: str = os.getcwd(), test: bool = False) \
        -> Tuple[np.array, int]:
    """ Load dataset label by name.

    :param name: Name of a dataset
    :param root_dir:
    :param test: True to load the test dataset
    :return: (ys, n_classes)
    """
    return load_dataset(name, root_dir, test, True)

