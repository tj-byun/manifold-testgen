import tensorflow.compat.v1 as tf1
from tensorflow import keras

import utility
from coverage import utils

import argparse
import numpy as np
import os
from tensorflow.keras.models import load_model
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from scipy import stats
import seaborn as sn
import pickle
from sklearn.metrics import roc_auc_score


import two_stage_vae.util as vae_util
import two_stage_vae.dataset as data
from two_stage_vae import latent_lec

# from scipy.interpolate import griddata

logger = None
batch_size = 128


def get_model_name(model_path):
    return os.path.split(model_path)[1].split('.')[0]


def get_figure_path(args, name):
    model_name = os.path.split(args.lec_path)[1].split('.')[0]
    fig_dir = os.path.join(args.exp_dir, 'figures')
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    return os.path.join(fig_dir, '{}-{}.png'.format(name, model_name))


def exp_heatmap(args):
    """ Visualize a contour map of 2D latent space to sigma
    """
    is_val = False
    sess = tf1.Session()
    vae = vae_util.load_vae_model(sess, args.exp_dir, args.dataset,
                                  batch_size=batch_size)
    mae, _ = vae_util.evaluate_models(sess, vae.outer, vae.inner,
                                      args.dataset)
    logger.info("Mae: {}".format(mae))
    assert not vae.is_conditional

    x, y = data.get_test_dataset(args.dataset, val=is_val)
    if args.drop:
        x, y = data.drop_class_except(x, y, args.drop)
    z, _ = vae.outer.extract_posterior()
    z = stats.norm.cdf(z)
    x_hat = vae.outer.reconstruct(x)
    recon_err = np.abs(x - x_hat).mean(axis=1).mean(axis=1).mean(axis=1)

    x, y, z = z[:, 0], z[:, 1], recon_err
    print('mu_x: {} - {}, mu_y: {} - {}'.format(x.min(), x.max(), y.min(),
                                                y.max()))
    xi = np.linspace(0., 1., 200)
    yi = np.linspace(0., 1., 200)
    zi = griddata(x, y, z, xi, yi, interp='linear')  # , # interp='linear')
    plt.title("{} and {} to {}".format('mu[0]', 'mu[1]', 'sd'))
    plt.xlabel('mu[0]')
    plt.ylabel('mu[1]')
    ctr = plt.contourf(xi, yi, zi, levels=32)
    plt.colorbar(ctr)
    plt.savefig(get_figure_path(args, 'heatmap'))


def exp_conf_matrix(args):
    xs, ys = data.get_test_dataset(args.dataset, val=True)
    model = load_model(args.lec_path)
    preds = model.predict(xs)
    preds = np.argmax(preds, axis=1)
    plt.figure(figsize=(10, 10))
    cmat = confusion_matrix(ys, preds, labels=list(range(10)))
    for i in range(10):
        cmat[i][i] = 0
    df = pd.DataFrame(cmat, range(10), range(10))
    sn.heatmap(df, annot=True)
    plt.savefig(get_figure_path(args, 'confusion'), dpi=100)


def exp_scatter(args):
    """ Draw scatter plot of the 2D manifold. color-code samples by
    category--train (green), normal (blue), fault-finding (red).
    """
    # configuration
    figsize = (20, 20)
    vmin, vmax = -3., 3.
    plot_manifold = True
    plot_train = True
    plot_normal = False
    plot_ff = True
    is_val = False
    uniform = True
    if uniform:
        vmin, vmax = 0., 1.

    sess = tf1.Session()
    vae = vae_util.load_vae_model(sess, args.exp_dir, args.dataset,
                                  batch_size=batch_size)
    x_train, _ = data.load_dataset(args.dataset, 'train', normalize=True)
    y_train, _ = data.load_dataset(args.dataset, 'train', label=True)
    if args.drop:
        logger.info("Dropping class {}".format(args.drop))
        x_train, y_train = data.drop_class(x_train, y_train, args.drop)
    model = load_model(args.lec_path)
    xs, ys = data.get_test_dataset(args.dataset, val=is_val)
    if args.drop:
        xs, ys = data.drop_class_except(xs, ys, args.drop)
    preds = model.predict(xs)
    preds = np.argmax(preds, axis=1)

    # divide the test dataset into normal vs. fault-revealing subsets
    ff_inputs = [x for x, y, _y in zip(xs, ys, preds) if y != _y]
    normal_inputs = [x for x, y, _y in zip(xs, ys, preds) if y == _y]
    print("ff: {}, normal: {}".format(len(ff_inputs), len(normal_inputs)))

    def encode(_x: np.array):
        _z, _ = vae.outer.extract_posterior()
        return stats.norm.cdf(_z) if uniform else _z

    z_train = encode(x_train)
    z_ff = encode(np.array(ff_inputs))
    z_normal = encode(np.array(normal_inputs))

    def per_axis(_z: np.ndarray):
        return (_z[:, i] for i in range(_z.shape[1]))

    plt.figure(figsize=figsize)
    if plot_manifold and vae.latent_dim == 2:
        save_path = get_figure_path(args, 'manifold')
        img = vae_util.visualize_2d_manifold(sess, vae, bound=vmax,
                                             cnt_per_row=40,
                                             save_path=save_path)
        plt.clf()

    if vae.latent_dim == 2:
        ax = plt
    elif vae.latent_dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim(vmin, vmax)
    else:
        raise Exception("Invalid latent dimension {}".format(vae.latent_dim))

    if plot_normal:
        ax.scatter(*per_axis(z_normal), vmin=vmin, vmax=vmax, s=1,
                   c='deepskyblue')
    if plot_train:
        ax.scatter(*per_axis(z_train), vmin=vmin, vmax=vmax, s=1,
                   c='lightgreen')
    if plot_ff:
        ax.scatter(*per_axis(z_ff), vmin=vmin, vmax=vmax, s=1, c='r')

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    if args.show:
        plt.show()
    plt.savefig(get_figure_path(args, 'scatter'), dpi=100)

    # Save figure handle to disk
    with open(get_figure_path(args, 'scatter') + '.pickle', 'wb') as f:
        pickle.dump(plt.gcf(), f)


def exp_proximity(args):
    """ Pick clusters of samples by latent space proximity and visualize """
    n_repeat, n_sample = 10, 10

    sess = tf1.Session()
    vae = vae_util.load_vae_model(sess, args.exp_dir, args.dataset,
                                  batch_size=batch_size)
    x_train, _ = data.load_dataset(args.dataset, 'train', normalize=True)
    xs, ys = data.get_test_dataset(args.dataset, val=False)
    zs = vae.outer.encode(xs)

    inds = []
    for i in range(n_repeat):
        z_ind = np.random.randint(len(xs))
        z_picked = zs[z_ind]
        dist = [(i, np.linalg.norm(z_picked - z),) for i, z in enumerate(zs)]
        dist.sort(key=lambda x: x[1])
        # pick the closest `n_samples` samples and extract their indices
        inds += [tup[0] for tup in dist[:n_sample]]

    figure = vae_util.create_image_grid(xs[inds], n_repeat, n_sample)
    plt.imshow(figure)
    plt.savefig(get_figure_path(args, 'proximity'))


def get_signals(args: argparse.Namespace, x: np.array):
    signals = dict()

    tf1.reset_default_graph()
    with tf1.Session() as sess:
        vae = vae_util.load_vae_model(sess, args.exp_dir, args.dataset,
                                      batch_size=batch_size)
        mae, __ = vae_util.evaluate_models(sess, vae.outer, vae.inner,
                                           args.dataset)
        logger.info("Mae: {}".format(mae))
        assert not vae.is_conditional
        z, __ = vae.outer.extract_posterior()
        x_hat = vae.outer.reconstruct(x)
        x_dist = np.abs(x - x_hat).mean(axis=1).mean(axis=1).mean(axis=1)
        signals['x_dist'] = x_dist

        u, __ = vae.inner.extract_posterior()
        z_hat = vae.inner.decode(u)
        z_uni, z_hat_uni = stats.norm.cdf(z), stats.norm.cdf(z_hat)
        z_dist = np.abs(z_uni - z_hat_uni).mean(axis=1)
        signals['z_dist'] = z_dist

    lec = load_model(args.lec_path)
    y_lec = lec.predict(x)

    tf1.reset_default_graph()
    with tf1.Session() as sess:
        llec = latent_lec.load_latent_lec(sess, args.exp_dir)
        y_latent = llec.predict(z_uni)
        # stats.entropy
        # signals['entropy_latent'] = np.linalg.norm(y_latent)
        signals['y_dist'] = np.linalg.norm(y_latent - y_lec, axis=1)
        # signals['y_dist_lec'] = stats.entropy(y_lec, y_latent)

    # signals['entropy_lec'] = stats.entropy(y_lec)
    print(signals)
    return np.vstack([signals['x_dist'], signals['z_dist'],
                      signals['y_dist']]).reshape(len(x), 3)


def get_ood_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(9, activation='relu', use_bias=True))
    model.add(keras.layers.Dense(9, activation='relu', use_bias=True))
    model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))

    adam = keras.optimizers.Adam(lr=0.00001)
    model.compile(adam, loss=keras.losses.binary_crossentropy,
                  metrics=['mae', 'accuracy'])
    model.summary()
    return model


def plot_roc(group1: np.array, group2: np.array):
    """ Plot histogram and ROC (receiver operating characteristics)

    :param group1:
    :param group2:
    :return:
    """

    pass


def plot_histogram(group1: np.array, group2: np.array):
    pass


def get_roc_auc(group1: np.array, group2: np.array):
    pass


def exp_plot_ood(args):
    x1, x2, y1, y2 = vae_util.get_semantically_partial_dataset(args.dataset,
                                                               args.exp_dir)

def exp_train_ood_classifier(args):
    """ How well does each metric separate the in-distribution and
    out-of-distribution samples?
    """
    x_train, __ = data.load_dataset(args.dataset, 'train', normalize=True)
    y_train, __ = data.load_dataset(args.dataset, 'train', label=True)
    # x_train, y_train = data.get_test_dataset(args.dataset, val=False)
    if args.drop:
        x_train, y_train = data.drop_class_except(x_train, y_train, args.drop)

    x_val, y_val = data.get_test_dataset(args.dataset, val=True)
    if args.drop:
        x_val, y_val = data.drop_class_except(x_val, y_val, args.drop)

    signal_train = get_signals(args, x_train)
    signal_val = get_signals(args, x_val)

    lec = load_model(args.lec_path)
    y_pred_train = lec.predict(x_train)
    y_pred_val = lec.predict(x_val)
    ood_train = (y_train != np.argmax(y_pred_train, axis=1)).astype(int)
    ood_val = (y_val != np.argmax(y_pred_val, axis=1)).astype(int)

    model = get_ood_model((signal_train.shape[1],))
    model.fit(signal_train, ood_train,
              validation_data=(signal_val, ood_val),
              epochs=100, batch_size=16, shuffle=True)
    print('weights', model.layers[-1].get_weights())


def main(args):
    logger.info('In main')
    if args.mode in {'scatter'}:
        exp_scatter(args)
    elif args.mode == 'heatmap':
        exp_heatmap(args)
    elif args.mode == 'proximity':
        exp_proximity(args)
    elif args.mode in {'conf_matrix', 'confusion', 'conf'}:
        exp_conf_matrix(args)
    elif args.mode == 'train_ood':
        exp_train_ood_classifier(args)
    elif args.mode == 'plot_ood':
        exp_plot_ood(args)
    else:
        raise Exception("Invalid mode {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help="Directory wherein experiment result is (to be) "
                             "stored. output_path/dataset/exp_name")
    parser.add_argument("lec_path", type=str, help='Path of the LEC under test')
    parser.add_argument('mode', type=str,
                        help="{coverage, ff, cramers, t, ood}")

    parser.add_argument('--drop', type=int,
                        help="Drop a specified class in training dataset")
    parser.add_argument('--show', action='store_true',
                        help="Show interactive plot")
    parser.add_argument('--cnt', default=100, type=int,
                        help='The size of the test pool')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='Debug mode')
    parser.add_argument("--limit-gpu", type=float, default=0.5,
                        help='Limit GPU mem usage by percentage 0 < f <= 1')
    _args = utility.parse_and_process_args(parser)

    logger = utils.setup_logger('experiment', debug=_args.debug)
    logger.propagate = False

    main(_args)
