import yaml
import random
import io
import cv2
import logging
from typing import Union
import argparse

from two_stage_vae.dataset import load_dataset
from two_stage_vae.network.vae import *

import plotly.graph_objects as go
from pyclustering.cluster.clique import clique
import matplotlib.pyplot as plt
from scipy.stats import norm

from two_stage_vae.network.vae import VaeWrapper, VaeInterface
from utility import one_hot


def get_training_config(exp_dir):
    """ Load training configuration. Includes (but not limited to) [base_dim,
    batch_size, dataset, latent_dim,]

    :param exp_dir: The folder wherein the training data (models and more) is
                       stored.
    :return: A dictionary of training configuration
    """
    with open(os.path.join(exp_dir, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())
        if 'root_folder' in config:
            config['root_dir'] = config['root_folder']
            del config['root_folder']
        return config


def get_outer_vae(args: argparse.Namespace, sess: tf1.Session) -> OuterVaeModel:
    """ Create a first-stage (outer) VAE model """
    _, dim = load_dataset(args.dataset, 'train', args.root_dir)
    _, n_classes = load_dataset(args.dataset, 'train', args.root_dir,
                                label=True)
    x_holder = tf1.placeholder(tf.float32,
                               [args.batch_size, dim[0], dim[1], dim[2]], 'x')
    n_conditions = n_classes if args.conditional else 0
    if 'taxinet' == args.dataset.lower():
        model1 = TaxiNet('outer',
                         sess,
                         args.exp_dir,
                         x_holder,
                         args.batch_size,
                         args.latent_dim1,
                         n_conditions=n_conditions,
                         acai=args.acai1,
                         beta=args.beta,
                         block_per_scale=args.block_per_scale,
                         depth_per_block=args.depth_per_block,
                         kernel_size=args.kernel_size, fc_dim=args.fc_dim
                         )
    elif 'infogan2' in args.network_structure.lower():
        model1 = InfoGan2('outer', sess, args.exp_dir, x_holder,
                          args.batch_size, args.latent_dim1,
                          n_conditions=n_conditions, acai=args.acai1
                          )
    elif 'infogan' in args.network_structure.lower():
        model1 = InfoGan('outer', sess, args.exp_dir, x_holder,
                         args.batch_size, args.latent_dim1,
                         n_conditions=n_conditions, acai=args.acai1
                         )
    elif 'resnet' in args.network_structure.lower():
        model1 = ResNet('outer', sess, args.exp_dir, x_holder, args.batch_size,
                        args.latent_dim1, n_conditions=n_conditions,
                        num_scale=args.num_scale,
                        block_per_scale=args.block_per_scale,
                        depth_per_block=args.depth_per_block,
                        kernel_size=args.kernel_size, base_dim=args.base_dim,
                        fc_dim=args.fc_dim
                        )
    else:
        raise Exception("Invalid model type")

    return model1


def get_inner_vae(args: argparse.Namespace,
                  sess: tf1.Session,
                  model1: OuterVaeModel) -> InnerVaeModel:
    """ Create a second-stage VAE """
    y, n_classes = load_dataset(args.dataset, 'train', args.root_dir,
                                label=True)
    n_conditions = n_classes if args.conditional else 0
    z_holder = tf1.placeholder(tf.float32, [args.batch_size, args.latent_dim1],
                               'z')
    model2 = InnerVaeModel('inner', sess, args.exp_dir, z_holder,
                           args.batch_size, args.latent_dim2,
                           n_conditions=n_conditions, acai=args.acai2,
                           beta=args.beta, depth=args.second_depth,
                           fc_dim=args.second_dim, outer_vaes=[model1]
                           )
    return model2


def load_latent_var_weights(exp_folder: str):
    path = os.path.join(exp_folder, 'latent_weights.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)


def load_vae_model(sess: tf1.Session, exp_dir: str, dataset: str,
                   batch_size: int = 0) -> VaeWrapper:
    """ Load the two-stage VAE models

    :param sess: A tf.Session
    :param exp_dir: An experiment folder
    :param dataset: The name of the dataset
    :param batch_size: Batch size
    :param conditional: conditional VAE
    :return: a pair Two-stage VAE models
    """
    config = get_training_config(exp_dir)
    root_dir = config["root_dir"] \
        if "root_dir" in config else config["root_folder"]
    x, dim = load_dataset(dataset, 'train', root_dir)
    y, n_classes = load_dataset(dataset, 'train', root_dir,
                                label=True)
    if batch_size > 0:
        config["batch_size"] = batch_size
    n_conditions = n_classes if config["conditional"] else 0
    if 'beta' not in config:
        config['beta'] = 1.0

    input_x = tf1.placeholder(tf.float32,
                              [config['batch_size'], dim[0], dim[1], dim[2]],
                              'x')
    if dataset == 'taxinet':
        model1 = TaxiNet('outer',
                         sess,
                         exp_dir,
                         input_x,
                         config["batch_size"],
                         config["latent_dim1"],
                         n_conditions=n_conditions,
                         acai=config["acai1"],
                         beta=config["beta"],
                         block_per_scale=config["block_per_scale"],
                         depth_per_block=config["depth_per_block"],
                         kernel_size=config["kernel_size"],
                         fc_dim=config["fc_dim"]
                         )
    else:
        if config["network_structure"].lower() == 'infogan':
            model1 = InfoGan('outer',
                             sess,
                             exp_dir,
                             input_x,
                             config["batch_size"],
                             config["latent_dim1"],
                             n_conditions=n_conditions,
                             acai=config["acai1"],
                             beta=config["beta"],
                             )
        elif config["network_structure"].lower() == 'infogan2':
            model1 = InfoGan2('outer',
                              sess,
                              exp_dir,
                              input_x,
                              config["batch_size"],
                              config["latent_dim1"],
                              n_conditions=n_conditions,
                              acai=config["acai1"],
                              beta=config["beta"],
                              )
        elif config["network_structure"].lower() == 'resnset':
            model1 = ResNet('outer',
                            sess,
                            exp_dir,
                            input_x,
                            config["batch_size"],
                            config["latent_dim1"],
                            n_conditions=n_conditions,
                            acai=config["acai1"],
                            beta=config["beta"],
                            num_scale=config["num_scale"],
                            block_per_scale=config["block_per_scale"],
                            depth_per_block=config["depth_per_block"],
                            kernel_size=config["kernel_size"],
                            base_dim=config["base_dim"],
                            fc_dim=config["fc_dim"]
                            )
        else:
            raise Exception("Failed to load VAE model")
    z_holder = tf1.placeholder(tf.float32,
                               [config["batch_size"], config["latent_dim1"]],
                               'z')

    model2 = InnerVaeModel('inner',
                           sess,
                           exp_dir,
                           z_holder,
                           config["batch_size"],
                           config["latent_dim2"],
                           n_conditions=n_conditions,
                           acai=config["acai2"],
                           beta=config["beta"],
                           depth=config["second_depth"],
                           fc_dim=config["second_dim"],
                           outer_vaes=[model1]
                           )

    sess.run(tf1.global_variables_initializer())
    saver = tf1.train.Saver()
    saver.restore(sess, os.path.join(exp_dir, 'model', 'stage2'))

    vae = VaeWrapper(model1, model2)
    vae.latent_weights = load_latent_var_weights(exp_dir)
    return vae


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf1.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf1.expand_dims(image, 0)
    return image


def get_latent_plot(sess: tf1.Session, vae: VaeInterface, dataset: str):

    xs, dim = load_dataset(dataset, 'test')
    ys, n_classes = load_dataset(dataset, 'test', label=True)
    # chain of encoding
    us = vae.encode(xs, ys)

    fig = plt.figure(figsize=(12, 10))
    plt.scatter(us[:, 0], us[:, 1], c=ys, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar()
    plt.xlabel("u[0]")
    plt.ylabel("u[1]")
    # plt.savefig(output_path); plt.show()

    return fig


def create_image_grid(images: np.ndarray, row: int, col: int):
    """ Merge list of images into 2D grid

    :param images: Shape: (cnt, height, width, channel)
    :param col: number of columns
    :param row: number of rows
    :return:
    """
    assert len(images.shape) == 4
    shape = images.shape
    figure = images.reshape((row, col, shape[1], shape[2], shape[3],))
    figure = np.hstack(np.hstack(figure))
    # Drop the color channel if black-and-white
    if figure.shape[2] == 1:
        figure = figure.reshape(figure.shape[0:2])
    return figure


def visualize_2d_manifold(sess: tf1.Session, vae: VaeWrapper,
                          cnt_per_row: int = 30, bound: float = 3.0, label=None,
                          n_class=None, save_path: str = None):
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-bound, bound, cnt_per_row)
    grid_y = np.linspace(-bound, bound, cnt_per_row)[::-1]

    zs = np.array([[[xi, yi] for j, xi in enumerate(grid_x)]
                   for i, yi in enumerate(grid_y)])
    zs = np.vstack(zs)
    c = None
    if label is not None:
        c = one_hot(np.array([label] * len(zs)), n_class)
    decoded = vae.decode(zs, c=c)
    shape = decoded[0].shape
    height, width, depth = decoded.shape[1], decoded.shape[2], decoded.shape[3]
    figure = create_image_grid(decoded, cnt_per_row, cnt_per_row)

    if save_path is not None:
        plt.figure(figsize=(20, 20))
        plt.xlabel("u[0]")
        plt.ylabel("u[1]")

        # TODO 200526: refactor using width, height, depth.
        start_range = shape[0] // 2
        end_range = cnt_per_row * shape[0] + start_range + 1
        pixel_range = np.arange(start_range, end_range, shape[0])
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.imshow(figure, cmap='Greys_r' if shape[2] == 1 else None)
        plt.savefig(save_path)
    return figure


def get_trainable_parameters(name, verbose=False):
    total_params = 0
    for var in tf1.trainable_variables():
        # shape is an array of tf.compat.v1.Dimension
        if name not in var.name:
            continue
        shape = var.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim
        if verbose:
            print(var.name, var_params)
        total_params += var_params
    return total_params


def scale_up(x: np.array):
    converted = np.around(x * 255.).astype(np.uint8)
    converted = cv2.cvtColor(converted, cv2.COLOR_BGR2RGB)
    return converted

def stitch_imgs(x: np.ndarray, y: Union[np.ndarray, None],
                row_size: int = 10, col_size: int = 10):
    x_shape = x.shape
    print("stitch: x.shape", x_shape)
    assert (len(x_shape) == 4), "x_shape: {}".format(x_shape)
    output = np.zeros(
        [row_size * x_shape[1], col_size * x_shape[2], x_shape[3]])
    idx = 0
    inds = []
    for r in range(row_size):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        for c in range(col_size):
            start_col = c * x_shape[2]
            end_col = start_col + x_shape[2]
            if y is not None:
                while y[idx] != r:
                    # increment idx until a desired class is found
                    idx += 1
            output[start_row:end_row, start_col:end_col] = x[idx]
            inds.append(idx)
            idx += 1
            if idx == x_shape[0]:
                break
        if idx == x_shape[0]:
            break
    if np.shape(output)[-1] == 1:
        output = np.reshape(output, np.shape(output)[0:2])
    return output, inds


def setup_logger(filename: str, debug=False):
    # Logging
    logger = logging.getLogger("two_stage_vae")
    # handler = logging.StreamHandler()
    fhandler = logging.FileHandler(filename)
    formatter = logging.Formatter("  [%(levelname)-3.3s] %(message)s")
    # handler.setFormatter(formatter)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    # logger.addHandler(fhandler)
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.info("DEBUG flag is set.")
    else:
        logger.info("Logging at INFO level")
        logger.setLevel(logging.INFO)

    # Turn off Tensorflow logging
    debug_level = logging.DEBUG if debug else logging.WARNING
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    return logger


def analyze_manifold(sess: tf1.Session, model: VaeWrapper, dataset: str):
    # TODO (3/17)
    # fig = get_latent_plot(sess, model, dataset)
    # plt.savefig('manifold.png')
    # plt.show()
    xs, _ = load_dataset(dataset)
    visualize_2d_manifold(sess, model, xs.shape[1:3], cnt_per_row=10,
                          bound=3.0, label=5, n_class=10)
    pass


def analyze_manifold_old(model: VaeWrapper, sess, xs, ys, stage=1):
    # TODO (3/17): deprecate

    inds = list(range(len(xs)))
    cnt = 1000
    np.random.shuffle(inds)
    xs, ys = xs[inds][:cnt], ys[inds][:cnt]
    zs = model.encode(xs, stage=stage)
    zst = zs.T
    corr = np.corrcoef(zst)
    print(corr)

    # create CLIQUE algorithm for processing
    intervals = 20  # defines amount of cells in grid in each dimension
    threshold = 0
    clique_instance = clique(zs, intervals, threshold)
    # start clustering process and obtain results
    clique_instance.process()
    clusters = clique_instance.get_clusters()  # allocated clusters
    # points that are considered as outliers
    noise = clique_instance.get_noise()
    cells = clique_instance.get_cells()  # CLIQUE blocks that forms grid
    print("Amount of clusters:", len(clusters))
    encodings = clique_instance.get_cluster_encoding()
    print(encodings)

    if model.latent_dim == 2:
        # visualize clustering results
        # clique_visualizer.show_grid(cells, zs)
        # clique_visualizer.show_clusters(zs, clusters, noise)
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
        cluster_labels = clusterer.fit_predict(zs)
        print("# clusters by HDBSCAN", clusterer.labels_.max())
        print("probabilities", clusterer.probabilities_)
        print("labels", clusterer.labels_)
        clusterer.condensed_tree_.plot()
        plt.show()

    elif model.latent_dim == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=zst[0], y=zst[1], z=zst[2], mode='markers',
            marker=dict(size=2, color=ys, colorscale='Viridis', opacity=0.8),
        )])
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()


def plot_latent(vae: VaeWrapper, sess: tf1.Session, dataset: str,
                output_path: str, stage: int, posterior: bool = False):

    xs, dim = load_dataset(dataset, 'test')
    ys, n_classes = load_dataset(dataset, 'test', label=True)
    us = vae.encode(xs, stage)
    if posterior:
        assert stage == 1
        us = vae.decode(us)

    plt.figure(figsize=(12, 10))
    plt.scatter(us[:, 0], us[:, 1], c=ys, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar()
    plt.xlabel("u[0]")
    plt.ylabel("u[1]")
    plt.savefig(output_path)
    plt.show()

    return
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # print("xi: {}, yi: {}".format(xi, yi))
            z_sample = np.array([[xi, yi]])
            if vae.is_conditional_decoder:
                #TODO: FIX
                x_decoded = vae.decoder._get_discriminator([z_sample, np.array([
                    random.randint(0, 9) / 10.])])
            else:
                x_decoded = vae.decoder._get_discriminator(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def evaluate_models(sess: tf1.Session, outer: OuterVaeModel,
                    inner: InnerVaeModel, dataset: str, root_dir='.') \
        -> Tuple[float, float]:
    """ Evaluate inner and outer VAE for the MAE of reconstruction

    :param sess: tf Session
    :param outer: outer VAE
    :param inner: inner VAE
    :param dataset: Dataset
    :param root_dir: dataset root folder
    :return: (mae1, mae2)
    """
    x, dim = load_dataset(dataset, 'test', root_dir, normalize=True)
    y, n_classes = load_dataset(dataset, 'test', root_dir, label=True)
    y_encoded = one_hot(y, n_classes)

    encoded = outer.encode(x, c=y_encoded)
    mae1 = outer.evaluate(x, c=y_encoded)
    mae2 = inner.evaluate(encoded, c=y_encoded)
    return mae1, mae2


def get_semantically_partial_dataset(dataset: str, exp_dir: str,
                                     root_dir: str = os.getcwd()) \
        -> Tuple[np.array, np.array, np.array, np.array]:
    """ Get partial dataset, separated by "semantics", or semantic
    similarity captured by VAE. The VAE to use is loaded from `exp_dir`.

    :param dataset: Name of the dataset
    :param exp_dir: VAE directory
    :param root_dir: Project root directory
    :return: (x1, x2, y1, y2)
    """
    ratio = .5
    x, __ = load_dataset(dataset, 'train', root_dir, normalize=True)
    y, n_class = load_dataset(dataset, 'train', root_dir, label=True)
    c = one_hot(y, n_class)

    with tf1.Session() as sess:
        vae = load_vae_model(sess, exp_dir, dataset)
        z, __ = vae.extract_posterior(x, c)
        inds1 = [i for i, z in enumerate(z)
                 if z[:, 0] >= norm.ppf(ratio)]
        inds2 = [i for i, z in enumerate(z)
                 if z[:, 0] < norm.ppf(ratio)]
    return x[inds1], x[inds2], y[inds1], y[inds2]
