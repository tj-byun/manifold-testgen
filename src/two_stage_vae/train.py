""" Train a Two-stage VAE

Written by Taejoon Byun <taejoon@umn.edu>
Originally forked from https://github.com/daib13/TwoStageVAE
"""

import argparse
from datetime import datetime
import wandb
import os
import sys
import cv2
import numpy as np
import yaml
from typing import Tuple
import pprint
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework.ops import disable_eager_execution

import utility
from two_stage_vae.network.vae import VaeWrapper, InnerVaeModel, OuterVaeModel
import two_stage_vae.util as vae_util
from two_stage_vae.fid_score import get_fid
from two_stage_vae.dataset import load_dataset, drop_class

disable_eager_execution()
tf1.logging.set_verbosity(tf1.logging.ERROR)
logger = None


def log_epoch(epoch, total_epochs, lr, means):
    now = datetime.now()
    log = 'Date: {d}  [{0:4d}/{1:4d}]  LR: {2:.6f}, ' \
          'KL: {3:.2f}, Recon: {4:.2f}' \
        .format(epoch, total_epochs, lr, means[0], means[1],
                d=now.strftime('%m-%d %H:%M:%S'))
    logger.info(log)


def evaluate(args, model1: OuterVaeModel, model2: InnerVaeModel,
             sess: tf1.Session):

    maes = vae_util.evaluate_models(sess, model1, model2, args.dataset,
                                    args.root_dir)
    logger.info(maes)
    total_params = vae_util.get_trainable_parameters('outer')
    logger.info("stage1 trainable params: {}".format(total_params))
    total_params = vae_util.get_trainable_parameters('inner')
    logger.info("stage2 trainable params: {}".format(total_params))

    # test dataset
    x, dim = load_dataset(args.dataset, 'test', args.root_dir,
                          normalize=True)
    y, n_class = load_dataset(args.dataset, 'test', args.root_dir,
                              label=True)
    inds = np.array(list(range(len(x))))
    np.random.shuffle(inds)
    x = x[inds][0:args.fid_cnt]
    y = y[inds][0:args.fid_cnt]
    y_encoded = utility.one_hot(y, n_class) if args.conditional else None

    # reconstruction and generation
    def generate_label(cnt):
        return utility.one_hot(np.random.randint(0, n_class, cnt), n_class)

    def decode(_v):
        return np.array([np.where(__v == 1)[0][0] for __v in _v])

    img_recons = model1.reconstruct(x, c=y_encoded)
    print('recon.shape', img_recons.shape)

    y, y1, y2 = None, None, None
    img_gens1, y1 = model1.generate(args.fid_cnt, generate_label)
    img_gens2, y2 = model2.generate(args.fid_cnt, generate_label)
    logger.debug('recon.shape: {}, img1.shape: {}, img2.shape: {}'
                 ''.format(img_recons.shape, img_gens1.shape, img_gens2.shape))
    y1 = decode(y1) if y1 is not None else None
    y2 = decode(y2) if y2 is not None else None

    col = 5 if args.dataset == 'taxinet' else 10
    img_recons_sample, recon_inds = vae_util.stitch_imgs(img_recons, None,
                                                         row_size=n_class,
                                                         col_size=col)
    print('img_recons_sample: {}, recon_inds: {}'.format(
        img_recons_sample.shape, recon_inds))
    # x = np.rint(x[recon_inds] * 255.0)
    img_originals, _ = vae_util.stitch_imgs(x[recon_inds], y,
                                            row_size=n_class, col_size=col)
    print('img_originals', img_originals.shape)
    img_originals = cv2.cvtColor(img_originals.astype(np.uint8),
                                 cv2.COLOR_BGR2RGB)
    # y1, y2
    img_gens1_sample, _ = vae_util.stitch_imgs(img_gens1, y1,
                                               row_size=n_class,
                                               col_size=col)
    img_gens2_sample, _ = vae_util.stitch_imgs(img_gens2, y2,
                                               row_size=n_class,
                                               col_size=col)
    cv2.imwrite(os.path.join(args.exp_dir, 'recon_original.png'),
                img_originals)
    cv2.imwrite(os.path.join(args.exp_dir, 'recon_sample.png'),
                vae_util.scale_up(img_recons_sample))
    cv2.imwrite(os.path.join(args.exp_dir, 'gen1_sample.png'),
                vae_util.scale_up(img_gens1_sample))
    cv2.imwrite(os.path.join(args.exp_dir, 'gen2_sample.png'),
                vae_util.scale_up(img_gens2_sample))

    # calculating FID score
    batches, parallel = 100, 4
    tf1.reset_default_graph()
    fid_recon = get_fid(img_recons, args.dataset, args.root_dir,
                        args.fid_cnt, num_batches=batches, parallel=parallel)
    logger.info('FID = {:.2f}\n'.format(fid_recon))
    fid_gen1 = get_fid(img_gens1, args.dataset, args.root_dir, args.fid_cnt,
                       num_batches=batches, parallel=parallel)
    logger.info('FID = {:.2f}\n'.format(fid_gen1))
    fid_gen2 = get_fid(img_gens2, args.dataset, args.root_dir, args.fid_cnt,
                       num_batches=batches, parallel=parallel)
    logger.info('FID = {:.2f}\n'.format(fid_gen2))

    logger.info('Reconstruction Results: FID = {:.2f}'.format(fid_recon))
    logger.info('Generation Results (Stage 1): FID = {:.2f}'.format(fid_gen1))
    logger.info('Generation Results (Stage 2): FID = {:.2f}'.format(fid_gen2))

    with open(os.path.join(args.exp_dir, 'fid.txt'), 'w') as f:
        f.write("recon: {:.2f}, 1st: {:.2f}, 2nd: {:.2f}\n".format(
            fid_recon, fid_gen1, fid_gen2))
    if args.train1 and args.wandb:
        # wandb is initialized only when train1 is True
        wandb.log({
            'fid_recon': fid_recon,
            'fid_gen1': fid_gen1,
            'fid_gen2': fid_gen2,
        })


def load_configuration(args: argparse.Namespace, config_filename: str):
    if not (args.train1 or args.train2):
        logger.info("Loading saved VAE from {}".format(args.exp_name))
        if not os.path.exists(config_filename):
            logger.error("Error: Cannot load the experiment.")
            return
        exclude = {'train1', 'train2', 'eval', 'fid', 'manifold'}
        if not args.train1 and args.train2:
            # when stage 2 needs to be trained, do not load these params.
            exclude |= {'latent_dim2', 'epochs2', 'lr2', 'lr-fac2', 'acai2',
                        'second_dim', 'second_depth'}
        for name, val in yaml.load(open(config_filename, 'r')).items():
            if name not in exclude:
                setattr(args, name, val)


def save_configuration(args: argparse.Namespace, config_filename: str,
                       num_sample: int):
    """ Save the configuration into

    :param args: Configuration in Namespace
    :param config_filename: path to save the configuration
    :param num_sample: Number of samples used during training
    :return:
    """
    with open(config_filename, 'w') as outfile:
        args.num_sample = num_sample
        yaml.dump(vars(args), outfile, default_flow_style=False)


def interpolate(vae, sess, exp_folder, xs, ys, cnt):
    # TODO (200622): move to `experiment.py`.
    mix_zs, mix_ys = [], []
    INTERP_CNT = 11
    for _ in range(cnt):
        while True:
            idx = np.random.randint(0, len(xs), 2)
            y1, y2 = tuple(ys[idx])
            if y1 == y2:
                break

        y_encoded = utility.one_hot(ys[idx], n_class=np.max(ys) + 1)
        zs = vae.encode(xs[idx], y_encoded)
        z0, z1 = np.expand_dims(zs[0], 0), np.expand_dims(zs[1], 0)
        alpha = np.linspace(0, 1.0, num=INTERP_CNT).reshape((INTERP_CNT, 1))
        mix = np.matmul(alpha, z0) + np.matmul(1 - alpha, z1)
        y = np.array([y_encoded[0]] * INTERP_CNT)
        mix_zs.append(mix)
        mix_ys.append(y)

    mix_zs = np.concatenate(mix_zs, axis=0)
    mix_ys = np.concatenate(mix_ys, axis=0)

    decoded = vae.decode(mix_zs, mix_ys)
    decoded = np.rint(decoded * 255)
    imgs, _ = vae_util.stitch_imgs(decoded, None, row_size=cnt,
                                   col_size=INTERP_CNT)
    cv2.imwrite(os.path.join(exp_folder, 'interpolation.png'), imgs)


def main(args):
    global logger
    tf1.reset_default_graph()

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    model_path = os.path.join(args.exp_dir, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger = vae_util.setup_logger(os.path.join(args.exp_dir, 'training.log'),
                                   args.debug)
    logger.info("Experiment at {}".format(args.exp_dir))
    logger.info(vars(args))

    # dataset
    xs, dim = load_dataset(args.dataset, 'train', args.root_dir,
                           normalize=True)
    ys, n_class = load_dataset(args.dataset, 'train', args.root_dir,
                               label=True)
    if args.limit:
        xs, ys = xs[:6400], ys[:6400]
    logger.info('Train data len: {}, dim: {}, classes: {}'.format(len(xs),
                                                                  dim, n_class))

    xs_val, _ = load_dataset(args.dataset, 'test', args.root_dir,
                             normalize=True)
    ys_val, _ = load_dataset(args.dataset, 'test', args.root_dir, label=True)

    if args.drop >= 0:
        logger.info("Dropping class {}".format(args.drop))
        xs, ys = drop_class(xs, ys, args.drop)
        xs_val, ys_val = drop_class(xs_val, ys_val, args.drop)
    cs = utility.one_hot(ys)

    n_sample = np.shape(xs)[0]
    logger.info('Num Sample = {}.'.format(n_sample))

    # Load from configuration
    config_filename = os.path.join(args.exp_dir, 'config.yml')
    load_configuration(args, config_filename)
    pprinter = pprint.PrettyPrinter(indent=4)
    logger.info("Configuration: {}".format(pprinter.pformat(vars(args))))

    # Save/update the config only when any of the VAE gets trained
    if args.train1 or args.train2:
        logger.info("Saving configuration to " + config_filename)
        save_configuration(args, config_filename, n_sample)

    # session
    config = tf1.ConfigProto(device_count={'GPU': 0}) if args.cpu else None
    sess = tf1.Session(config=config)

    # model
    outer_vae = vae_util.get_outer_vae(args, sess)
    outer_params = vae_util.get_trainable_parameters('outer')
    logger.info("Created VAE models:")
    logger.info("{}, {} params".format(outer_vae, outer_params))

    # train model
    if args.train1:
        if args.wandb:
            wandb.init(project=args.dataset, name=args.exp_name,
                       sync_tensorboard=True, config=args)
        mae = outer_vae.train(lambda: (xs, cs), args.epochs1, args.lr1,
                              os.path.join(model_path, 'stage1'),
                              log_epoch=log_epoch)
        logger.info("Finished training stage 1 VAE. Mae: {:.2%}".format(mae))

    if args.train2:
        inner_vae = vae_util.get_inner_vae(args, sess, outer_vae)
        sess.run(tf1.global_variables_initializer())
        outer_vae.restore(os.path.join(model_path, 'stage1'))

        mu_z, sd_z = outer_vae.extract_posterior(xs, cs)

        def get_data():
            zs = mu_z + sd_z * np.random.normal(0, 1,
                                                [len(mu_z), args.latent_dim1])
            return zs, cs

        mae = inner_vae.train(get_data, args.epochs2, args.lr2,
                              os.path.join(model_path, 'stage2'),
                              log_epoch=log_epoch)
        logger.info("Finished training stage 2 VAE. Mae: {:.2%}".format(mae))

    # load
    if not (args.train1 or args.train2):
        # saver.restore(sess, os.path.join(model_path, 'stage1'))
        if os.path.exists(os.path.join(model_path, 'stage2.index')):
            inner_vae = vae_util.get_inner_vae(args, sess, outer_vae)
            inner_vae.restore(os.path.join(model_path, 'stage2'))
            logger.info("Loaded Stage 2 VAE")
        elif os.path.exists(os.path.join(model_path, 'stage1.index')):
            outer_vae.restore(os.path.join(model_path, 'stage1'))
            logger.info("Loaded Stage 1 VAE")
        else:
            raise Exception("No checkpoint found!")

    if args.eval:
        logger.info("Evaluating...")
        evaluate(args, outer_vae, inner_vae, sess)

    if args.interpolate:
        interpolate(VaeWrapper(outer_vae, inner_vae), sess, args.exp_dir, xs, ys, 20)

    if args.manifold:
        logger.info("Analyze manifold")
        vae_util.analyze_manifold(sess, VaeWrapper(outer_vae, inner_vae), args.dataset)


def build_arg_parser() -> argparse.ArgumentParser:
    """ Build an argument parser and return """
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help="Directory wherein experiment result is (to be) "
                             "stored. It should be structured as "
                             "<output_dir>/<dataset>/<exp_name>")

    parser.add_argument('--root-dir', type=str, default=os.getcwd())
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--network-structure', type=str, default='InfoGan')
    parser.add_argument('--batch-size', type=int, default=64)

    # stage 1
    parser.add_argument('--epochs1', type=int, default=500)
    parser.add_argument('--latent-dim1', type=int, default=64)
    parser.add_argument('--lr1', type=float, default=0.0001)
    parser.add_argument('--acai1', type=float, default=0.0)
    # parser.add_argument('--lr-epochs', type=int, default=150)
    parser.add_argument('--lr-fac', type=float, default=0.5)

    # stage 2
    parser.add_argument('--latent-dim2', type=int, default=64)
    parser.add_argument('--acai2', type=float, default=0.0)
    parser.add_argument('--epochs2', type=int, default=1000)
    parser.add_argument('--lr2', type=float, default=0.0001)
    # parser.add_argument('--lr-epochs2', type=int, default=300)
    parser.add_argument('--lr-fac2', type=float, default=0.5)
    parser.add_argument('--second-dim', type=int, default=1024)

    parser.add_argument('--num-scale', type=int, default=4)
    parser.add_argument('--block-per-scale', type=int, default=1)
    parser.add_argument('--depth-per-block', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--base-dim', type=int, default=16)
    parser.add_argument('--fc-dim', type=int, default=256)
    parser.add_argument('--second-depth', type=int, default=4)

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--interpolate', default=False, action='store_true')
    parser.add_argument('--train1', default=False, action='store_true')
    parser.add_argument('--train2', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--fid-cnt', type=int, default=10000)
    parser.add_argument('--limit', action='store_true', default=False)

    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--regression', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--manifold', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help="Disable GPU computation")
    parser.add_argument('--limit-gpu', type=float, default=0.9,
                        help="Limit gpu usage by the provided factor.")
    parser.add_argument('--drop', type=int, default=-1,
                        help="Drop a specified class in training dataset")
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    _args = utility.parse_and_process_args(parser, sys.argv[1:])

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

    main(_args)

