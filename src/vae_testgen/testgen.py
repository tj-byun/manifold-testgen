"""
Manifold-based test generation.
https://arxiv.org/abs/2002.06337

Author: Taejoon Byun <taejoon@umn.edu>

# Particle Swarm Optimizer Hyper-parameters
- c1: cognitive parameter (how much confidence a particle has in itself)
- c2: social parameter (how much confidence a particle has in its
      neighbors)
- w: inertial weight (higher values allow particles to escape local
     minima)
- k: number of nearest neighbors for ring structure
- p: L1 or L2 distance computation

"""

from datetime import datetime
import sys
import os
import argparse

import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf1
from tensorflow.python.framework.ops import disable_eager_execution

from two_stage_vae.dataset import load_dataset
from two_stage_vae.latent_lec import load_latent_lec
from two_stage_vae.util import load_vae_model, get_training_config
from vae_testgen.test_generator import CostFunctionFactory, VaeFuzzer, \
    VaeTestGenerator, LECUnderTest
from vae_testgen.utils import setup_logger
from utility import parse_and_process_args, set_root_dir, get_root_dir

logger = None
DEBUG = False
cwd = os.path.dirname(__file__)

disable_eager_execution()


def print_cost(_calculate_cost, us):
    cost_dict = pd.DataFrame(_calculate_cost(us, return_dict=True))
    pd.options.display.float_format = '{:.3f}'.format
    print(cost_dict)


def get_batch_size(n_test) -> int:
    """ Get an optimal batch size based on the number of tests to generate

    :param n_test: Number of test cases to generate
    :return: Batch size, power of two
    """
    return 2 ** min(int(np.floor(np.log(n_test) / np.log(2))), 10)


def get_pso_option(args):
    return {'c1': args.c1, 'c2': args.c2, 'w': args.w, 'k': args.k, 'p': args.p}


def main_random(args):
    setup_globals(args)

    testset_dir = os.path.join(args.exp_dir, args.name)

    # Load the VAE and VAE classifier
    sess_vae = tf1.Session()
    vae_model = load_vae_model(sess_vae, args.exp_dir, args.dataset,
                               batch_size=get_batch_size(args.cnt))

    # Load test model
    logger.info("Testing {} model at {}".format(args.dataset, args.lec_path))
    target_model = LECUnderTest(args.dataset, args.lec_path)

    # Test generator
    fuzzer = VaeFuzzer(vae_model, target_model, args.dataset,
                       vae_model.latent_dim, testset_dir)
    us, xs, ys = fuzzer.generate(args.cnt)

    __, dim = load_dataset(args.dataset, 'train', root_dir=get_root_dir())
    fuzzer.save_to_npy(xs, ys, us, dim)

    # cost_factory = CostFunctionFactory(None, None, vae_model, sess_vae,
    #                                    target_model)
    # calc_cost = cost_factory.get_conditional_cost(ys)
    # cost = calc_cost(us, return_dict=True)
    # print(sorted(cost['plaus']))


def main_search(args):
    # TODO(200624): Fix and refactor!
    setup_globals(args)
    # Load the VAE and VAE classifier
    sess_vae, sess_classifier = tf1.Session(), tf1.Session()
    vae_model = load_vae_model(sess_vae, args.exp_dir, args.dataset,
                               batch_size=get_batch_size(args.cnt))
    try:
        latent_lec = load_latent_lec(sess_classifier, args.exp_dir,
                                     batch_size=get_batch_size(args.cnt))
    except:
        latent_lec = None

    # Load test model
    logger.info("Testing {} model at {}".format(args.dataset, args.lec_path))
    target_model = LECUnderTest(args.dataset, args.lec_path)

    # Test generator
    testset_dir = os.path.join(args.exp_dir, args.name)
    # sigma_threshold = target_model.get_uncertainty_threshold()

    gtor = VaeTestGenerator(vae_model, latent_lec, target_model,
                            args.dataset, testset_dir, get_pso_option(args),
                            args.n_iter)
    cost_factory = CostFunctionFactory(latent_lec, sess_classifier, vae_model,
                                       sess_vae, target_model)
    us, xs, ys = gtor.optimize_conditional(cost_factory, args.cnt,
                                           reshape=True)

    logger.info("intended labels: " + str(ys))
    calculate_cost = cost_factory.get_conditional_cost(ys,
                                                       plaus_weight=args.plaus)
    print_cost(calculate_cost, us)
    __, dim = load_dataset(args.dataset, 'train', root_dir=get_root_dir())
    gtor.save_to_npy(xs, ys, us, dim)

    # log(testset_dir, str(datetime.datetime.now()))
    # log(testset_dir, 'Total: {}, bug finding: {}\n'.format(gtor.total_cnt,
    #                                                        args.n_test))

    sess_vae.close()
    sess_classifier.close()


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help="Directory wherein VAE is saved")
    parser.add_argument('lec_path', type=str,
                        help="Path of the LEC model under test")
    parser.add_argument('cnt', type=int, help="Number of tests to generate")
    parser.add_argument('--name', type=str,
                        default=datetime.now().strftime('%y%m%d-%H%M'),
                        help="Name of the generating test set")
    parser.add_argument("--search", action="store_true", default=False,
                        help="Generate by applying search on latent space")
    parser.add_argument("--plaus", type=float, default=3,
                        help='Weight of the plausibility term')
    parser.add_argument("--label", type=int, default=-1,
                        help="Target class label to synthesize towards")
    parser.add_argument("--label2", type=int, default=-1,
                        help="2nd target class label to synthesize towards")

    # program configurations
    parser.add_argument('--root-dir', type=str, default=os.getcwd())
    parser.add_argument("--debug", action='store_true', default=False,
                        help='Debug mode')
    parser.add_argument("--limit-gpu", type=float, default=0.4,
                        help='Limit GPU mem usage by percentage 0 < f <= 1')

    # PSO (particle swarm optimizer) parameters
    parser.add_argument("--n-iter", type=int, default=10, help='PSO iterations')
    parser.add_argument("--c1", type=float, default=.8,
                        help='PSO c1, cognitive parameter: how much '
                             'confidence a particle has in itself')
    parser.add_argument("--c2", type=float, default=.3,
                        help='PSO c2, social parameter: how much confidence a'
                             'particle has in its neighbors')
    parser.add_argument("--w", type=float, default=.2,
                        help='PSO w, inertial weight: higher values allow '
                             'particles to more easily escape local minima')
    parser.add_argument("--k", type=int, default=4,
                        help='PSO k, number of nearest neighbors for ring '
                             'structure')
    parser.add_argument("--p", type=int, default=1,
                        help='PSO p: L1 or L2 distance computation')
    return parser


def setup_globals(args: argparse.Namespace):
    global DEBUG, logger
    logger = setup_logger(debug=args.debug)
    DEBUG = args.debug
    set_root_dir(args.root_dir)


if __name__ == "__main__":
    parser = build_argument_parser()
    _args = parse_and_process_args(parser, sys.argv[1:])

    if _args.search_based:
        main_search(_args)
    else:
        main_random(_args)
