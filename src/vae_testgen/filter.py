import numpy as np
from vae_testgen.test_generator import LECUnderTest
import os
import argparse


def get_plausibility(us):
    p_prod = 1.0 / np.power(np.e, np.power(us, 2))
    return np.mean(p_prod, axis=-1)


def get_uncertainty(model, xs):
    sigma = model.measure_uncertainty(xs, repeat=100)
    return np.mean(sigma, axis=-1)


def get_scores(model, xs, us):
    scores = [get_uncertainty(model, xs[i]) + get_plausibility(us[i])
              for i in range(xs)]
    return np.array(scores)


cwd = os.path.dirname(__file__)
test_model_dir = os.path.join(cwd, '..', 'testmodels')

parser = argparse.ArgumentParser()
parser.add_argument('output_path')
parser.add_argument('dataset')
parser.add_argument('exp_name')
parser.add_argument('testset')
args = parser.parse_args()

exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name,
                          args.testset)
x = np.load(os.path.join(exp_folder, 'x.npy'))
us = np.load(os.path.join(exp_folder, 'us.npy'))
v = np.load(os.path.join(exp_folder, 'v.npy'))
assert len(x) == len(us) == len(v)

test_model_path = os.path.join(test_model_dir, args.dataset + '.h5')
model = LECUnderTest(args.dataset, test_model_path)
scores = get_scores(model, x, us)
np.save(os.path.join(exp_folder, 's.npy'), scores)
fit = scores > np.median(scores)
np.save(os.path.join(exp_folder, 'f.npy'), fit)

