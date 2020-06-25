from two_stage_vae import get_trainable_parameters, get_outer_vae
from two_stage_vae import load_dataset
import numpy as np
import argparse
import cv2
import tensorflow as tf
from two_stage_vae import evaluate_fid_score2
import os
import random
from copy import copy


parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('output_path')
parser.add_argument('--cnt', type=int, default=2048)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--valid-only', action='store_true')
args = parser.parse_args()

# load test data
x = np.load(os.path.join(args.output_path, 'x.npy'))
v = np.load(os.path.join(args.output_path, 'v.npy'))
if args.valid_only:
    valids, invalids = [], []
    for i, valid in enumerate(v):
        if valid:
            valids.append(x[i])
        else:
            invalids.append(x[i])
    random.shuffle(invalids)
    x = invalids[:324]
    x = valids
# extend
if len(x) < args.cnt:
    times = int(np.ceil(args.cnt / len(x)))
    x_copies = [copy(x) for _ in range(times)]
    x = np.concatenate(x_copies, axis=0)[:args.cnt]

# get test dataset
x_test, dim = load_dataset(args.dataset, '.', test=True)


sess = tf.InteractiveSession()
tf.reset_default_graph()

print("Measuring FID {} times".format(args.repeat))
fids = []
for i in range(args.repeat):
    inds = np.array(list(range(len(x_test))))
    np.random.shuffle(inds)
    x_test = x_test[inds][0:args.cnt]

    fid_x = evaluate_fid_score2(x[:args.cnt], x_test)
    print('({:2d}/{:2d}) FID = {:.2f}\n'.format(i, args.repeat, fid_x))
    fids.append(fid_x)
fids = np.array(fids)
print("Mean: {:.2f},  SD: {:2f}".format(fids.mean(), fids.std()))


