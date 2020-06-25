import numpy as np
import os
import sys
from keras.models import load_model
import cv2
from PIL import ImageFont, ImageDraw, Image
from two_stage_vae import limit_keras_gpu_usage
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dirs', nargs='+', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()
print(args)

if os.path.exists(args.output_dir):
    assert False, "Output dir {} already exists".format(args.output_dir)
else:
    os.mkdir(args.output_dir)

xs, ys, vs = [], [], []

for input_dir in args.input_dirs:
    xs.append(np.load(os.path.join(input_dir, 'x.npy')))
    ys.append(np.load(os.path.join(input_dir, 'y.npy')))
    vs.append(np.load(os.path.join(input_dir, 'v.npy')))

Xs = np.concatenate(xs, axis=0)
Ys = np.concatenate(ys, axis=0)
Vs = np.concatenate(vs, axis=0)

np.save(os.path.join(args.output_dir, "x.npy"), Xs)
np.save(os.path.join(args.output_dir, "y.npy"), Ys)
np.save(os.path.join(args.output_dir, "v.npy"), Vs)

