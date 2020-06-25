"""
Evaluate the performance of the test generator by running the synthesized
test inputs against pre-trained MNIST models.
Author: Taejoon (TJ) Byun
"""

import os
import argparse
import numpy as np
import keras.models
from autoencoder.vae import VaeModel, VaeClassifier
import matplotlib.pyplot as plt
from autoencoder.utils import plot_digits
try:
    from vae_testgen.test_generator import VaeTestGenerator
except SystemError:
    from test_generator import VaeTestGenerator


def evaluate(n_tests):
    # Load models
    models = [keras.models.load_model(m) for m in model_paths]
    ref_model = keras.models.load_model(ref_model_path)

    # init generators
    vae = VaeModel(saved_weight_path=vae_weights_path)
    classifier = VaeClassifier(vae, load=True, weights=classifier_weights_path)
    generator = VaeTestGenerator(vae, classifier)

    # generate
    xs = generator.synthesize(n_tests, reshape=True)

    # evaluate
    ref_logit = ref_model._get_discriminator(xs)
    ref_label = np.argmax(ref_logit, axis=1)
    logits_per_model = [model._get_discriminator(xs) for model in models]
    label_per_model = [np.argmax(l, axis=1) for l in logits_per_model]
    xentropy_per_model = [- np.sum(p * np.log(ref_logit)) for p in
                          logits_per_model]

    # print and plot the result
    print("truth:", ref_label)
    for i in range(len(models)):
        print("label: {}, x-entropy: {:.2f}".format(label_per_model[i],
                                                    xentropy_per_model[i]))
    plot_digits(xs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_tests", type=int, help="Number of test to generate")
    return parser.parse_args()


IMG_DIM = 28

cwd = os.path.dirname(__file__)
model_dir = os.path.join(cwd, "mnist_models")
ref_model_path = os.path.join(model_dir, "emnist-cnn_huge_mid-127.h5")
model_paths = ["mnist-cnn_small.h5", "mnist-cnn_big.h5", "mnist-cnn_huge.h5"]
model_paths = [os.path.join(model_dir, p) for p in model_paths]
figure_path = os.path.join(cwd, "generated_xs.png")

vae_weights_path = os.path.join("autoencoder", "mnist_vae.h5")
classifier_weights_path = os.path.join("autoencoder", "mnist_classifier.h5")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.n_tests)
