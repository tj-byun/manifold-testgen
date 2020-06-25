import os.path
import numpy as np
from scipy import linalg
from six.moves import range
import numpy as np
from tensorflow_gan.python.eval import inception_metrics
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from two_stage_vae import dataset as datasets
import functools


def get_inception_activations_helper(input_tensor1, input_tensor2, classifier_fn,
                                     num_batches=1, parallel=1):
    """ A helper function for evaluating the fréchet classifier distance.
    copied from: https://github.com/tensorflow/gan
    """

    input_list1 = tf.split(input_tensor1, num_or_size_splits=num_batches)
    input_list2 = tf.split(input_tensor2, num_or_size_splits=num_batches)
    stack1 = tf.stack(input_list1)
    stack2 = tf.stack(input_list2)

    # Compute the activations using the memory-efficient `map_fn`.
    def compute_activations(elems):
        return tf.map_fn(
            fn=classifier_fn,
            elems=elems,
            parallel_iterations=parallel,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier'
        )

    print('\t...computing activations for stack 1')
    at1 = compute_activations(stack1)
    print('\t...computing activations for stack 2')
    at2 = compute_activations(stack2)
    # Ensure the activations have the right shapes.
    at1 = tf.concat(tf.unstack(at1), 0)
    at2 = tf.concat(tf.unstack(at2), 0)
    print('\t...done')
    return at1, at2


get_inception_activations = functools.partial(
    get_inception_activations_helper,
    classifier_fn=inception_metrics.classifier_fn_from_tfhub(
        inception_metrics.INCEPTION_TFHUB,
        inception_metrics.INCEPTION_FINAL_POOL, True))


def measure_fid(codes_g, codes_r, eps=1e-6):
    """ real / fake """
    # print("[fid_score] real.shape: {}, fake.shape: {}".format(
    # codes_g.shape, codes_r.shape))
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d
    
    mn_g = codes_g.mean(axis=0)
    mn_r = codes_r.mean(axis=0)
    cov_g = np.cov(codes_g, rowvar=False)
    cov_r = np.cov(codes_r, rowvar=False)

    cov_mean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
    if not np.isfinite(cov_mean).all():
        cov_g[range(d), range(d)] += eps
        cov_r[range(d), range(d)] += eps 
        cov_mean = linalg.sqrtm(cov_g.dot(cov_r))

    return np.sum((mn_g - mn_r) ** 2) + (
                np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(cov_mean))


def preprocess_fake_images(fake_images: np.ndarray):
    if np.shape(fake_images)[-1] == 1:
        fake_images = np.concatenate([fake_images, fake_images, fake_images], -1) 
    return np.rint(fake_images * 255).astype(np.float32)
    # return np.rint(fake_images).astype(int)


def preprocess_real_images(real_images):
    if np.shape(real_images)[-1] == 1:
        real_images = np.concatenate([real_images, real_images, real_images], -1)
    return real_images.astype(np.float32)


def get_fid(fake_images, dataset, root_folder, n, num_batches=1,
            parallel=1):
    real_images, _ = datasets.load_dataset(dataset, 'train', root_folder,
                                           normalize=False)
    np.random.shuffle(real_images)
    real_images = preprocess_real_images(real_images[:n])
    fake_images = preprocess_fake_images(fake_images[:n])

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        at1, at2 = get_inception_activations(real_images, fake_images,
                                             num_batches=num_batches,
                                             parallel=parallel)
        score = measure_fid(at1.eval(), at2.eval())

    return float(score)

