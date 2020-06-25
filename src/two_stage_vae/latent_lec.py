import os
import argparse
import math
import time
import yaml
import random
import numpy as np
import functools

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras.layers as layers
from tensorflow.compat.v1.metrics import accuracy

import two_stage_vae.dataset as data
import two_stage_vae.network.util as net_util
import two_stage_vae.util as vae_util
import utility
from two_stage_vae.network.vae import VaeWrapper
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Callable
from tqdm import tqdm
from scipy.stats import norm

# tf1.summary.image('confusion', self.confusion_matrix),


def extend(x: np.array, batch_size: int):
    if len(x) >= batch_size:
        return np.concatenate([x, x[0:batch_size]], 0)
    else:
        return np.concatenate([x] * batch_size, 0)


def combine_u_c(u: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.vstack((u.T, c.T)).T


class LatentSpaceLEC(object):

    def __init__(self, sess: tf1.Session, n_class: int, n_condition: int,
                 latent_dim: int = 128, batch_size: int = 64):
        self.sess = sess
        self.n_class = n_class
        self.n_condition = n_condition
        self.is_conditional = self.n_condition > 0
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        # self.is_training = tf1.placeholder(tf.bool, [], 'is_training')

        self.z = tf1.placeholder(name='z',
                                 shape=[None, self.latent_dim],
                                 dtype=tf.float32)
        if self.is_conditional:
            self.c = tf1.placeholder(name='c',
                                     shape=[None, self.n_condition],
                                     dtype=tf.float32)
        self.y = tf1.placeholder(name='y',
                                 shape=[None, self.n_class],
                                 dtype=tf.uint8)
        self.dropout = tf1.placeholder_with_default(True, shape=())
        with tf1.variable_scope('', reuse=tf1.AUTO_REUSE):
            self.global_step = tf1.get_variable('latent_lec_global_step', [],
                                                tf.int32,
                                                tf1.zeros_initializer(),
                                                trainable=False)
        # self.y = tf.one_hot(self.y_raw, 10, on_value=1.0, off_value=0.0,
        # axis=-1)

        self.__build_network()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()
        self.__init_vars()
        self.saver = tf1.train.Saver()

    def __build_network(self):
        with tf1.variable_scope("latent_lec", reuse=tf1.AUTO_REUSE):
            if self.is_conditional:
                y = tf1.concat([self.z, self.c], -1)
            else:
                y = self.z
            y = tf1.layers.Dense(128, activation='relu', name='d1')(y)
            y = tf1.layers.Dense(128, activation='relu', name='d2')(y)
            y = tf1.layers.Dense(64, activation='relu', name='d3')(y)
            y = tf1.layers.Dense(64, activation='relu', name='d4')(y)
            y = tf1.layers.Dense(64, activation='relu', name='d5')(y)
            y = tf1.layers.Dense(64, activation='relu', name='d6')(y)
            y = tf1.layers.Dropout(0.3)(y, training=self.dropout)
            y = tf1.layers.Dense(32, activation='relu', name='d7')(y)
            y = tf1.layers.Dense(16, activation='relu', name='d8')(y)
            self.y_pred = tf1.layers.Dense(self.n_class, tf1.nn.softmax,
                                           name='d9')(y)

    def __build_loss(self):
        with tf1.variable_scope("latent_lec", reuse=tf1.AUTO_REUSE):
            self.loss = tf1.losses.softmax_cross_entropy(self.y, self.y_pred)
            self.accuracy = accuracy(tf.argmax(self.y, axis=1),
                                     tf.argmax(self.y_pred, axis=1))

    def __build_summary(self):
        with tf1.variable_scope('summary', reuse=tf1.AUTO_REUSE):
            summary = [tf1.summary.scalar('loss', self.loss), ]
            # tf1.summary.scalar('accuracy_per_class', self.pc_acc),
            self.summary = tf1.summary.merge(summary)

    def __build_optimizer(self):
        variables = [var for var in tf1.global_variables() if 'latent_lec' in
                     var.name]
        self.lr = tf1.placeholder(tf.float32, [], 'lr')
        self.opt = tf1.train.AdamOptimizer(self.lr).minimize(
            self.loss, self.global_step, var_list=variables)

    def measure_accuracy(self, label: np.array, z: np.array, c: np.array) -> \
            float:
        """ Predict and measure the accuracy, with respect to the given label.

        :param label: Label. No one-hot.
        :param z: integral-transformed z
        :param c: condition (can be None)
        :return: 0. <= Accuracy <= 1.
        """
        assert len(label) == len(z) and len(label.shape) == 1
        y = self.predict(z, c)
        n_correct = np.count_nonzero(np.argmax(y, axis=1) == label)
        return n_correct / len(label)

    def predict(self, z_uni: np.ndarray, cs: np.ndarray = None,
                dropout: bool = False) -> np.ndarray:
        """ Predict y from z

        :param z_uni: zs, uniform-ed
        :param cs: 'c's
        :param dropout: True to turn on test time dropout
        :return: y logits
        """
        assert np.count_nonzero((z_uni < 0.) + (z_uni > 1.)) == 0,\
            "z_uni should be in the range of 0.0 to 1.0"
        if self.is_conditional:
            in_tensors, in_arrays = [self.z, self.c], [z_uni, cs]
        else:
            in_tensors, in_arrays = [self.z], [z_uni]
        return self.__run_batch([self.y_pred], in_tensors, in_arrays)[0]

    def __init_vars(self):
        c_vars = [v for v in tf1.global_variables() if 'latent_lec' in v.name]
        # print("init_vars::c_vars", c_vars)
        c_vars += tf1.get_collection(tf1.GraphKeys.LOCAL_VARIABLES,
                                     scope="accuracy")
        self.sess.run(tf1.variables_initializer(c_vars))
        self.sess.run(tf1.local_variables_initializer())
        self.sess.run(tf1.global_variables_initializer())

    def __process_metric(self, _metric):
        print("metric.shape", _metric.shape)
        _mean = np.mean(_metric)
        _metric = np.array(_metric)[:, :, 0]
        _per_class = np.around(np.mean(_metric, axis=1) * 100, decimals=1)
        return _mean, _per_class

    def __print_summary(self, outputs, epoch, epochs, _lr):
        # [self.loss, self.opt, self.accuracy, self.precision, self.recall]
        loss, acc = outputs[0], outputs[2]
        # mean_acc, accuracies = self.__process_metric(outputs[2:self.n_class
        # + 2])
        print('({date}) [{0:3d}/{1:3d}] lr: {lr:.6f} L: {2:.6f} Acc: {3:.6f}'
              .format(epoch, epochs, np.mean(loss), np.mean(acc),
                      date=time.strftime('%H:%M:%S'), lr=_lr))

    def train(self, encode: Callable, y: np.ndarray,
              c: np.ndarray = None, epochs: int = 20, lr: float = 0.0001,
              lr_epochs: int = 10, lr_fac: float = 0.5,
              encode_frequency: int = 10) -> None:
        """ Train a latent-space LEC (a classifier or regressor)

        :param encode: a curried encoder function
        :param y: Label. Either the truth or the output of the LEC under test
        :param c: condition
        :param epochs: Number of epochs
        :param lr: learning rate
        :param lr_epochs: period at which to drop the learning rate
        :param lr_fac: lr drop factor (0.5 by default)
        :param encode_frequency: the VAE encoder is probabilistic,
        and the latent-space LEC (this class) has to take that into account
        during training. But how often shall we call the encoder function?
        Calling it every epoch will be expensive and slow down the training
        by a lot. When called too infrequently, we miss capturing the
        randomness. Right balance is the key.
        """

        def get_lr(_epoch):
            if lr_epochs <= 0:
                return lr
            else:
                return lr * math.pow(lr_fac, float(_epoch) // float(lr_epochs))

        self.__init_vars()
        vs = [v.name for v in tf1.get_default_graph().as_graph_def().node]
        with open('variables.txt', 'w') as f:
            for v in vs:
                f.write(v + '\n')

        ind = list(range(len(y)))
        z = None
        for i, epoch in enumerate(range(epochs)):
            np.random.shuffle(ind)
            if i % encode_frequency == 0:
                z = encode()

            _lr = get_lr(epoch)
            out_tensors = [self.loss, self.opt, self.accuracy]
            if self.is_conditional:
                input_tensor = [self.z, self.c, self.y]
                input_array = [z[ind], c[ind], y[ind]]
            else:
                input_tensor = [self.z, self.y]
                input_array = [z[ind], y[ind]]
            outputs = self.__run_batch(out_tensors, input_tensor,
                                       input_array, [self.lr], [_lr])
            self.__print_summary(outputs, epoch, epochs, _lr)
            del outputs

    def __run_batch(self, output_tensors: List[tf.Tensor],
                    array_input_tensors: List[tf.Tensor],
                    array_inputs: List[np.ndarray],
                    scalar_input_tensors: List[tf.Tensor] = None,
                    scalar_inputs: List = None) -> Tuple:
        out_list = []
        for __ in range(len(output_tensors)):
            out_list.append(list())

        original_cnt = len(array_inputs[0])
        array_inputs = [extend(item, self.batch_size) for item in array_inputs]
        cnt = len(array_inputs[0])
        for ia, ib, __ in net_util.minibatch_idx_generator(cnt,
                                                           self.batch_size):
            feed = {k: v[ia:ib] for k, v in
                    zip(array_input_tensors, array_inputs)}
            if scalar_input_tensors and scalar_inputs:
                feed.update({k: v for k, v in zip(scalar_input_tensors,
                                                  scalar_inputs)})
            outs = self.sess.run(output_tensors, feed_dict=feed)
            for i, out in enumerate(outs):
                out_list[i].append(out)

        return_values = []
        for out in out_list:
            if type(out[0]) == np.ndarray:
                out = np.concatenate(out[:original_cnt], axis=0)
            return_values.append(out[:original_cnt])
        return tuple(return_values)

    def get_uncertainty(self, sess: tf1.Session, latent_vectors: np.ndarray,
                        is_regression: bool = False, repeat: int = 100):
        """ Calculate uncertainty for the given list of latent_vectors

        :param sess: tf1.Session
        :param latent_vectors: list of latent_vectors (as produced by encoder)
        :param is_regression: True if the output is continuous (regression)
        :param repeat: How many times to repeat MC/MC sampling
        :return: sigma
        """
        outs = np.array([self.predict(sess, latent_vectors, dropout=True)
                        for _ in range(repeat)])
        if is_regression:
            return outs.std(axis=0).sum(axis=1)
        else:
            # sum of softmaxes across samples:
            #     $ p(y=c | x, X, Y) \approx $
            #     $ \frac{1}{T} \sum_{t=1}^{T} {\it softmax}(f^{\hat{w_t}}(x)) $
            # shape: [input x output_vector]
            psum = np.sum(outs, axis=0) / float(len(outs))
            # $ H(p) = - \sum_{c=1}^{C} p_c \log p_c $
            get_h = lambda p: - p * np.ma.log2(p)
            # since log(0) is not defined, the value gets masked when p
            # equals 0. Fill the masked with 0.0 since the limit of p *
            # log(p) when p -> 0 is 0.0 .
            sigmas = np.ma.filled(np.sum(get_h(psum), axis=1), 0.0)
            return sigmas

    def load(self, model_path: str):
        assert os.path.exists(model_path), "Invalid model path " + model_path
        self.saver.restore(self.sess, os.path.join(model_path, 'latent_lec'))

    def save(self, exp_folder: str):
        model_path = os.path.join(exp_folder, 'model', 'latent_lec')
        self.saver.save(self.sess, model_path)


def get_encode(sess: tf1.Session, vae: VaeWrapper, x: np.array,
               c: np.array, stage: int, probabilistic: bool = True) -> Callable:
    """ Curry the encode function

    :param sess: tf1 Session
    :param vae: Vae wrapper instance
    :param x: input
    :param c: condition
    :param stage: 1 or 2
    :param probabilistic: when False, turn off randomness
    :return: function that encodes and normalizes
    """
    def _encode():
        """ Encode and apply probability integral transform """
        if probabilistic:
            z = vae.encode(x, c, stage=stage)
        else:
            z, __ = vae.extract_posterior(x, c, stage=stage)
        return norm.cdf(z)
    return _encode


def train_latent_lec(args):
    print('exp_folder: {}, dataset: {}'.format(args.exp_dir, args.dataset))

    tf1.reset_default_graph()
    tf_cfg = tf1.ConfigProto(device_count={'GPU': 0}) if args.cpu else None
    with tf1.Session(config=tf_cfg) as sess_vae:
        vae_config = vae_util.get_training_config(args.exp_dir)
        vae = vae_util.load_vae_model(sess_vae, args.exp_dir, args.dataset)

        x, _ = data.load_dataset(args.dataset, 'train',
                                 vae_config["root_dir"], normalize=True)
        c, n_class = data.load_dataset(args.dataset, 'train',
                                       vae_config["root_dir"], label=True)
        y, __ = data.load_dataset(args.dataset, 'train',
                                  vae_config["root_dir"], label=True)
        n_condition = n_class if vae_config["conditional"] else 0
        if args.drop:
            print("Dropping class {}".format(args.drop))
            # x, c = data.drop_class(x, c, args.drop)
            x, y = data.drop_class(x, y, args.drop)

        lec = tf.keras.models.load_model(args.lec_path)
        y_lec = lec.predict(x)

        # Train the latent space LEC
        with tf1.Session(config=tf_cfg) as sess_classifier:
            latent_lec = LatentSpaceLEC(sess_classifier, n_class, n_condition,
                                        latent_dim=vae_config["latent_dim1"],
                                        batch_size=64)

            c_one_hot = utility.one_hot(c, n_condition) \
                if vae_config["conditional"] else None
            y_one_hot = utility.one_hot(y, n_class)
            encode = get_encode(sess_vae, vae, x, c=c_one_hot, stage=args.stage)
            latent_lec.train(encode, y_one_hot, c_one_hot,
                             epochs=args.epochs, lr=args.lr,
                             lr_epochs=args.lr_epochs,
                             encode_frequency=10)
            latent_lec.save(args.exp_dir)
    # test_classifier(exp_folder, dataset, cnt=128)


def load_latent_lec(sess: tf1.Session, exp_dir: str, batch_size: int = 64) \
        -> LatentSpaceLEC:
    """ Load a classifier model in the given experiment folder

    :param sess:
    :param exp_dir: The folder in which the model checkpoint is stored
    :param batch_size: Batch size
    :return: A VaeClassifier
    """
    vae_config = vae_util.get_training_config(exp_dir)
    dataset = vae_config["dataset"]
    __, n_class = data.load_dataset(dataset, 'train',
                                   vae_config["root_dir"], label=True)
    n_condition = n_class if vae_config["conditional"] else 0
    classifier = LatentSpaceLEC(sess, n_class, n_condition,
                                latent_dim=vae_config["latent_dim1"],
                                batch_size=batch_size)
    model_path = os.path.join(exp_dir, 'model')
    classifier.load(model_path)
    return classifier


def get_lec_prediction(model_path, x):
    lec = tf.keras.models.load_model(model_path)
    y_lec = lec.predict(x)
    return y_lec


def test_latent_lec(exp_dir: str, lec_path: str, dataset: str,
                    on_cpu: bool = False, drop: int = None, stage: int = 1):
    vae_config = vae_util.get_training_config(exp_dir)
    x, y = data.get_test_dataset(dataset)
    if drop:
        x, y = data.drop_class_except(x, y, drop)
        print("dropped {}. len: {}".format(drop, len(x)))

    assert len(x) == len(y), "len(x): {}, len(y): {}".format(len(x), len(y))
    conditional = vae_config["conditional"]
    c_encoded = utility.one_hot(y) if conditional else None

    tf1.reset_default_graph()
    tf_cfg = tf1.ConfigProto(device_count={'GPU': 0}) if on_cpu else None

    with tf1.Session(config=tf_cfg) as sess_vae:
        vae_model = vae_util.load_vae_model(sess_vae, exp_dir, dataset,
                                            batch_size=64)
        encode = get_encode(sess_vae, vae_model, x, c=c_encoded,
                            stage=stage, probabilistic=False)
        z = encode()

    y_lec = get_lec_prediction(lec_path, x)

    tf1.reset_default_graph()
    with tf1.Session(config=tf_cfg) as sess_classifier:
        latent_lec = load_latent_lec(sess_classifier, exp_dir)
        acc_truth = latent_lec.measure_accuracy(y, z, c_encoded)
        acc_lec = latent_lec.measure_accuracy(np.argmax(y_lec, axis=1), z,
                                              c_encoded)
        print("acc_truth: {:.2%}, acc_lec: {:.2%}".format(acc_truth, acc_lec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help="Directory wherein experiment result is (to be) "
                             "stored. output_path/dataset/exp_name")
    parser.add_argument('lec_path', type=str,
                        help="Path of the model under test")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train a classifier model. Requires "
                             "two-stage VAEs to be trained a priori")
    parser.add_argument('--test', action='store_true', default=False,
                        help="Load the trained classifier and measure its "
                             "accuracy")
    parser.add_argument('--epochs', type=int, default=40,
                        help="Training epoch")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument('--lr-epochs', type=int, default=100,
                        help="Learning rate decay epoch")
    parser.add_argument('--cpu', default=False, action='store_true',
                        help="Disable GPU computation")
    parser.add_argument('--drop', type=int, default=None,
                        help="Drop a specified class in training dataset")
    parser.add_argument('--stage', type=int, default=1,
                        help="1st / 2nd stage latent space to consider")
    parser.add_argument("--limit-gpu", type=float, default=0.5,
                        help='Limit GPU mem usage by percentage 0 < f <= 1')
    _args = utility.parse_and_process_args(parser)

    if _args.train:
        train_latent_lec(_args)
    if _args.test:
        test_latent_lec(_args.exp_dir, _args.lec_path, _args.dataset,
                        on_cpu=_args.cpu, drop=_args.drop, stage=_args.stage)

    if not (_args.train or _args.test):
        print("You might want to --train or --test.")

