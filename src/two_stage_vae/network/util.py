import numpy as np
import math
from typing import Tuple

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1 import variable_scope, get_variable

he = tf.initializers.he_uniform
l2 = None


def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf1.reshape(input_, [-1, input_.get_shape().as_list()[-1]])

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf1.nn.l2_normalize(tf1.matmul(tf.transpose(w), u), dim=None,
                                epsilon=1e-12)
        u = tf1.nn.l2_normalize(tf1.matmul(w, v), dim=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf1.assign(u_var, u, name="update_u")]):
        u = tf1.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf1.stop_gradient(u)
    v = tf1.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf1.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf1.reshape(w_normalized, input_.shape)
    return w_tensor_normalized


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf1.truncated_normal_initializer, use_sn=False):
    with variable_scope(name):
        w = get_variable(
            "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=initializer(stddev=stddev))
        if use_sn:
            conv = tf1.nn.conv2d(input_, spectral_norm(w),
                                 strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf1.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1],
                                 padding="SAME")
        biases = get_variable(
            "biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0,
           use_sn=False):
    shape = input_.get_shape().as_list()

    with variable_scope(scope or "Linear"):
        matrix = get_variable("Matrix", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer(stddev=stddev))
        bias = get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias_start))
        if use_sn:
            return tf.matmul(input_, spectral_norm(matrix)) + bias
        else:
            return tf.matmul(input_, matrix) + bias


def lrelu(input_, leak=0.2, name="lrelu"):
    return tf.maximum(input_, leak * input_, name=name)


def batch_norm(x, is_training, scope, eps=1e-5, decay=0.999, affine=True):

    with variable_scope(scope + '_w'):
        def mean_var_with_update(moving_mean, moving_variance):
            if len(x.get_shape().as_list()) == 4:
                statistics_axis = [0, 1, 2]
            else:
                statistics_axis = [0]
            _mean, variance = tf1.nn.moments(x, statistics_axis, name='moments')
            with tf.control_dependencies(
                    [assign_moving_average(moving_mean, _mean, decay),
                     assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(_mean), tf.identity(variance)

        params_shape = x.get_shape().as_list()[-1:]
        moving_mean = get_variable('mean', params_shape,
                                   initializer=tf.zeros_initializer(),
                                   trainable=False)
        moving_var = get_variable('variance', params_shape,
                                  initializer=tf.ones_initializer,
                                  trainable=False)

        mean, var = tf1.cond(is_training,
                             lambda: mean_var_with_update(moving_mean,
                                                          moving_var),
                             lambda: (moving_mean, moving_var))
        if affine:
            beta = get_variable('beta', params_shape,
                                initializer=tf.zeros_initializer())
            gamma = get_variable('gamma', params_shape,
                                 initializer=tf.ones_initializer)
            return tf1.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        else:
            return tf1.nn.batch_normalization(x, mean, var, None, None, eps)


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02,
             name="deconv2d"):
    with variable_scope(name):
        w = get_variable("w",
                         [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                         initializer=tf.random_normal_initializer(
                             stddev=stddev))
        deconv = tf1.nn.conv2d_transpose(input_, w,
                                         output_shape=output_shape,
                                         strides=[1, d_h, d_w, 1])
        biases = get_variable("biases", [output_shape[-1]],
                              initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


def down_sample(x, out_dim, kernel_size, name):
    with variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert (len(input_shape) == 4)
        x = tf1.layers.conv2d(x, out_dim, kernel_size, 2, 'same')
        print("downsample; x.shape: ", x.shape)
        return x


def up_sample(x, out_dim, kernel_size, name):
    with variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert (len(input_shape) == 4)
        x = tf1.layers.conv2d_transpose(x, out_dim, kernel_size, 2, 'same')
        print("upsample; x.shape: ", x.shape)
        return x


def res_block(x, out_dim, is_training, name, depth=2, kernel_size=3):
    with variable_scope(name):
        y = x
        for i in range(depth):
            y = tf.nn.relu(batch_norm(y, is_training, 'bn' + str(i)))
            y = tf1.layers.conv2d(y, out_dim, kernel_size, padding='same',
                                  name='layer' + str(i))
        s = tf1.layers.conv2d(x, out_dim, kernel_size, padding='same',
                              name='shortcut')
        return y + s


def res_fc_block(x, out_dim, is_training, name, depth=2):
    with variable_scope(name):
        y = x
        for i in range(depth):
            y = tf1.layers.dense(tf.nn.relu(y), out_dim, name='layer' + str(i),
                                 kernel_regularizer=l2)
        s = tf1.layers.dense(x, out_dim, name='shortcut',
                             kernel_regularizer=l2)
        return y + s


def scale_block(x, out_dim, is_training, name, block_per_scale=1,
                depth_per_block=2, kernel_size=3):
    with variable_scope(name):
        y = x
        for i in range(block_per_scale):
            y = res_block(y, out_dim, is_training, 'block' + str(i),
                          depth_per_block, kernel_size)
        print("scale_block; x.shape: ", y.shape)
        return y


def scale_fc_block(x, out_dim, is_training, name, block_per_scale=1,
                   depth_per_block=2):
    with variable_scope(name):
        y = x
        for i in range(block_per_scale):
            y = res_fc_block(y, out_dim, is_training, 'block' + str(i),
                             depth_per_block)
        return y


def dense_batch_norm_relu(x, dim, depth=1, encoder=True, drop_factor=0.75):
    t = x
    for i in range(depth):
        if encoder:
            _dim = int(dim // (drop_factor ** i))
        else:
            _dim = int(dim // (drop_factor ** (depth - i - 1)))

        t = tf1.layers.dense(t, _dim, name='fc' + str(i))
        t = tf1.layers.batch_normalization(t, name='bn' + str(i))
        t = tf.nn.relu(t)
    return t


def cosine_decay(global_step,
                 lr_base,
                 total_steps,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_steps=0):
    """ https://gist.github.com/Tony607/28b8de1cd01a859e62cc77547d601fb5
    Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps <= warmup_steps:
        raise ValueError('total_steps must be larger than warmup_steps.')
    learning_rate = 0.5 * lr_base * (1 + np.cos(
        np.pi * (global_step - warmup_steps - hold_steps) / float(
            total_steps - warmup_steps - hold_steps)))
    if hold_steps > 0:
        learning_rate = np.where(
            global_step > warmup_steps + hold_steps,
            learning_rate, lr_base)
    if warmup_steps > 0:
        if lr_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (lr_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def minibatch_idx_generator(n_sample: int, batch_size: int) \
        -> Tuple[int, int, bool]:
    """ Generate mini-batch indices

    :param n_sample: number of samples, len(x)
    :param batch_size: size of the mini-batch. e.g. 32, 64
    :return: (start_index: int, end_index: int, is_last_item: bool)
    """
    num_iter = math.ceil(float(n_sample) / float(batch_size))
    for i in range(num_iter):
        yield i * batch_size, (i + 1) * batch_size, i == num_iter - 1

