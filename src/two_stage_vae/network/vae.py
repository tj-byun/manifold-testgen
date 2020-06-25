from typing import Callable, List
from numba import jit
from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from tensorflow.compat.v1.summary import scalar, histogram
from two_stage_vae.network.util import *
from utility import one_hot

he = tf.initializers.he_uniform
HALF_LN_TWO_PI = 0.91893853


def extend(x: np.array, batch_size: int):
    if len(x) >= batch_size:
        return np.concatenate([x, x[0:batch_size]], 0)
    else:
        return np.concatenate([x] * batch_size, 0)


def mini_batch_gtor(xs: np.ndarray, ys: np.ndarray, batch_size: int,
                    verbose: bool = False)\
        -> Tuple[np.ndarray, np.ndarray, float]:
    """ Generate mini-batch for given pair of xs and ys.

    :param xs: inputs np.ndarray
    :param ys: label (condition) np.ndarray
    :param batch_size: batch size
    :param verbose: print progress if True
    :return: input mini-batch, condition mini-batch, is_last
    """
    n_iter = math.ceil(len(xs) / batch_size)
    iterator = range(n_iter)
    if verbose:
        iterator = tqdm(iterator)
    for i in iterator:
        ia, ib = i * batch_size, (i + 1) * batch_size
        if i < n_iter - 1:
            yield xs[ia:ib], ys[ia:ib] if ys is not None else None, False
        else:
            pad_idx = batch_size - len(xs[ia:ib])
            # print('ia {}, ib {}, len {}, pad_idx {}'
            # .format(ia, ib, len(xs[ia:ib]), pad_idx))
            _xs = np.concatenate([xs[ia:ib], xs[:pad_idx]], 0)
            if ys is not None:
                _ys = np.concatenate([ys[ia:ib], ys[:pad_idx]], 0)
            yield _xs, _ys if ys is not None else None, True
    return


class VaeInterface(ABC):

    @abstractmethod
    def generate(self, n_samples: int, generate_label: Callable = None):
        """ Synthesize new inputs, sampling from Gaussian prior.

        :param n_samples: Number of inputs to synthesize
        :param generate_label: A function for sampling method
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self, x: np.array, c: np.array = None) -> np.ndarray:
        """ Encode a given input to a latent code

        :param x: input array
        :param c: condition array
        :return: latent encoding of shape (sample, dim)
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z: np.array, c: np.array = None) -> np.ndarray:
        """ Decode given latent codes into input space

        :param z: latent code of shape (#sample, latent_dim)
        :param c: One-hot encoded condition (#sample,n_class)
        :return: Synthesized inputs (sample,whatever_input_shape)
        """
        raise NotImplementedError()

    @abstractmethod
    def reconstruct(self, x: np.array, c: np.array = None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def extract_posterior(self, x, c=None) -> Tuple[np.array, np.array]:
        raise NotImplementedError()


class VaeModel(VaeInterface):

    def __init__(self, name: str, sess: tf1.Session, exp_dir: str,
                 x: tf1.placeholder, batch_size: int, latent_dim: int,
                 n_conditions: int = 0, acai: float = 0.0, beta: float = 1.0):
        """ Create a VAE model

        :param name: Name of the model
        :param sess: tf1 Session
        :param exp_dir: Save dir
        :param x: place-holder tensor for input
        :param batch_size:
        :param latent_dim: Size of the latent dimension
        :param n_conditions: Number of classes. Positive value makes a
        conditional VAE, and 0 makes it un-conditional (default)
        :param acai: Enable Adversarially Constrained Autoencoder
        Interpolation. A positive value enables it, and adjust the weight of
        the ACAI term.
        :param beta: Weight of the KL divergence term. 1.0 by default.
        """

        assert 0.0 <= acai < 100.0
        self.name = name
        self.sess = sess
        self.exp_dir = exp_dir
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.acai = acai   # adversarially constrained autoencoder interpolation
        self.beta = beta
        self.n_conditions = n_conditions
        self.is_conditional = n_conditions > 0

        # initialize tensors
        self.x = x
        with variable_scope(self.name, reuse=tf1.AUTO_REUSE):
            if self.is_conditional:
                self.c = tf1.placeholder(tf.float32, [batch_size, n_conditions],
                                         'c')
            self.is_training = tf1.placeholder(tf.bool, [], 'is_training')
            self.lr = tf1.placeholder(tf.float32, [], 'lr')
            self.global_step = tf1.get_variable('global_step', [], tf.int32,
                                                tf.zeros_initializer(),
                                                trainable=False)
        self.latent_plot = tf1.placeholder(tf.uint8, shape=(1, None, None, 4))

        # self.x_hat, self.gamma, self.log_gamma = None, None, None
        # self.mu, self.sd, self.log_sd = None, None, None
        self.__build_encoder()
        self.__build_decoder()
        if self.acai:
            self.__build_acai()
        self.__build_loss()
        self.__build_optimizer()

        self.summary_list = self._get_summaries()
        self.summary = tf1.summary.merge(self.summary_list)

        # utilities
        self.saver = tf1.train.Saver()
        self.writer = tf1.summary.FileWriter(self.exp_dir, self.sess.graph)

    def __str__(self):
        return "{:16s}(name: {}, batch_size: {}, latent_dim: {} " \
               "n_conditions: {}, acai: {}, beta: {})" \
               "".format(self.__class__.__name__, self.name, self.batch_size,
                         self.latent_dim, self.n_conditions, self.acai,
                         self.beta)

    @abstractmethod
    def build_encoder(self, x, c=None, reuse: bool = False) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        raise NotImplementedError("_get_encoder not implemented")

    @abstractmethod
    def build_decoder(self, x, c=None, reuse: bool = False) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        raise NotImplementedError("_get_decoder not implemented")

    def train(self, get_data: Callable, epochs: int, lr: float,
              model_path: str, log_epoch: Callable = None) -> float:
        """ Train a VAE

        :param get_data: Get data (input, condition). Called every epoch
        :param epochs: training epochs
        :param lr: initial learning rate
        :param model_path: save path
        :param get_batch: A call-back which generates a batch. If not
        specified, use `mini_batch_gtor` by default.
        :param log_epoch: A call-back function to log at each epoch
        :return:
        """
        self.sess.run(tf1.global_variables_initializer())
        best_mae = 1.0
        for epoch in range(epochs):
            x, c = get_data()
            current_lr = cosine_decay(epoch, lr, epochs, 2e-6, 1, epochs / 10)
            losses_list = []
            for _x, _c, _last in mini_batch_gtor(x, c, self.batch_size,
                                                 verbose=True):
                losses = self.__step(_x, current_lr, _c, _last)
                losses_list.append(list(losses))

            # book-keeping
            if log_epoch:
                mean_losses = np.array(losses_list).mean(axis=0)
                log_epoch(epoch, epochs, current_lr, mean_losses)

            # after training 50 %, evaluate the model and save the best only
            if epoch > 0.5 * epochs:
                mae = self.evaluate(x, c)
                if mae < best_mae:
                    best_mae = mae
                    self.save(model_path)
        return best_mae

    def save(self, path: str):
        self.saver.save(self.sess, path)

    def restore(self, path: str):
        """ Load pre-trained weights from a saved checkpoint

        :param path: Path of the saved model directory
        :return:
        """
        self.saver.restore(self.sess, path)

    def __step(self, input_batch: np.ndarray, lr: float,
               c_batch: np.ndarray = None, summarize=False) \
            -> Tuple[tf1.Tensor, tf1.Tensor]:
        # assert self.is_conditional != c_batch is None
        # print('c_batch.shape', c_batch.shape)
        loss_kl, loss_gen, summary, _ = self.sess.run(
            [self.kl_loss, self.gen_loss, self.summary, self.opt],
            feed_dict=self._get_feed_dict(input_batch, c_batch, lr))

        if summarize:
            global_step = self.global_step.eval(self.sess)
            self.writer.add_summary(summary, global_step)

        return loss_kl, loss_gen

    @abstractmethod
    def generate(self, n_samples: int, generate_label: Callable = None):
        raise NotImplementedError("generate not implemented")

    def evaluate(self, x: np.array, c: np.array = None) -> np.ndarray:
        mae_list = self._run_batch(self.x, [self.mae], x, c)
        return np.mean(mae_list)

    def encode(self, x: np.array, c: np.array = None) -> np.ndarray:
        return self._run_batch(self.x, [self.z], x, c)[0]

    def decode(self, z: np.array, c: np.array = None) -> np.ndarray:
        return self._run_batch(self.z, [self.x_hat], z, c)[0]

    def extract_posterior(self, x: np.array, c: np.array = None) \
            -> Tuple[np.array, np.array]:
        return self._run_batch(self.x, [self.mu, self.sd], x, c)[:2]

    def update_latent_plot(self, plot) -> None:
        if self.latent_dim != 2:
            return
        global_step = self.global_step.eval(self.sess)
        self.writer.add_summary(self.latent_summary.eval(
            feed_dict={self.latent_plot: plot.eval()}), global_step)

    def reconstruct(self, x, c=None):
        assert len(x) > 0, "len(x): {}".format(len(x))
        return self.decode(self.encode(x, c), c)

    def _run_batch(self, input_tensor: tf.Tensor, run_tensors: List[tf.Tensor],
                   inputs: np.array, c: np.array = None) -> Tuple[np.array]:
        assert not self.is_conditional or c is not None

        out_list = []
        for _ in range(len(run_tensors)):
            out_list.append(list())

        for _xs, _cs, _last in mini_batch_gtor(inputs, c, self.batch_size):
            d = {
                input_tensor: _xs,
                self.is_training: False
            }
            if self.is_conditional:
                d[self.c] = _cs

            outs = self.sess.run(run_tensors, feed_dict=d)
            for i, out in enumerate(outs):
                out_list[i].append(out)

        return_vals = []
        for out in out_list:
            if type(out[0]) == np.ndarray:
                out = np.concatenate(out[:len(inputs)], axis=0)
            return_vals.append(out[:len(inputs)])
        return tuple(return_vals)

    def _get_feed_dict(self, input_batch, c_batch, lr):
        d = {
            self.x: input_batch,
            self.lr: lr,
            self.is_training: True
        }
        if self.is_conditional:
            d[self.c] = c_batch
        return d

    def __build_encoder(self):
        c = self.c if self.is_conditional else None
        encoder_output = self.build_encoder(self.x, c=c)
        self.z, self.mu, self.sd, self.log_sd = encoder_output

    def __build_decoder(self):
        c = self.c if self.is_conditional else None
        decoder_output = self.build_decoder(self.z, c=c)
        self.x_hat, self.gamma, self.log_gamma = decoder_output

    def __build_acai(self):
        self.alpha = tf1.random_uniform([self.batch_size, 1], 0, 0.5)
        encode_mix = self.alpha * self.z + (1 - self.alpha) * self.z[::-1]
        decode_mix = self._get_decode_mix(encode_mix)
        # VAE latent space interpolation regularization term
        self.disc_alpha = self._get_discriminator(decode_mix)

    def __build_loss(self):
        with variable_scope('{}/loss'.format(self.name)):
            self.kl_loss = tf.reduce_sum(
                tf.square(self.mu) + tf.square(self.sd) - 2 * self.log_sd - 1)\
                           / 2.0 / float(self.batch_size)

            self.gen_loss = tf.reduce_sum(
                tf.square((self.x - self.x_hat) / self.gamma) / 2.0
                + self.log_gamma + HALF_LN_TWO_PI) / float(self.batch_size)

            self.mae = tf.reduce_sum(tf.abs(self.x - self.x_hat)) / float(
                self.batch_size) / np.prod(self.x[0].shape)

            self.loss = self.beta * self.kl_loss + self.gen_loss

            if self.acai > 0.0:
                self.disc_loss = tf.reduce_sum(
                    tf.abs(self.disc_alpha - self.alpha)) / float(
                    self.batch_size)
                self.ae_disc_loss = tf.reduce_sum(self.disc_alpha) / float(
                    self.batch_size)
                self.loss = self.loss + self.acai * self.ae_disc_loss

    def __build_optimizer(self):
        adam = tf1.train.AdamOptimizer(self.lr)
        var_vars = [var for var in tf1.global_variables() if
                    self.name in var.name]
        _grads, _vars = zip(*adam.compute_gradients(self.loss, var_vars))
        _grads, _ = tf.clip_by_global_norm(_grads, 1.0)
        opt = adam.apply_gradients(zip(_grads, _vars), self.global_step)

        if self.acai > 0.0:
            adam2 = tf1.train.AdamOptimizer(self.lr / 10.0)
            disc_vars = [v for v in tf1.global_variables() if
                         'discriminator' in v.name]
            _grads, _vars = zip(
                *adam2.compute_gradients(self.disc_loss, disc_vars))
            _grads, _ = tf.clip_by_global_norm(_grads, 5.0)
            opt_acai = adam.apply_gradients(zip(_grads, _vars),
                                            self.global_step)
            opt = tf.group(opt, opt_acai)

        self.opt = opt

    def _get_discriminator(self, x):
        with tf1.variable_scope('{}/discriminator'.format(self.name),
                                reuse=False):
            # alpha = tf1.random_uniform([tf.shape(self.x)[0], 1, 1, 1], 0, 1)
            # alpha = 0.5 - tf.abs(alpha - 0.5)  # Make interval [0, 0.5]
            y = conv2d(x, 64, 4, 4, 2, 2, name='conv1', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn1')
            y = tf.nn.relu(y)
            y = conv2d(y, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = tf.nn.relu(y)
            y = tf.reshape(y, [self.batch_size, -1])

            # append condition after feature extraction
            if self.is_conditional:
                y = tf.concat([y, self.c], -1)
            y = linear(y, 128, scope="fc1", use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = tf.nn.relu(y)
            # \alhpa \in [0, 0.5]
            y = linear(y, 1, scope="fc2", use_sn=True)
            y = tf.nn.sigmoid(y) / 2.0
            return y

    def _get_summaries(self):
        with tf.name_scope(self.name):
            summary = [
                scalar('kl_loss', self.kl_loss),
                scalar('loss', self.loss),
                scalar('gamma', self.gamma),
                scalar('mae', self.mae),
                scalar('lr', self.lr),
                histogram('mu', self.mu),
                histogram('sd', self.sd),
            ]
            if self.acai > 0.0:
                assert self.disc_loss is not None and\
                       self.ae_disc_loss is not None
                summary += [
                    histogram('disc_alpha', self.disc_alpha),
                    scalar('ae_disc_loss', self.ae_disc_loss),
                    scalar('disc_loss', self.disc_loss),
                ]
            self.latent_summary = tf1.summary.image('latent', self.latent_plot)
            return summary

    def _get_decode_mix(self, encode_mix: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("_get_decode_mix is not implemented")


class InnerVaeModel(VaeModel):

    def __init__(self, name: str, sess: tf1.Session, exp_dir: str,
                 x: tf.Tensor, batch_size: int, latent_dim: int,
                 n_conditions: int = 0, acai: float = 0.0, beta: float = 1.0,
                 depth: int = 3, fc_dim: int = 1024,
                 outer_vaes: List[VaeModel] = None):
        """ A 2nd-stage VAE model """
        self.input_dim = x.shape[1]     # outer VAE latent dim
        self.depth = depth
        self.fc_dim = fc_dim
        self.outer_vaes = outer_vaes if outer_vaes else []
        super(InnerVaeModel, self).__init__(name, sess, exp_dir, x, batch_size,
                                            latent_dim, n_conditions, acai,
                                            beta)

    def generate(self, n_samples: int, generate_label: Callable = None):

        # conditional -> generate_label
        assert (not self.is_conditional) or (generate_label is not None)

        num_iter = math.ceil(float(n_samples) / float(self.batch_size))
        x_hats, y_hats = [], []

        for i in range(num_iter):
            # u ~ N(0, I)
            u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])

            d = {
                self.z: u,
                self.is_training: False
            }
            if self.is_conditional:
                _c = generate_label(self.batch_size)
                d[self.c] = _c
                y_hats.append(_c)

            z, gamma_z = self.sess.run([self.x_hat, self.gamma], feed_dict=d)
            # z ~ N(f_2(u), \gamma_z I)
            z = z + gamma_z * np.random.normal(0, 1, [self.batch_size,
                                                      self.input_dim])
            x_hats.append(z)

        x_hats = np.concatenate(x_hats, 0)[:n_samples]
        y_hats = np.concatenate(y_hats, 0)[:n_samples] \
            if self.is_conditional else None

        for vae in self.outer_vaes:
            x_hats = vae.decode(x_hats, y_hats)

        return x_hats, y_hats

    def build_encoder(self, z, c=None, reuse=False):
        assert self.is_conditional == (c is not None)
        with tf1.variable_scope('{}/encoder'.format(self.name),
                                reuse=reuse):
            t = z
            if self.is_conditional:
                t = tf.concat([t, c], -1)
            t = dense_batch_norm_relu(t, self.fc_dim,
                                      depth=self.depth, encoder=True)
            t = tf.concat([z, t], -1)

            mu = tf1.layers.dense(t, self.latent_dim, name='mu_u')
            log_sd = tf1.layers.dense(t, self.latent_dim, name='logsd_u')
            sd = tf.exp(log_sd)

            u = mu + sd * tf.random.normal(
                [self.batch_size, self.latent_dim])
            return u, mu, sd, log_sd

    def build_decoder(self, u, c=None, reuse=False):
        assert self.is_conditional == (c is not None)
        with tf1.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            t = u
            if self.is_conditional:
                t = tf.concat([t, c], -1)
            t = dense_batch_norm_relu(t, self.fc_dim, depth=self.depth,
                                      encoder=False)
            t = tf.concat([u, t], -1)

            z_hat = tf1.layers.dense(t, self.input_dim, name='z_hat')
            log_gamma = tf1.get_variable('log_gamma', [], tf.float32,
                                         tf.zeros_initializer())
            gamma = tf.exp(log_gamma)
            return z_hat, gamma, log_gamma

    def _get_decode_mix(self, encode_mix: tf.Tensor) -> tf.Tensor:
        # TODO (3/3) what to do with c? Test.
        c = self.c if self.is_conditional else None
        decoded = self.build_decoder(encode_mix, c=c, reuse=True)[0]
        for vae in self.outer_vaes:
            decoded = vae.build_decoder(decoded, c=c, reuse=True)[0]
        return decoded

    def _get_feed_dict(self, input_batch, c_batch, lr):
        d = {
            self.x: input_batch,
            self.lr: lr,
            self.is_training: True
        }
        if self.is_conditional:
            d[self.c] = c_batch
        for vae in self.outer_vaes:
            d[vae.is_training] = False
        return d


class OuterVaeModel(VaeModel, ABC):
    """ A 1st-stage VAE model """

    def generate(self, n_samples, generate_label: Callable = None):
        # conditional -> generate_label
        assert not self.is_conditional or (generate_label is not None)

        num_iter = math.ceil(float(n_samples) / float(self.batch_size))
        gen_samples, gen_ys = [], []
        for i in range(num_iter):
            z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            # x = f_1(z)
            d = {
                self.z: z,
                self.is_training: False
            }
            if self.is_conditional:
                _c = generate_label(self.batch_size)
                d[self.c] = _c
                gen_ys.append(_c)
            x = self.sess.run(self.x_hat, feed_dict=d)
            gen_samples.append(x)

        gen_samples = np.concatenate(gen_samples, 0)[:n_samples]

        if self.is_conditional:
            gen_ys = np.concatenate(gen_ys, 0)[:n_samples]
        else:
            gen_ys = None

        return gen_samples, gen_ys

    def _get_summaries(self):
        with tf.name_scope(self.name):
            summaries = [
                tf1.summary.image('input', self.x, max_outputs=6),
                tf1.summary.image('recon', self.x_hat, max_outputs=6),
            ]
        summaries = super()._get_summaries() + summaries
        return summaries

    def _get_decode_mix(self, encode_mix: tf.Tensor) -> tf.Tensor:
        c = self.c if self.is_conditional else None
        decoded = self.build_decoder(encode_mix, c=c, reuse=True)[0]
        return decoded


class InfoGan(OuterVaeModel):

    def build_encoder(self, x, c=None, reuse=False):
        with tf1.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            y = conv2d(x, 64, 4, 4, 2, 2, name='conv1', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn1')
            y = lrelu(y)
            y = conv2d(y, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = lrelu(y)
            y = tf.reshape(y, [x.get_shape().as_list()[0], -1])

            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            y = linear(y, 1024, scope="fc3", use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = lrelu(y)

            gaussian_params = linear(y, 2 * self.latent_dim, scope="en4",
                                     use_sn=True)
            mu = gaussian_params[:, :self.latent_dim]
            sd = 1e-6 + tf.nn.softplus(gaussian_params[:, self.latent_dim:])
            log_sd = tf1.log(sd)
            z = mu + tf.random.normal(
                [self.batch_size, self.latent_dim]) * sd

            return z, mu, sd, log_sd

    def build_decoder(self, z, c=None, reuse=False):
        side_len = self.x.get_shape().as_list()[1]
        quarter_len, half_len = side_len // 4, side_len // 2
        data_depth = self.x.get_shape().as_list()[-1]

        with tf1.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            y = z
            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            # dense_batch_norm_relu(y, 1024)
            y = linear(y, 1024, scope='fc1')
            y = batch_norm(y, is_training=self.is_training, scope='bn1')
            y = tf.nn.relu(y)

            y = linear(y, 128 * (quarter_len ** 2), scope='fc2')
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = tf.nn.relu(y)

            y = tf.reshape(y, [self.batch_size, quarter_len, quarter_len, 128])
            y = deconv2d(y, [self.batch_size, half_len, half_len, 64],
                         4, 4, 2, 2, name='conv3')
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = tf.nn.relu(y)

            y = deconv2d(y, [self.batch_size, side_len, side_len, data_depth],
                         4, 4, 2, 2, name='conv4')
            x_hat = tf.nn.sigmoid(y)

            log_gamma = tf1.get_variable('log_gamma', [], tf.float32,
                                         tf.zeros_initializer())
            gamma = tf.exp(log_gamma)

            return x_hat, gamma, log_gamma


class InfoGan2(OuterVaeModel):

    def build_encoder(self, x, c=None, reuse=False):
        with tf1.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            y = conv2d(x, 64, 4, 4, 2, 2, name='conv1', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn1')
            y = lrelu(y)
            y = conv2d(y, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = lrelu(y)
            y = tf.reshape(y, [x.get_shape().as_list()[0], -1])

            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            y = linear(y, 512, scope="fc3", use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = lrelu(y)

            gaussian_params = linear(y, 2 * self.latent_dim, scope="en4",
                                     use_sn=True)
            mu = gaussian_params[:, :self.latent_dim]
            sd = 1e-6 + tf.nn.softplus(gaussian_params[:, self.latent_dim:])
            log_sd = tf1.log(sd)
            z = mu + tf.random.normal(
                [self.batch_size, self.latent_dim]) * sd

            return z, mu, sd, log_sd

    def build_decoder(self, z, c=None, reuse=False):
        side_len = self.x.get_shape().as_list()[1]
        quarter_len, half_len = side_len // 4, side_len // 2
        data_depth = self.x.get_shape().as_list()[-1]

        with tf1.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            y = z
            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            # dense_batch_norm_relu(y, 1024)
            y = linear(y, 1024, scope='fc1')
            y = batch_norm(y, is_training=self.is_training, scope='bn1')
            y = tf.nn.relu(y)

            y = linear(y, 128 * (quarter_len ** 2), scope='fc2')
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = tf.nn.relu(y)

            y = tf.reshape(y, [self.batch_size, quarter_len, quarter_len, 128])
            y = deconv2d(y, [self.batch_size, half_len, half_len, 64],
                         4, 4, 2, 2, name='conv3')
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = tf.nn.relu(y)

            y = deconv2d(y, [self.batch_size, side_len, side_len, data_depth],
                         4, 4, 2, 2, name='conv4')
            x_hat = tf.nn.sigmoid(y)

            log_gamma = tf1.get_variable('log_gamma', [], tf.float32,
                                         tf.zeros_initializer())
            gamma = tf.exp(log_gamma)

            return x_hat, gamma, log_gamma


class ResNet(OuterVaeModel):

    def __init__(self, name: str, sess: tf1.Session, exp_dir: str,
                 x: tf.Tensor, batch_size: int, latent_dim: int,
                 n_conditions: int = 0, acai: float = 0.0, beta: float = 1.0,
                 num_scale: int = 4, block_per_scale: int = 1,
                 depth_per_block: int = 2, kernel_size: int = 3,
                 base_dim: int = 16, fc_dim: int = 512):

        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        super().__init__(name, sess, exp_dir, x, batch_size, latent_dim,
                         n_conditions, acai, beta)

    def build_encoder(self, x, c=None, reuse=False):
        with tf1.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            dim = self.base_dim
            y = tf1.layers.conv2d(self.x, dim, self.kernel_size, 1, 'same',
                                  name='conv0')
            for i in range(self.num_scale):
                # originally:
                # self.block_per_scale, self.depth_per_block,
                # encoding should be easy.
                y = scale_block(y, dim, self.is_training, 'scale' + str(i),
                                1, 1, self.kernel_size)

                if i != self.num_scale - 1:
                    dim *= 2
                    y = down_sample(y, dim, self.kernel_size,
                                    'downsample' + str(i))
            y = tf.reduce_mean(y, [1, 2])
            if self.is_conditional:
                y = tf.concat([y, c], -1)

            y = scale_fc_block(y, self.fc_dim, self.is_training, 'fc', 1,
                               self.depth_per_block)
            mu = tf1.layers.dense(y, self.latent_dim)
            log_sd = tf1.layers.dense(y, self.latent_dim)
            sd = tf.exp(log_sd)
            z = mu + tf.random.normal([self.batch_size, self.latent_dim]) * sd

            return z, mu, sd, log_sd

    def build_decoder(self, z, c=None, reuse=False):
        desired_scale = self.x.get_shape().as_list()[1]
        scales, dims = [], []
        current_scale, current_dim = 2, self.base_dim
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(current_dim * 2, 1024)
        assert(scales[-1] == desired_scale)
        dims = list(reversed(dims))

        with tf1.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            y = z
            if self.is_conditional:
                y = tf.concat([y, c], -1)  # condition
            data_depth = self.x.get_shape().as_list()[-1]

            fc_dim = 2 * 2 * dims[0]
            y = tf1.layers.dense(y, fc_dim, name='fc0')
            y = tf.reshape(y, [-1, 2, 2, dims[0]])

            for i in range(len(scales) - 1):
                y = up_sample(y, dims[i + 1], self.kernel_size, 'up' + str(i))
                y = scale_block(y, dims[i + 1], self.is_training,
                                'scale' + str(i), self.block_per_scale,
                                self.depth_per_block, self.kernel_size)

            y = tf1.layers.conv2d(y, data_depth, self.kernel_size, 1, 'same')
            x_hat = tf.nn.sigmoid(y)

            log_gamma = tf1.get_variable('log_gamma', [], tf.float32,
                                         tf.zeros_initializer())
            gamma = tf.exp(log_gamma)

            return x_hat, gamma, log_gamma


class TaxiNet(OuterVaeModel):

    def __init__(self, name: str, sess: tf1.Session, exp_dir: str,
                 x: tf.Tensor, batch_size: int, latent_dim: int,
                 acai: float = 0.0, beta: float = 1.0, n_conditions: int = 0,
                 block_per_scale: int = 1, depth_per_block: int = 2,
                 kernel_size: int = 3, fc_dim: int = 512):

        # second_depth=3, second_dim=1024,
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.dims = [32, 64, 128, 192, 256, 320, 384]
        self.dims = [64, 64, 128, 256, 256, 512, 512]
        # self.dims = [64, 64, 128, 256, 384, 512, 1024]
        # self.dims = [64, 64, 128, 256, 384, 512, 640]
        super().__init__(name, sess, exp_dir, x, batch_size, latent_dim,
                         n_conditions, acai, beta)

    def build_encoder(self, x, c=None, reuse=False):
        with tf1.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            y = tf1.layers.conv2d(x, self.dims[0], self.kernel_size, 1, 'same',
                                  name='conv0')
            for i, dim in enumerate(self.dims[1:]):
                if i > 1:
                    y = down_sample(y, dim, self.kernel_size,
                                   'downsample' + str(i))
                y = scale_block(y, dim, self.is_training, 'scale' + str(i),
                                self.block_per_scale, self.depth_per_block,
                                self.kernel_size)
            # y = tf1.layers.conv2d(y, 384, 1, 1, 'same', name='bottleneck3')
            # y = tf1.layers.conv2d(y, 384, (2, 3), 1, 'same', name='conv1d',
            #                   kernel_initializer=he())
            y = tf.reduce_mean(y, [1, 2])

            y = tf.reshape(y, [x.get_shape().as_list()[0], -1])
            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            # y = tf.layers.dropout(y, 0.1, training=self.is_training)
            y = scale_fc_block(y, self.fc_dim, self.is_training, 'fc', 1,
                               self.depth_per_block)

            mu = tf1.layers.dense(y, self.latent_dim)
            log_sd = tf1.layers.dense(y, self.latent_dim)
            sd = tf.exp(log_sd)
            z = self.mu + tf.random.normal(
                [self.batch_size, self.latent_dim]) * sd

            return z, mu, sd, log_sd

    def build_decoder(self, x, c=None, reuse=False):
        with tf1.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            y = x
            if self.is_conditional:
                y = tf.concat([y, c], -1)      # condition
            data_depth = self.x.get_shape().as_list()[-1]

            y = tf1.layers.dense(y, 2 * 3 * self.dims[6], name='fc0')
            y = tf.reshape(y, [-1, 2, 3, self.dims[6]])

            for i, dim in enumerate(reversed(self.dims[1:])):
                y = up_sample(y, dim, self.kernel_size, 'up' + str(i))
                y = scale_block(y, dim, self.is_training,
                                'scale' + str(i), self.block_per_scale,
                                self.depth_per_block, self.kernel_size)

            y = tf1.layers.conv2d(y, data_depth, self.kernel_size, 1, 'same')
            x_hat = tf.nn.sigmoid(y)

            log_gamma = tf1.get_variable('log_gamma', [], tf.float32,
                                          tf.zeros_initializer())
            gamma = tf.exp(log_gamma)

            return x_hat, gamma, log_gamma


class VaeWrapper(VaeInterface):
    """ Wraps the outer and inner VAEs to provide a simpler interface. """

    def __init__(self, model1: OuterVaeModel, model2: InnerVaeModel):
        self.latent_weights = None
        self.outer = model1
        self.inner = model2

    def generate(self, n_samples, generate_label: Callable = None):
        return self.inner.generate(n_samples, generate_label)

    def encode(self, x: np.array, c: np.array = None,
               stage: int = 2) -> np.ndarray:
        if stage == 1:
            return self.outer.encode(x, c)
        elif stage == 2:
            return self.inner.encode(self.outer.encode(x, c), c)
        else:
            assert False, "stage is either 1 or 2."

    def decode(self, z: np.array, c: np.array = None,
               stage: int = 2) -> np.ndarray:
        if stage == 1:
            return self.outer.decode(z, c)
        elif stage == 2:
            return self.outer.decode(self.inner.decode(z, c), c)
        else:
            assert False, "stage is either 1 or 2."

    def reconstruct(self, x: np.array, c: np.array = None,
                    stage: int = 2) -> np.ndarray:
        return self.decode(self.encode(x, c, stage), c, stage)

    def get_z_distance(self, x: np.array, c: np.array = None) -> np.ndarray:
        """ Get distance between z (stage 1 encoding) and z_hat (1st-stage
        encoding reconstructed from 2nd-stage encoding). """
        zs, _ = self.outer.extract_posterior(c)
        us, _ = self.extract_posterior(x, c)
        z_hats = self.inner.decode(us, c)
        return np.linalg.norm(zs - z_hats, axis=1)

    def get_x_recon_error(self, x: np.array, c: np.array = None) -> np.array:
        """ Get distance between z (stage 1 encoding) and z_hat (1st-stage
        encoding reconstructed from 2nd-stage encoding). """
        z = self.outer.encode(x, c)
        z_hat = self.inner.reconstruct(z, c)
        x_hat = self.outer.decode(z_hat, c)
        assert len(x.shape) == 4
        return np.abs(x - x_hat).mean(axis=1).mean(axis=1).mean(axis=1)

    @property
    def is_conditional(self):
        c1, c2 = self.outer.is_conditional, self.inner.is_conditional
        assert c1 == c2, "Both inner and outer VAE should be (un)conditional."
        return c1

    @property
    def n_conditions(self):
        """
        :return: The number of conditions for conditional VAE. Returns 0 if
        unconditional.
        """
        return self.inner.n_conditions

    @property
    def latent_dim(self):
        return self.inner.latent_dim

    def extract_posterior(self, x, c=None, stage: int = 1):
        mu, sd = self.outer.extract_posterior(c)
        if stage == 1:
            return mu, sd
        elif stage == 2:
            return self.inner.extract_posterior(c)
        else:
            assert False, "stage is either 1 or 2"

    def prioritize_latent(self, xs: np.ndarray, ys: np.ndarray, cnt: int = None,
                          exp_dir: str = None) -> np.ndarray:
        xs, ys = xs[:cnt], ys[:cnt]
        fc0 = [v for v in tf1.global_variables()
               if v.name == 'inner/decoder/fc0/kernel:0'][0]
        zeros = np.array([0] * fc0.shape[1])

        @jit(nopython=True)
        def calc_mae(_xs: np.ndarray, _x_hats: np.ndarray):
            return np.mean(np.abs(_xs - _x_hats))

        def get_recon_mae(_xs, _ys):
            _cs = one_hot(_ys, np.max(ys) + 1)\
                if self.outer.is_conditional else None
            x_hats = self.reconstruct(_xs, _cs)
            return calc_mae(_xs, x_hats)

        weight_per_dim_by_cond = []
        n_cond = self.n_conditions if self.is_conditional else 1
        weights = dict()
        for _y in range(n_cond):
            if self.is_conditional:
                _xs = xs[np.where(ys == _y)]
                _ys = np.array([_y] * len(_xs))
            else:
                _xs, _ys = xs, None

            avg_mae = get_recon_mae(_xs, _ys)
            rand_mae = calc_mae(_xs, np.random.rand(*_xs.shape))
            expected_error = (rand_mae - avg_mae) / self.latent_dim
            print('avg_mae:', avg_mae, 'rand_mae:', rand_mae, 'ee:',
                  expected_error)

            weight_per_dim = []
            for j in range(self.latent_dim):
                # save the original weights
                if j not in weights:
                    weights[j] = self.sess.run(fc0[j])
                # silence each latent variable
                self.sess.run(tf1.assign(fc0[j], zeros))
                # calculate its unique impact on overall reconstruction
                relative_weight = get_recon_mae(_xs, _ys) / expected_error
                weight_per_dim.append(relative_weight)
                # restore the weights
                self.sess.run(tf1.assign(fc0[j], weights[j]))
            weight_per_dim_by_cond.append(weight_per_dim)
        self.latent_weights = np.array(weight_per_dim_by_cond)

        if exp_dir is not None:
            np.save(os.path.join(exp_dir, 'latent_weights.npy'),
                    self.latent_weights)
        return self.latent_weights

    def __str__(self):
        return "VaeWrapper(dim: {}, cond: {}, #cond: {})".format(
            self.latent_dim, self.is_conditional, self.n_conditions)
