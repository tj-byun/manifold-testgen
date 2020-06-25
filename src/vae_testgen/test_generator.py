import logging
import os
from typing import Callable, List, Tuple
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyswarms.backend.topology import Ring
from pyswarms.single import GeneralOptimizerPSO
from scipy.stats import norm
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter

import tensorflow as tf
from tensorflow.compat import v1 as tf1
from tensorflow import keras as keras
from tensorflow.keras import backend as K

from two_stage_vae.dataset import load_dataset
from two_stage_vae.latent_lec import LatentSpaceLEC
from two_stage_vae.network.vae import VaeWrapper
from vae_testgen.utils import normal_pdf, soft_max, plot_images, plot_digits
from utility import one_hot

logger = logging.getLogger('vae_testgen.testgen')
dropout_repeat = 10


class LECUnderTest(object):
    """ Takes care of the LEC (model) under test """

    def __init__(self, dataset: str, model_path: str):
        self.dataset = dataset
        self.model = keras.models.load_model(model_path,
                                             custom_objects={'tf': tf})
        # Keras function. (input, is_learning_phase) -> output
        self.dropout_sample = K.function([self.model.layers[0].input,
                                          K.learning_phase()],
                                         [self.model.layers[-1].output])

    def predict(self, xs):
        return self.model.predict(xs)

    def measure_uncertainty(self, xs, is_regression: bool = False,
                            repeat: int = 10):
        logger.debug("Measuring uncertainty with dropout={}".format(repeat))
        outs = np.array([self.dropout_sample([xs, 1])[0]
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

    def get_class_probability(self, xs, label: int):
        assert label >= 0
        return self.predict(xs)[:, label]

    def get_uncertainty_threshold(self, cut: float = 0.7) -> float:
        """ Compute the uncertainty threshold based on the test dataset. The
        threshold is determined to find `cut` * 100 percentage of the "bugs"
        in the test dataset.

        :param cut: 0.0 < cut <= 1.0, percentage of bugs to be caught by the
                    threshold to determine.
        :return: threshold uncertainty value
        """
        assert 0.0 < cut <= 1.0
        # Get uncertainty (sigma) value for the test dataset
        x_test, _ = load_dataset(self.dataset, 'test', normalize=True)
        sigmas = self.measure_uncertainty(x_test, repeat=dropout_repeat)

        # sort the indices by sigma in a descending order
        sigma_tup_list = [(i, sig) for i, sig in enumerate(sigmas)]
        sigma_tup_list.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [tup[0] for tup in sigma_tup_list]

        # check if each input is fault-finding
        y_pred = np.argmax(self.model.predict(x_test), axis=-1)
        y_test, _ = load_dataset(self.dataset, 'test', label=True)
        y_buggy = ~(y_pred == y_test)[sorted_indices]

        # Get sigma threshold
        total_bug_cnt = np.count_nonzero(y_buggy)
        thres_buggy_cnt = int(total_bug_cnt * cut)
        i, bug_cnt = 0, 0
        for i in range(len(sigmas)):
            bug_cnt += 1 if y_buggy[i] else 0
            if bug_cnt >= thres_buggy_cnt:
                break
        sigma_thres = sigmas[sorted_indices][i]
        print(sigmas[sorted_indices])
        logger.info("Sigma {:.4f} at index {} which covers {:.2f}% "
                    "of {} buggy inputs = {}, out of {}".format(
            sigma_thres, i, cut * 100, total_bug_cnt, bug_cnt, len(sigmas)))
        return sigma_thres


class CostFunctionFactory(object):
    """ Generate cost functions to be used as optimization objectives """

    def __init__(self, latent_lec: LatentSpaceLEC, sess_classifier: tf1.Session,
                 vae: VaeWrapper, sess_vae: tf1.Session,
                 target_model: LECUnderTest):
        """
        :param latent_lec: Latent space LEC (classifier / regressor)
        :param sess_classifier: tf1.Session
        :param vae: TwoStageVaeModel
        :param sess_vae: tf1.Session for the VAE
        """
        self.classifier = latent_lec
        self.sess_classifier = sess_classifier
        self.vae = vae
        self.sess_vae = sess_vae
        self.dropout_repetition = 40
        self.model = target_model

    def get_model_uncertainty_cost_function(self):
        def calculate_cost(_latent_vectors: np.ndarray,
                           return_dict: bool = False):
            costs = dict()
            costs.update(self.__get_model_uncertainty(_latent_vectors,
                                                      repeat=50))
            costs.update(self.__get_model_prediction(_latent_vectors))
            # costs.update(self.__get_uncertainty(_latent_vectors))
            costs.update(self.__get_plausibility(_latent_vectors))
            scales = {
                "model_y": 0,
                "m_sigma": 1,
                "latent_p": 1,
                "class_p": 0,
                "model_class_p": 0,
                "uncertainty": 0,
            }
            return self.__get_total_cost(costs, scales=scales,
                                         return_dict=return_dict)

        return calculate_cost

    def get_conditional_cost(self, condition, plaus_weight: float = 2.,
                             fast: bool = False):
        def calc_cost(_latent_vectors: np.ndarray,
                      return_dict: bool = False):
            # probability integral transform
            _latent_vectors = norm.cdf(_latent_vectors)
            costs = dict()
            costs.update(self.__get_model_uncertainty(
                _latent_vectors, condition, repeat=dropout_repeat))
            if not fast:
                # costs.update(self.__get_model_class_p(_latent_vectors, label))
                costs.update(
                    self.__get_model_prediction(_latent_vectors, condition))
            costs.update(self.__get_plausibility(_latent_vectors))
            scales = {"plaus": plaus_weight, "m_sigma": 1., }
            return self.__get_total_cost(costs, scales=scales,
                                         return_dict=return_dict)

        return calc_cost

    def get_targeted_cost(self, label: int = -1, fast: bool = False):

        def calc_cost(_latent_vectors: np.ndarray,
                      return_dict: bool = False):
            costs = dict()
            costs.update(self.__get_model_uncertainty(_latent_vectors,
                                                      repeat=dropout_repeat))
            if not fast:
                # costs.update(self.__get_model_class_p(_latent_vectors, label))
                costs.update(self.__get_model_prediction(_latent_vectors))
            costs.update(self.__get_uncertainty(_latent_vectors))
            costs.update(self.__get_plausibility(_latent_vectors))
            scales = {
                "plaus": 2.,
                "m_sigma": 0.3,
                "sigma": -1,
                "class_p": 0,
                "model_class_p": 0,
            }
            return self.__get_total_cost(costs, scales=scales,
                                         return_dict=return_dict)

        return calc_cost

    def get_cost_function(self, label1: int = -1, label2: int = -1,
                          maximize_p: bool = False) -> Callable:
        """ Curry the optimization objective function
        (classifier.get_uncertainty and more) with the given tf1.Session.
        Set negative uncertainty as the minimization goal by default.
        (minimization of this term leads to a maximal uncertainty +
        other terms.)

        :param label1: A class label to condition the generated image with
                       (-1 for no conditioning)
        :param label2: Another class label to condition the generated image with
                       (-1 for no conditioning)
        :param maximize_p: Choose ones with maximal
                                     latent-space sampling probability
        :return: A curried function for calculating optimization objective
        """

        def calculate_cost(_latent_vectors: np.ndarray,
                           return_dict: bool = False):
            """ Calculate the objective score

            :param _latent_vectors: [n_inputs x latent_dimension]
            :param return_dict: Return a dictionary of individual loss if True
            :return: cost to minimize, or a dictionary of costs
            """
            costs = dict()
            costs.update(self.__get_class_p(_latent_vectors, label1, label2))
            costs.update(self.__get_uncertainty(_latent_vectors))
            if maximize_p:
                costs.update(self.__get_plausibility(_latent_vectors))
            scales = {
                'sigma': 0.1,
                'class_p': 1.0,
                'latent_p': 0.5,
            }
            return self.__get_total_cost(costs, scales, return_dict=return_dict)

        return calculate_cost

    def __get_uncertainty(self, _latent_vectors: np.ndarray):
        u = self.classifier.get_uncertainty(self.sess_classifier,
                                            _latent_vectors,
                                            repeat=self.dropout_repetition)
        return {'sigma': u}

    def __get_model_prediction(self, _latent_vectors: np.ndarray,
                               condition):
        if type(condition) == int:
            condition = np.array(
                [condition for _ in range(len(_latent_vectors))])
        else:
            assert type(condition) == np.ndarray \
                   and len(_latent_vectors) == len(condition), \
                "Length: {}, {}".format(len(_latent_vectors), len(condition))
        x_hats = self.vae.decode(_latent_vectors, condition)
        ys = self.model.predict(x_hats)
        # "latent_y": np.argmax( self.classifier.predict(self.sess_classifier,
        # latent_vectors), axis=-1),
        return {
            "model_y": np.argmax(ys, axis=-1),
            "max_p": np.max(ys, axis=-1),
        }

    @staticmethod
    def squash(x: float):
        exp = pow(np.e, x)
        return (exp - 1) / (exp + 1)

    def __get_model_uncertainty(self, _latent_vectors: np.ndarray,
                                condition, repeat=10):
        if type(condition) == int:
            condition = np.array(
                [condition for _ in range(len(_latent_vectors))])
        assert hasattr(self, 'model')
        x_hats = self.vae.decode(_latent_vectors, condition)
        sigma = self.model.measure_uncertainty(x_hats, repeat=repeat)
        return {'m_sigma': self.squash(sigma)}

    def __get_class_p(self, _latent_vectors: np.ndarray, label1, label2=-1):
        """ Get predicted class probability for the given label(s). """
        if label1 >= 0 and label2 == -1:
            assert label1 in range(self.classifier.n_class)
        elif label1 >= 0 and label2 >= 0:
            assert label1 in range(self.classifier.n_class) and \
                   label2 in range(self.classifier.n_class)
        else:
            assert False
        costs = dict()
        logits = self.classifier.predict(self.sess_classifier, _latent_vectors)
        if label1 >= 0 and label2 == -1:
            # predicted probability is always within [0., 1.]
            costs["class_p"] = logits[:, label1]
        elif label1 >= 0 and label2 >= 0:
            costs['class_p1'] = logits[:, label1]
            costs['class_p2'] = logits[:, label2]
            costs['class_p_product'] = costs['class_p1'] * costs['class_p2']
        return costs

    @staticmethod
    def __get_plausibility(_latent_vectors: np.array):
        """ Calculate latent probability """
        # p = unit_gauss_pdf(_latent_vectors)
        # p_prod = (1.0 / np.power(np.log2(p / P_STANDARD_NORMAL) - 1,
        # 2)).mean( axis=-1)
        # p_prod = np.mean(2.0 / np.power(np.e, 0.5 * _latent_vectors) - 1.0,
        #                 axis=-1)
        p_prod = 1.0 / np.power(np.e, np.power(_latent_vectors, 2))
        return {'plaus': np.mean(p_prod, axis=-1)}

    def __get_total_cost(self, costs, scales: dict, return_dict: bool = False,
                         product: bool = False):
        # scale the costs and sum
        total_cost = 1.0 if product else 0.0
        for key in costs:
            cost = ((costs[key] + 0.0001) * scales[key]) if key in scales else 0
            if product:
                total_cost *= cost
            else:
                total_cost += cost
        if return_dict:
            costs["total_cost"] = -total_cost
            return costs
        else:
            return -total_cost  # negate for minimization


class Labeler(object):
    """ Predict the label for a list of test inputs in latent vector
    representation u. """

    def __init__(self, _vae: VaeWrapper, _classifier: LatentSpaceLEC,
                 xs: np.ndarray, ys: np.ndarray, sample_rate: float):
        """
        :param _vae: a TwoStageVaeModel
        :param _classifier: a VaeClassifier
        :param xs: Xs from the training dataset
        :param ys: Ys from the training dataset
        """
        self.vae = _vae
        self.classifier = _classifier
        # testgen_config["train_data_sample_rate"]
        self.sample_rate = sample_rate
        self.xs = xs
        self.ys = ys
        indices = list(range(int(float(len(xs)) * self.sample_rate)))
        logging.info("Referencing {}/{} training data".format(
            len(indices), len(xs)))
        random.shuffle(indices)
        self.xs = self.xs[indices]
        self.ys = self.ys[indices]
        assert 0. <= self.sample_rate <= 1.

    def predict(self, vae_sess: tf1.Session, us: np.array) -> List[List[float]]:
        """ Generate label for the given list of latent vectors and return
        class label.

        :param vae_sess: tf1.Session
        :param us: 2D numpy array of latent vectors
        :return: np.array of class label for each latent vector
        """
        logger.info("Creating latent distributions ...")
        mu_zs, sd_zs = self.vae.extract_posterior()
        # zs = self.vae.decode(vae_sess, us, to_z=True)

        logger.info("Assigning labels ...")
        latent_inds = self.__get_significant_lantent_node_indices(vae_sess)
        logit_list = Parallel(n_jobs=-1)(
            delayed(self.__class__.__predict_class_probability)
            (u, mu_zs, sd_zs, self.ys, latent_inds) for u in tqdm(us)
        )
        return logit_list

    @staticmethod
    def __predict_class_probability(z: np.array, mus: np.array,
                                    sigmas: np.array, ref_labels: List,
                                    latent_indices):
        """ Compute the probability of a given synthesized latent
        vector u belonging to each class label

        :param z: A synthesized latent vectors that requires labeling.
        :param mus: $\\mu$s. [reference_data x latent_node]
        :param sigmas: $\\sigma$$s. [reference_data x latent_node]
        :param ref_labels: Labels for the reference data (used to obtain
                           distributions)
        :param latent_indices: Latent indices to consider
        :return: p(z.label == c_i) for each class from 0 to 9.
        """
        assert len(mus) == len(sigmas) == len(ref_labels) > 0

        # p(z.label == c_i) for each class from 0 to 9
        p_label_ci = [[] for _ in range(10)]

        # compute p(z | x_i) per each x_i \in reference_data
        for i in range(len(mus)):
            p_zj_xi = [normal_pdf(mus[i][j], sigmas[i][j], z[j])
                       for j in latent_indices]
            p_z_xi = np.mean(p_zj_xi)
            # p_z_xi = np.exp(np.log(np.power(p_zj_xi, 1.0 / latent_dim)).sum())
            # ,
            # 1.0 / latent_dim)
            # print("p_z_xi", p_z_xi)
            # p_z_xi = np.exp(np.log(p_zj_xi).sum())
            p_label_ci[ref_labels[i]].append(p_z_xi)

        return soft_max([np.array(ps).mean() for ps in p_label_ci])

    def __get_significant_lantent_node_indices(self, sess_vae) -> List[int]:
        u = self.vae.encode(self.xs)
        best_features = SelectKBest(score_func=f_classif, k=self.vae.latent_dim)
        fit = best_features.fit(u, self.ys)
        df_scores = pd.DataFrame({"score": fit.scores_, "pval": fit.pvalues_})
        df_filtered = df_scores.loc[
            df_scores['score'] > df_scores['score'].mean()]
        df_filtered = df_filtered.sort_values('score', ascending=False)
        latent_inds = df_filtered.index.tolist()
        return latent_inds


class VaeFuzzer(object):

    def __init__(self, _vae: VaeWrapper, _model_under_test: LECUnderTest,
                 dataset: str, latent_dim: int, output_dir: str):
        self.plaus_lower_bound = 0.1
        self.distance_lower_bound = 4.0
        self.uncertainty_upper_bound = 999.

        self.vae = _vae
        self.model_under_test = _model_under_test
        self.latent_dim = latent_dim
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.total_cnt = 0  # total number of test generation attempted

        self.xs, self.dim = load_dataset(dataset, 'train', normalize=True)
        self.ys, self.n_classes = load_dataset(dataset, 'train', label=True)

        min_bound = np.array([-1.0] * self.latent_dim)
        max_bound = np.array([1.0] * self.latent_dim)
        self.bounds = (min_bound, max_bound)

    def get_plausibility(self, us):
        p_prod = 1.0 / np.power(np.e, np.power(us, 2))
        return np.mean(p_prod, axis=-1)

    def get_uncertainty(self, xs: np.array):
        sigma = self.model_under_test.measure_uncertainty(xs, repeat=100)
        return np.mean(sigma, axis=-1)

    def generate(self, n_test: int, batch_size: int = 32)\
            -> Tuple[np.array, np.array, np.array]:
        """ Generate fault-revealing test cases

        :param n_test: Number of test cases to generate
        :param batch_size: Size of the batch to process at the same time.
        Higher batch size is more GPU efficient
        :return: (latent codes, synthesized inputs, synthesized labels)
        """
        cnt = Counter()
        synth_inputs, synth_latent_codes, synth_labels = [], [], []
        while cnt['bug'] < n_test:
            # new inputs are generated in batch of size `batch_size`.
            cnt['total'] += batch_size
            # synthesized labels
            ys_hat = np.array([random.randint(0, self.n_classes - 1) for _ in
                              range(batch_size)])
            cs_hat = one_hot(ys_hat, self.n_classes)
            # randomly chosen latent codes
            us_hat = np.array([[random.gauss(0, 1)
                                for _ in range(self.latent_dim)]
                               for _ in range(batch_size)])
            # synthesized inputs
            xs_hat = self.vae.decode(us_hat, cs_hat)
            ys_pred = np.argmax(self.model_under_test.predict(xs_hat), axis=-1)
            for u_hat, x_hat, y_hat, y_pred in zip(us_hat, xs_hat, ys_hat,
                                                   ys_pred):
                # when the predicted output and the expected output
                # don't match, it is considered a fault-finding test case.
                if y_pred != y_hat:
                    is_duplicate = False
                    for u in synth_latent_codes:
                        # if this u is close enough to any of the already
                        # synthesized ones, consider it as a duplicate.
                        if np.linalg.norm(u - u_hat) \
                                < self.distance_lower_bound:
                            is_duplicate = True
                            break
                    # keep track of low plausibility and high uncertainty
                    # cases, just for statistics
                    if self.get_plausibility(u_hat) < self.plaus_lower_bound:
                        cnt['low_plaus'] += 1
                    if self.get_uncertainty(np.expand_dims(x_hat, 0)) > \
                            self.uncertainty_upper_bound:
                        cnt['high_uncertainty'] += 1
                    if is_duplicate:
                        cnt['duplicate'] += 1
                    else:
                        synth_latent_codes.append(u_hat)
                        synth_inputs.append(x_hat)
                        synth_labels.append(y_hat)
                        cnt['bug'] += 1
        assert cnt['bug'] == len(synth_latent_codes) == \
               len(synth_inputs) == len(synth_labels)
        # Trim the excess beyond `n_test`, caused by batch-ed generation
        synth_latent_codes = np.array(synth_latent_codes[:n_test])
        synth_inputs = np.array(synth_inputs[:n_test])
        synth_labels = np.array(synth_labels[:n_test])
        print("xs.shape", synth_inputs.shape)
        return synth_latent_codes, synth_inputs, synth_labels

    def save_to_npy(self, xs, ys, us, dim: Tuple[int, int, int]):
        y_preds = np.argmax(self.model_under_test.predict(xs), axis=-1)
        xs = (xs.reshape(tuple([len(xs)] + list(dim))) * 255.0)
        xs = np.around(xs).astype(int)
        print("xs.shape:", xs.shape)
        np.save(os.path.join(self.output_dir, "x.npy"), xs)
        np.save(os.path.join(self.output_dir, "y.npy"), ys)
        np.save(os.path.join(self.output_dir, "us.npy"), us)
        np.save(os.path.join(self.output_dir, "y_preds.npy"), y_preds)
        try:
            plot_images(xs, dim, os.path.join(self.output_dir, "synth.png"))
        except:
            plot_digits(xs, os.path.join(self.output_dir, "synth.png"))


class VaeTestGenerator:
    """ Given a VAE and a classifier, generate test inputs with labels. """

    def __init__(self, _vae: VaeWrapper, _classifier: LatentSpaceLEC,
                 _model_under_test: LECUnderTest, dataset: str,
                 output_dir: str, pso_options: dict, n_iter: int):
        """ Initialize a test generator that synthesizes new
        high-uncertainty image inputs for a given pair of VAE and a classifier.

        :param _vae: A VAE model
        :param _classifier: A classifier model attached to the latent layer
                            of the VAE
        :param _model_under_test: Model under test
        :param dataset: name of a dataset
        :param output_dir: (str) Output directory path
        :param pso_options: a dictionary containing PSO hyper-parameters,
        which are {c1, c2, w, k, p}.
        :param n_iter: PSO iteration
        """
        self.threshold = 1.0
        self.vae = _vae
        self.classifier = _classifier
        self.model_under_test = _model_under_test
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.total_cnt = 0      # total number of test generation attempted

        self.xs, self.dim = load_dataset(dataset, 'train', normalize=True)
        self.ys, self.n_classes = load_dataset(dataset, 'train', label=True)

        # self.n_particle = testgen_config["optimizer"]["n_particle"]
        self.n_iter = n_iter
        self.options = pso_options
        self.topology = Ring(static=False)
        min_bound = np.array([-1.0] * self.vae.latent_dim)
        max_bound = np.array([1.0] * self.vae.latent_dim)
        self.bounds = (min_bound, max_bound)

    def synthesize(self, sess_vae: tf1.Session, us: np.ndarray,
                   ys: np.ndarray, reshape: bool = False) -> np.array:
        """
        :param sess_vae: tf1.Session for VAE
        :param reshape: True to return x_hats in a displayable 2d-reshaped form
        :param us: latent vectors; set to skip generating anew
        :param ys:
        :return: Synthesized test inputs
        """
        # if DEBUG:
        # TODO: read the parameters of `z` to set good bounds ??
        # get_z_mean = K.function(
        #     [self.vae.encoder.layers[0].input],
        #     [self.vae.encoder.get_layer('z_mean').output])
        # print("z_mean:", get_z_mean([np.array([[.5] * 28 ** 2])]))

        x_hats = self.vae.decode(us, ys)
        if reshape:
            x_hats = x_hats.reshape(tuple([len(us)] + list(self.dim)))
        return x_hats

    def optimize_conditional(self, cost_factory: CostFunctionFactory,
                             n_test: int, reshape: bool = False):
        # TODO (8/3): implement
        us, xs, ys = [], [], []
        self.total_cnt, bug_cnt = 0, 0
        while bug_cnt < n_test:
            self.total_cnt += 1
            y = random.randint(0, self.n_classes - 1)
            cost_function = cost_factory.get_conditional_cost(y)
            z_uni = self.__optimize(cost_function)[1]
            z = norm.ppf(z_uni)
            x = self.vae.decode(np.array([z]), np.array([y]))
            yp = np.argmax(self.model_under_test.predict(x), axis=-1)
            if y != yp:
                bug_cnt += 1
                logging.info(
                    "Buggy! current cnt: {}/{}".format(bug_cnt, self.total_cnt))
                is_duplicate = False
                for mu in us:
                    if np.linalg.norm(z - mu) < self.threshold:
                        logging.info("Duplicate. Skip")
                        is_duplicate = True
                        break
                if not is_duplicate:
                    us.append(z)
                    xs.append(x)
                    ys.append(y)
        us, xs, ys = np.array(us), np.array(xs), np.array(ys)
        np.save(os.path.join(self.output_dir, 'us.npy'), us)
        if reshape:
            xs = xs.reshape(tuple([len(us)] + list(self.dim)))
        return us, xs, ys

    def generate_gaussian(self, cnt: int) -> np.array:
        return np.array([[random.gauss(0, 1) for _ in range(self.latent_dim)]
                         for _ in range(cnt)])

    def filter_by_uncertainty(self, sess_vae: tf1.Session, us: np.array,
                              threshold: float):
        """ Generate tests by sampling from Gaussian and filtering for
        uncertainty.

        :param sess_vae: VAE session
        :param us: np.array of latent vectors
        :param threshold: uncertainty threshold
        """
        xs = self.vae.decode(us)
        sigmas = self.model_under_test.measure_uncertainty(
            xs, repeat=dropout_repeat)
        sigma_tup_list = [(i, sig) for i, sig in enumerate(sigmas)]
        filtered_tup_list = [tup for tup in sigma_tup_list
                             if tup[1] >= threshold]
        filtered_tup_list.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [tup[0] for tup in filtered_tup_list]
        return us[sorted_indices]

    def __optimize(self, cost_function: Callable) -> Tuple[float, np.array]:
        opt = GeneralOptimizerPSO(n_particles=self.vae.latent_dim,
                                  dimensions=self.vae.latent_dim,
                                  options=self.options,
                                  topology=self.topology,
                                  bounds=self.bounds)
        cost, z = opt.optimize(cost_function, iters=self.n_iter, fast=True)
        return cost, z

    def generate_labels(self, sess_vae: tf1.Session, us: np.array) -> List[int]:
        """ Generate labels

        :param sess_vae: tf1.Session for Vae
        :param us: latent vectors
        :return: labels
        """
        sample_rate = 1.0
        logit_list = Labeler(self.vae, self.classifier, self.xs, self.ys,
                             sample_rate).predict(sess_vae, us)
        labels = [np.argmax(logits) for logits in logit_list]
        # for k, logits in enumerate(logit_list):
        #     # print("logits:", logits, "label:", labels[k])
        print("labels:", labels)
        return labels

    def save_to_npy(self, xs, ys, us, dim: Tuple[int, int, int]):
        y_preds = np.argmax(self.model_under_test.predict(xs), axis=-1)
        xs = (xs.reshape(tuple([len(xs)] + list(dim))) * 255.0)
        xs = np.around(xs).astype(int)
        print("xs.shape:", xs.shape)
        np.save(os.path.join(self.output_dir, "x.npy"), xs)
        np.save(os.path.join(self.output_dir, "y.npy"), ys)
        np.save(os.path.join(self.output_dir, "us.npy"), us)
        np.save(os.path.join(self.output_dir, "y_preds.npy"), y_preds)
        plot_images(xs, dim, os.path.join(self.output_dir, "synth.png"))
