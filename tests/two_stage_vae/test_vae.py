import pytest
import os
import sys
import shutil
import two_stage_vae.dataset as data
import two_stage_vae.util as vae_util
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework.ops import disable_eager_execution

import utility

disable_eager_execution()
__root_dir = os.environ['PYTHONPATH'].split(os.pathsep)[0]
dataset = 'mnist'
exp_dir = os.path.join(__root_dir, 'exps/{}/test'.format(dataset))


class TestVae(object):

    @pytest.fixture(scope="module", autouse=True)
    def vae(self):
        sess = tf1.Session()
        return vae_util.load_vae_model(sess, exp_dir, dataset)

    @pytest.fixture(scope="module", autouse=True)
    def xyc(self):
        x, __ = data.load_dataset(dataset, 'test', __root_dir)
        y, __ = data.load_dataset(dataset, 'test', __root_dir, label=True)
        c = utility.one_hot(y)
        return x, y, c

    def test_encode(self, vae, xyc):
        x, y, c = xyc
        z = vae.encode(x, c, stage=1)
        u = vae.encode(x, c, stage=2)

    def test_decode(self, vae, xyc):
        x, y, c = xyc
        z = vae.encode(x, c, stage=1)
        u = vae.encode(x, c, stage=2)
        x_hat1 = vae.decode(z, c, stage=1)
        x_hat2 = vae.decode(u, c, stage=2)
