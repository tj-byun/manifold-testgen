import pytest
import os
import sys
import shutil
import two_stage_vae.train as train
import two_stage_vae.util as vae_util
import tensorflow.compat.v1 as tf1

import utility

__root_dir = os.environ['PYTHONPATH'].split(os.pathsep)[0]
dataset = 'mnist'
exp_dir = os.path.join(__root_dir, 'exps/{}/test'.format(dataset))

# TODO (200617)
#  - make it work on other datasets as well.
#  - measure coverage


class TestTrain(object):

    def test_eval(self, test_train):
        """ Test if the trained VAE can be loaded and evaluated """
        args = [exp_dir, '--root-dir', __root_dir, '--eval', '--fid-cnt', '1024']
        parser = train.build_arg_parser()
        args = utility.parse_and_process_args(parser, args)
        train.main(args)

    # @pytest.mark.skip("slow")
    @pytest.fixture
    def test_train(self):
        """ Train a test vae for a small number of epochs, for both 1st stage
        and 2nd stage """
        args = [exp_dir, '--root-dir', __root_dir, '--train1',
                '--epochs1', '5', '--train2', '--epochs2', '5']
        parser = train.build_arg_parser()
        args = utility.parse_and_process_args(parser, args)
        train.main(args)
        assert os.path.exists(exp_dir) and os.path.isdir(exp_dir)
        # print("Tearing down test-trained VAE")
        # shutil.rmtree(exp_dir)
        # assert not os.path.exists(exp_dir)
