import pytest
import os
from vae_testgen import testgen
import utility

root_dir = os.environ['PYTHONPATH'].split(os.pathsep)[0]
dataset = 'mnist'
exp_name = '32d-32d-cond-default'
exp_dir = os.path.join(root_dir, 'exps', dataset, exp_name)
lec_path = os.path.join(root_dir, 'models', dataset, 'test.h5')
n_test = 100


class TestTestGen(object):

    def test_testgen_random(self):
        """ Test if the trained VAE can be loaded and evaluated """
        print("path: ", os.environ['PYTHONPATH'])
        args = [exp_dir, lec_path, str(n_test), '--root-dir', root_dir,
                '--name', 'pytest', '--limit-gpu', '0.1']
        parser = testgen.build_argument_parser()
        parser.print_help()
        args = utility.parse_and_process_args(parser, args)
        testgen.main_random(args)
