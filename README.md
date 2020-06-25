# Manifold-based Test Generation


- Code written by Taejoon Byun, Abhishek Vijayakumar, and Tyler Heim.
- README Last updated: Jun 24 2020
- The `two_stage_vae` module was originally forked/adapted from Bin Dai's Git
 repository: https://github.com/daib13/TwoStageVAE

## Papers

[Manifold-based Test Generation for Image Classifiers](https://arxiv.org/abs/2002.06337)

[Manifold for Machine Learning Assurance](https://arxiv.org/abs/2002.03147)

## Disclaimer

The current version may contain bug and might not work as expected. I will
 test the code more rigorously and update it soon.

## How to Run

1. Download datasets and place them in `data/<dataset_name>` directory at the
 project root.
2. Prepare a model under test, or train one (description below).
3. Train a (conditional two-stage) VAE for the dataset (description below).
4. Run the test generator (description below).

### Train a model (as a test target)

Supported dataset: `mnist`, `svhn`, `cifar10`, `fashion`

```shell script
PYTHONPATH=src python -m lecs.train <dataset> <model_name>
```

Once the training is done, the model will be saved at
`models/<dataset>/<model_name>`. For more training options, run with `--help`.

The following command trains an MNIST model. The trained model can be found at
`models/mnist/test_model.h5`.

```shell script
PYTHONPATH=src python -m lecs.train mnist test_model
```


### Train a VAE

For the full option, run `two_stage_vae` module with `--help` option.

The following command trains a conditional two-stage VAE for MNIST dataset,
and save the result at `exps/mnist/32`. The default latent dimension size is 32.
 
```shell script
PYTHONPATH=src python -m two_stage_vae.train exps/mnist/16d-cond --conditional --train1 --train2
```

To monitor how the training progresses, launch TensorBoard on the output
 directory:

```shell script
tensorboard --logdir=exps/mnist/32d
```

To load and test the FID score of the trained VAE, run:

```shell script
PYTHONPATH=src python -m two_stage_vae.train exps/mnist/32d --eval
```

### Generate Fault-finding Test Cases

You can generate fault-finding test cases for a target model with the
trained VAE as the test case generator.

In the paper, we introduce search-based test generation on the latent space
, along with a random generation, or sampling from Gaussian prior. The search
-based generation is currently not working in this release. We will

```shell script
PYTHONPATH=src python -m vae_testgen.testgen <vae_dir_path> <model_path
> <cnt_tests>
```

For example, the following code generates 100 fault-revealing test cases for the
 MNIST model we just trained, with the VAE model we just trained above.
 
```shell script
PYTHONPATH=src python -m vae_testgen.testgen exps/mnist/32d-cond models/mnist
/32d 100
```
