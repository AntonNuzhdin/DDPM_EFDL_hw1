Bugs in the original code:

1. diffusion.py
   These bugs were found during the running tests (some bugs were obvious after running them) and reviewing the original code. I read the code of the original paper and checked the formulas with the code.
- in forward: move timestep to device; incorrect formula for x_t (should be sqrt_one_minus_alpha_prod in original VP-SDE process); eps should be sampled from N(0,I) -> torch.randn_like
- in sample: move x_i and z to device
- in get_schedules: assert 0 < beta1 < beta2 < 1.0, betas should be positive numbers

2. unet.py
- in forward: the temb should be expanded with two fictitious dimentions to enable broadcasting. This was found during runing initial tests

Other modifications: 
- add ```test_training``` in ```tests/test_pipeline``` using pytest. The test covers the whole training process.
Testing coverage:
----------- coverage: platform darwin, python 3.12.4-final-0 -----------
Name                                        Stmts   Miss  Cover
-----------------------------------------------------------------
modeling/__init__.py                            0      0    100%
modeling/diffusion.py                          34      0    100%
modeling/training.py                           30      0    100%
modeling/unet.py                               68      0    100%
-----------------------------------------------------------------
TOTAL                                        132      0    100%

- add hydra configuration and reformat the main.py for convenient usage with hydra
- add logger/writer.py with convenient wandb class for logging of the experiments
- integrated the DVC and configure it with hydra 

  


# Week 2 home assignment

This assignment consists of 4 parts: you can earn the full amount of points by completing the first two and either of 
tasks 3 and 4 (or both of them for bonus points).
However, completing tasks 3 or 4 without the first two will not give you any points.

# Problem statement
You are given a small codebase that should train an **unconditional** [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239)
on the CIFAR-10 dataset.
However, this project contains several bugs of different severity, and even some of the tests are written incorrectly.
A correct implementation will achieve *somewhat* decent results after training for 100 epochs (~2 hours on an average GPU),
but you should not expect much in terms of quality.
In this homework, we are going to have a deeper look at the training pipeline, try to fix any errors we find and make 
the code more reliable and reproducible.

# Task 1 (6.5 points)
Implement *correct* tests for the training pipeline.
Specifically, have a look at the current [tests](./tests) folder: it contains several files with tests, 
some of which fail, fail sometimes or are plainly incorrect.
Your task is to identify the bugs and make the test suite pass deterministically: this will involve changes 
both to `modeling` and to `tests`, as some parts of the testing code need to be modified as well.

In your report, please tell us how you found the bugs in all parts of the code.
You can find the original implementation of DDPM that we use in this assignment, but giving it as an explanation for 
your fixes will give you no points.
Obviously, "solving" the assignment by removing all tests or having unreasonably high thresholds will not earn
you a good grade as well.

After that, implement the `test_training` function in `test_pipeline.py` that runs an integration test for the
entire training procedure with different hyperparameters and expects different outcomes.
This test should increase the coverage of the `modeling.training` file (measured by [pytest-cov](https://github.com/pytest-dev/pytest-cov)) to **>80%**.

Importantly, you should ensure that your test code running the actual model can run both on CPU and GPU.
Since training on CPU even for 1 epoch might take too long, you need to implement training on a subset of data.


# Task 2 (1.5 points)
Implement logging of the metrics and artifacts during training with [Weights and Biases](https://wandb.ai/site).
You should log the following values:
* Training loss and the learning rate
* All training hyperparameters (including batch size, number of epochs etc., as well as all model and diffusion hyperparameters)
* Inputs to the model (1 batch is enough) and samples from it after each epoch

However, you should **NOT** log the training code for the model.

Logging the hyperparameters and metrics will likely involve some refactoring of the original codebase.
You can either place the necessary hyperparameters in a config file or simply have them as constants/argparse defaults 
defined somewhere reasonable in the training code.

After finishing this task, train the model for at least 100 epochs with default hyperparameters and attach the link to
your W&B project containing this run to the final report.

# Task 3 (2 points)
Improve the configuration process of this pipeline using the [Hydra](https://hydra.cc/) library.
You should create a config that allows adjusting at least the following attributes:
* Peak learning rate and optimizer momentum
* Optimizer (Adam by default, at least SGD should be supported)
* Training batch size and the number of epochs
* Number of workers in the dataloader
* Existence of random flip augmentations

Demonstrate that your integration works by running at least three *complete* runs (less than 100 epochs is OK) 
with hyperparameters changed via the config file.
From these runs, it should be evident that changing hyperparameters affects the training procedure.
Here, you should log the config using [run.log_artifact](https://docs.wandb.ai/ref/python/run#log_artifact)
and show that this changes the hyperparameters of the run in W&B.

# Task 4 (2 points)
Make the pipeline reproducible using [Data Version Control](https://dvc.org/). 
You should end up with a `dvc.yaml` that represents two stages of your experiment with corresponding inputs and outputs: 
getting the data (yes, you need to refactor that part of the code) and training the model itself.
Also, you should specify the relevant code and configuration as dependencies of the corresponding pipeline stages.
Lastly, after running your code, you should have a `dvc.lock` that stores hashes of all artifacts in your pipeline.
Submit both `dvc.yaml` and `dvc.lock` as parts of your solution.

Importantly, modifying any of the relevant modules or hyperparameters should trigger an invalidation of the
corresponding pipeline stages: that is, `dvc repro` should do nothing if and only if `dvc.lock` is consistent with
hashes of all dependencies in the pipeline.

If you have also done the Hydra configuration assignment, make sure to check out [this guide](https://dvc.org/doc/user-guide/experiment-management/hydra-composition)
on integrating Hydra with DVC experiment management.

# Submission format
When submitting this assignment, you should attach a .zip archive that contains:
- The source code with all your fixes and improvements
- A Markdown/PDF report in the root of the project folder that:
  1. Details the changes you made to the original code (we will run `diff` and see if everything is explained)
  2. Tells how to run the modified code (i.e., which command line arguments you have added and how to use them)
  3. Describes your process of fixing and adding new tests for Task 1 and reports the test coverage
  4. Gives a link to the Weights and Biases project with all necessary logs for tasks 2 and 3
- If you solved Tasks 3 or 4, please ensure that the archived project contains the corresponding configuration/lock files as well.
- An updated `requirements.txt` file, if your solution requires new dependencies such as `wandb`, `hydra-core` or `dvc`.
