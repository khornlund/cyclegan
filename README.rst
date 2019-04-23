========
cyclegan
========
PyTorch implementation of paper: `<https://arxiv.org/abs/1703.10593>`_

.. contents:: Table of Contents
   :depth: 2

Requirements
============
* Python >= 3.6
* PyTorch >= 0.4

Features
========
* `.yaml` config file support for more convenient parameter tuning.
* Checkpoint saving and resuming.

Folder Structure
================

::

  cyclegan/
  │
  ├── cyclegan/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │   ├── base_data_loader.py - abstract base class for data loaders
  │    │   ├── base_model.py - abstract base class for models
  │    │   └── base_trainer.py - abstract base class for trainers
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │   └── data_loaders.py
  │    │
  │    ├── model/ - models, losses, optimizers, and schedulers
  │    │   ├── loss.py
  │    │   ├── model.py
  │    │   ├── optimizer.py
  │    │   └── scheduler.py
  │    │
  │    ├── trainer/ - trainers
  │    │   └── trainer.py
  │    │
  │    └── utils/
  │        ├── logger.py - class for train logging
  │        ├── visualization.py - class for tensorboardX visualization support
  │        └── saving.py - manages pathing for saving models + logs
  │
  ├── logging.yaml - logging configuration
  │
  ├── datasets/ - default directory for storing input data
  │
  ├── experiments/ - default directory for storing configuration files
  │
  ├── saved/ - default checkpoints folder
  │
  └── tests/ - default tests folder


Usage
=====

.. code-block:: bash

  $ cd path/to/repo
  $ conda create --name <name> python=3.6
  $ pip install -e .
  $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
  $ cyclegan train -c experiments/config.yaml


Config file format
------------------
Config files are in `.yaml` format:

.. code-block:: HTML

  name: CycleGAN
  n_gpu: 1
  save_dir: saved/

  arch:
    type: CycleGan
    args:
      input_nc: 3
      output_nc: 3
      img_size: 256
      verbose: 2

  data_loader:
    type: BerkelyDataLoader
    args:
      batch_size: 2
      img_size: 256
      data_dir: datasets/
      dataset: summer2winter_yosemite
      num_workers: 2
      shuffle: true
      validation_split: 0.0

  criterion: CycleGanCriterion

  optimizer:
    type: CycleGanOptimizer
    lr: 0.0002
    beta_lower: 0.5
    beta_upper: 0.999

  lr_scheduler: CycleGanScheduler

  training:
    loss_weight:
      identity: 5
      cycle: 10
    early_stop: 10
    epochs: 200
    save_period: 1
    tensorboardX: true
    verbose: 2

  testing:
    verbose: 2


Add addional configurations if you need.

Using config files
------------------
Modify the configurations in `.yaml` config files, then run:

.. code-block:: shell

  python train.py --config experiments/config.yaml

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block:: shell

  python train.py --resume path/to/checkpoint


Using Multiple GPU
------------------
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.

.. code-block:: shell

  python train.py --device 2,3 -c experiments/config.yaml

This is equivalent to

.. code-block:: shell

  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py


Testing
-------
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume`
argument.


TensorboardX Visualization
--------------------------
This template supports `<https://github.com/lanpa/tensorboardX>`_ visualization.

  1. Install

      Follow installation guide in `<https://github.com/lanpa/tensorboardX>`_

  2. Run training

      Set `tensorboardX` option in config file true.

  3. Open tensorboard server

      Type `tensorboard --logdir saved/` at the project root, then server will open at
      `http://localhost:6006`


Acknowledgments
===============

  1. `<https://github.com/aitorzip/PyTorch-CycleGAN>`_
  2. `<https://github.com/khornlund/cookiecutter-pytorch>`_
