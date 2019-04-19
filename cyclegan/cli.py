import click
import os
import yaml

import torch

from cyclegan.main import Runner


@click.group()
def cli():
    """CLI for cyclegan"""


@cli.command()
@click.option('-c', '--config-filename', default=None, type=str,
              help='config file path (default: None)')
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def train(config_filename, resume, device):
    if config_filename:
        # load config file
        with open(config_filename) as fh:
            config = yaml.safe_load(fh)

    elif resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with
        # changed configurations.
        config = torch.load(resume)['config']

    else:
        raise AssertionError('Configuration file need to be specified. '
                             'Add "-c experiments/config.yaml", for example.')

    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    Runner().train(config, resume)


@cli.command()
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def test(resume, device):
    if resume:
        config = torch.load(resume)['config']
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    Runner().test(config, resume)
