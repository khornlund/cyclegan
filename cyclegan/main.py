from tqdm import tqdm

import torch

import cyclegan.data_loader.data_loaders as module_data
import cyclegan.model.loss as module_loss
import cyclegan.model.optimizer as module_optimizer
import cyclegan.model.scheduler as module_scheduler
import cyclegan.model.model as module_arch
from cyclegan.trainer import CycleGanTrainer
from cyclegan.utils import setup_logger, setup_logging


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Runner:

    def train(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, config['training']['verbose'])

        logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config)

        logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        logger.debug('Getting loss and metric function handles')
        loss = getattr(module_loss, config['criterion'])()

        logger.debug('Building optimizer and lr scheduler')
        optimizer    = getattr(module_optimizer, config['optimizer']['type'])(model, config)
        lr_scheduler = getattr(module_scheduler, config['lr_scheduler'])(optimizer, config)

        logger.debug('Initialising trainer')
        trainer = CycleGanTrainer(model, loss, optimizer, resume,
                                  config, data_loader, lr_scheduler)

        trainer.train()
        logger.debug('Finished!')

