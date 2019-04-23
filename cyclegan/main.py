from tqdm import tqdm

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

import cyclegan.data_loader.data_loaders as module_data
import cyclegan.model.loss as module_loss
import cyclegan.model.optimizer as module_optimizer
import cyclegan.model.scheduler as module_scheduler
import cyclegan.model.model as module_arch
from cyclegan.trainer import CycleGanTrainer
from cyclegan.utils import setup_logger, setup_logging, test_paths


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

    def test(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, config['testing']['verbose'])

        logger.debug('Getting data_loader instance')
        config['data_loader']['args']['training'] = False
        data_loader = get_instance(module_data, 'data_loader', config)

        logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        logger.debug(f'Loading checkpoint {resume}')
        checkpoint = torch.load(resume)
        if checkpoint['config']['arch'] != config['arch']:
            logger.warning("Warning: Architecture configuration given in config file is "
                           "different from that of checkpoint. This may yield an exception "
                           "while state_dict is being loaded.")
        model.load_state_dict(checkpoint)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        bs = config['data_loader']['args']['batch_size']
        input_A = model.get_input_A(bs)
        input_B = model.get_input_B(bs)

        out_a, out_b = test_paths(config)

        logger.debug('Starting...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                fake_B = 0.5 * (model.G_A2B(real_A).data + 1)
                fake_A = 0.5 * (model.G_B2A(real_B).data + 1)

                save_image(fake_A, out_a + '/%04d.png' % (batch_idx+1))
                save_image(fake_B, out_b + '/%04d.png' % (batch_idx+1))

                logger.info('Generated images %04d of %04d' % (batch_idx+1, len(data_loader)))

        logger.info(f'Finished writing to "{out_a}" and "{out_b}".')



