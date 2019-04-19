from tqdm import tqdm

import torch

import cyclegan.data_loader.data_loaders as module_data
import cyclegan.model.loss as module_loss
import cyclegan.model.metric as module_metric
import cyclegan.model.model as module_arch
from cyclegan.trainer import Trainer
from temp_pr.utils import setup_logger, setup_logging


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Runner:

    def train(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, config['training']['verbose'])

        logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config)
        valid_data_loader = data_loader.split_validation()

        logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        logger.debug('Getting loss and metric function handles')
        loss = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        logger.debug('Initialising trainer')
        trainer = Trainer(model, loss, metrics, optimizer,
                        resume=resume,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

        trainer.train()
        logger.debug('Finished!')

    def test(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, config['testing']['verbose'])

        logger.debug('Getting data_loader instance')
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        logger.debug('Getting loss and metric function handles')
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.debug('Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        logger.debug('Starting...')
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        logger.info(log)
