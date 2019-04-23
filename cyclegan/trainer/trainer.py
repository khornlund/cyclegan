import random

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

from cyclegan.base import BaseTrainer


class CycleGanTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, optimizer, resume, config, data_loader, lr_scheduler):
        super(CycleGanTrainer, self).__init__(model, loss, [], optimizer, resume, config)
        self.config = config
        self.data_loader = data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        bs = config['data_loader']['args']['batch_size']
        input_nc  = config['arch']['args']['input_nc']
        output_nc = config['arch']['args']['output_nc']
        x_size = 256
        y_size = 256

        self.input_A = torch.cuda.FloatTensor(bs, input_nc, x_size, y_size)
        self.input_B = torch.cuda.FloatTensor(bs, output_nc, x_size, y_size)
        self.target_real = Variable(torch.cuda.FloatTensor(bs).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(torch.cuda.FloatTensor(bs).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        for batch_idx, batch in enumerate(self.data_loader):
            # Set model input
            real_A = Variable(self.input_A.copy_(batch['A']))
            real_B = Variable(self.input_B.copy_(batch['B']))

            # -- Generators A2B and B2A ---------------------------------------
            self.optimizer.G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.model.G_A2B(real_B)
            loss_identity_B = self.loss.identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = self.model.G_B2A(real_A)
            loss_identity_A = self.loss.identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = self.model.G_A2B(real_A)
            pred_fake = self.model.D_B(fake_B)
            loss_GAN_A2B = self.loss.gan(pred_fake, self.target_real)

            fake_A = self.model.G_B2A(real_B)
            pred_fake = self.model.D_A(fake_A)
            loss_GAN_B2A = self.loss.gan(pred_fake, self.target_real)

            # Cycle loss
            recovered_A = self.model.G_B2A(fake_B)
            loss_cycle_ABA = self.loss.cycle(recovered_A, real_A) * 10.0

            recovered_B = self.model.G_A2B(fake_A)
            loss_cycle_BAB = self.loss.cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = (loss_identity_A + loss_identity_B + loss_GAN_A2B +
                      loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB)
            loss_G.backward()

            self.optimizer.G.step()
            # -----------------------------------------------------------------

            # -- Discriminator A ----------------------------------------------
            self.optimizer.D_A.zero_grad()

            # Real loss
            pred_real   = self.model.D_A(real_A)
            loss_D_real = self.loss.gan(pred_real, self.target_real)

            # Fake loss
            fake_A      = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake   = self.model.D_A(fake_A.detach())
            loss_D_fake = self.loss.gan(pred_fake, self.target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            self.optimizer.D_A.step()
            # -----------------------------------------------------------------

            # -- Discriminator B ----------------------------------------------
            self.optimizer.D_B.zero_grad()

            # Real loss
            pred_real   = self.model.D_B(real_B)
            loss_D_real = self.loss.gan(pred_real, self.target_real)

            # Fake loss
            fake_B      = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake   = self.model.D_B(fake_B.detach())
            loss_D_fake = self.loss.gan(pred_fake, self.target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            self.optimizer.D_B.step()
            # -----------------------------------------------------------------

            losses = {
                'loss_G': loss_G,
                'loss_G_identity': (loss_identity_A + loss_identity_B),
                'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                'loss_D': (loss_D_A + loss_D_B)
            }

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            for loss_type, loss in losses.items():
                self.writer.add_scalar(loss_type, loss)
            self.writer.add_image('Real_A', make_grid(real_A))
            self.writer.add_image('Real_B', make_grid(real_B))
            self.writer.add_image('Fake_A', make_grid(fake_A))
            self.writer.add_image('Fake_B', make_grid(fake_B))

            if batch_idx % self.log_step == 0:
                self._log_batch(epoch, batch_idx, self.data_loader.batch_size,
                                len(self.data_loader), losses)

        self.lr_scheduler.G.step()
        self.lr_scheduler.D_A.step()
        self.lr_scheduler.D_B.step()

        return losses

    def _log_batch(self, epoch, batch_idx, batch_size, n_batches, losses):
            n_complete = batch_idx * batch_size
            n_samples = n_batches * batch_size
            percent = 100.0 * n_complete / n_samples
            msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] '
            msg += ' | '.join([f'{key}: {val:.4f}' for key, val in losses.items()])
            self.logger.debug(msg)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
