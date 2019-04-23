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
        self.invert_norm_transform = data_loader.invert_norm_transform()
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_loader)))

        bs = config['data_loader']['args']['batch_size']

        self.loss_weight_identity = config['training']['loss_weight']['identity']
        self.loss_weight_cycle    = config['training']['loss_weight']['cycle']

        self.input_A = model.get_input_A(bs)
        self.input_B = model.get_input_B(bs)
        self.target_real = model.get_target(bs)
        self.target_fake = model.get_target(bs)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
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
            loss_identity_B = self.loss.identity(same_B, real_B) * self.loss_weight_identity
            # G_B2A(A) should equal A if real A is fed
            same_A = self.model.G_B2A(real_A)
            loss_identity_A = self.loss.identity(same_A, real_A) * self.loss_weight_identity

            # GAN loss
            fake_B = self.model.G_A2B(real_A)
            pred_fake = self.model.D_B(fake_B)
            loss_GAN_A2B = self.loss.gan(pred_fake, self.target_real)

            fake_A = self.model.G_B2A(real_B)
            pred_fake = self.model.D_A(fake_A)
            loss_GAN_B2A = self.loss.gan(pred_fake, self.target_real)

            # Cycle loss
            recovered_A = self.model.G_B2A(fake_B)
            loss_cycle_ABA = self.loss.cycle(recovered_A, real_A) * self.loss_weight_cycle

            recovered_B = self.model.G_A2B(fake_A)
            loss_cycle_BAB = self.loss.cycle(recovered_B, real_B) * self.loss_weight_cycle

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

            if batch_idx % self.log_step == 0:
                self._log_batch(epoch, batch_idx, self.data_loader.batch_size,
                                len(self.data_loader), losses)

                self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
                for loss_type, loss in losses.items():
                    self.writer.add_scalar(loss_type, loss)
                self.writer.add_image('AtoB/Real_A', make_grid(real_A, normalize=True))
                self.writer.add_image('AtoB/Fake_B', make_grid(fake_B, normalize=True))
                self.writer.add_image('BtoA/Real_B', make_grid(real_B, normalize=True))
                self.writer.add_image('BtoA/Fake_A', make_grid(fake_A, normalize=True))

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
