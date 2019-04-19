import torch


class CycleGanScheduler:

    def __init__(self, optimizer, config):

        self._G   = torch.optim.lr_scheduler.LambdaLR(optimizer.G,
            lr_lambda=LambdaLR(config['training']['epochs'], 0, 100).step)
        self._D_A = torch.optim.lr_scheduler.LambdaLR(optimizer.D_A,
            lr_lambda=LambdaLR(config['training']['epochs'], 0, 100).step)
        self._D_B = torch.optim.lr_scheduler.LambdaLR(optimizer.D_B,
            lr_lambda=LambdaLR(config['training']['epochs'], 0, 100).step)

    @property
    def G(self): return self._G

    @property
    def D_A(self): return self._D_A

    @property
    def D_B(self): return self._D_B

    def __str__(self):
        return f'Schedulers:\n\tG: {self.G}\n\tD_A: {self.D_A}\n\tD_B: {self.D_B}'

    def __repr__(self):
        return self.__str__()


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before training ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / \
                     (self.n_epochs - self.decay_start_epoch)
