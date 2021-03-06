import itertools

import torch


class CycleGanOptimizer:

    def __init__(self, model, config):
        lr = config['optimizer']['lr']
        betas = (config['optimizer']['beta_lower'], config['optimizer']['beta_upper'])
        self._G = torch.optim.Adam(
            itertools.chain(model.G_A2B.parameters(), model.G_B2A.parameters()),
            lr=lr, betas=betas)
        self._D_A = torch.optim.Adam(model.D_A.parameters(), lr=lr, betas=betas)
        self._D_B = torch.optim.Adam(model.D_B.parameters(), lr=lr, betas=betas)

    @property
    def G(self): return self._G

    @property
    def D_A(self): return self._D_A

    @property
    def D_B(self): return self._D_B

    @property
    def _models(self):
        return {'G': self.G, 'D_A': self.D_A, 'D_B': self.D_B}

    def state_dict(self):
        return {name: model.state_dict() for name, model in self._models.items()}

    def load_state_dict(self, checkpoint):
        for name, model in self._models.items():
            model.load_state_dict(checkpoint[name])

    def __str__(self):
        return f'Optimizers:\n\tG: {self.G}\n\tD_A: {self.D_A}\n\tD_B: {self.D_B}'

    def __repr__(self):
        return self.__str__()
