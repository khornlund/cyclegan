import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)


def L1_loss(output, target):
    return F.l1_loss(output, target)


class CycleGanCriterion:

    def gan(self, output, target):
        return mse_loss(output, target)

    def cycle(self, output, target):
        return L1_loss(output, target)

    def identity(self, output, target):
        return L1_loss(output, target)
