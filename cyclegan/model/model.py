import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cyclegan.base import BaseModel


class ResidualBlock(BaseModel):
    def __init__(self, in_features, verbose=0):
        super(ResidualBlock, self).__init__(verbose=verbose)

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(BaseModel):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, verbose=0):
        super(Generator, self).__init__(verbose=verbose)

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)


class GeneratorA2B(Generator):
    pass


class GeneratorB2A(Generator):
    pass


class Discriminator(BaseModel):
    def __init__(self, input_nc, verbose=0):
        super(Discriminator, self).__init__(verbose=verbose)

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class DiscriminatorA(Discriminator):
    pass


class DiscriminatorB(Discriminator):
    pass


class CycleGan:
    """Container for Generators and Discriminators"""

    @property
    def G_A2B(self): return self.netG_A2B

    @property
    def G_B2A(self): return self.netG_B2A

    @property
    def D_A(self): return self.netD_A

    @property
    def D_B(self): return self.netD_B

    def __init__(self, input_nc, output_nc, img_size, verbose):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.img_size = img_size

        self.netG_A2B = GeneratorA2B(input_nc, output_nc, verbose=verbose)
        self.netG_B2A = GeneratorB2A(output_nc, input_nc, verbose=verbose)
        self.netD_A = DiscriminatorA(input_nc, verbose=verbose)
        self.netD_B = DiscriminatorB(output_nc, verbose=verbose)

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

    @property
    def _models(self):
        return [self.G_A2B, self.G_B2A, self.D_A, self.D_B]

    def to(self, device):
        [m.to(device) for m in self._models]
        return self

    def eval(self):
        [m.eval() for m in self._models]

    def state_dict(self):
        return {m.__class__.__name__: m.state_dict() for m in self._models}

    def load_state_dict(self, checkpoint):
        for m in self._models:
            m.load_state_dict(checkpoint[m.__class__.__name__])

    def get_input_A(self, batch_size):
        return torch.cuda.FloatTensor(batch_size, self.input_nc, self.img_size, self.img_size)

    def get_input_B(self, batch_size):
        return torch.cuda.FloatTensor(batch_size, self.output_nc, self.img_size, self.img_size)

    def get_target(self, batch_size):
        return Variable(torch.cuda.FloatTensor(batch_size).fill_(1.0), requires_grad=False)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)
