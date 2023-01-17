import torch.nn as nn
from config import Config


class GenerateorNet(nn.Module):
    """定义生成器神经网络"""
    def __init__(self, opt: Config) -> None:
        super().__init__()
        self.ngf = opt.ndf
        self.generator = nn.Sequential(
            # input: [opt.nz, 1, 1]
            # output: [ngf*8, 4, 4]
            nn.ConvTranspose2d(in_channels=opt.noise_dim, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias =False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            # input: [ngf*8, 4, 4]
            # output: [ngf*4, 8, 8]
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            # input: [ngf*4, 8, 8]
            # output: [ngf*2, 16, 16]
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            # input: [ngf*2, 16, 16]
            # output: [ngf, 32, 32]
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # input: [ngf, 32, 32]
            # output: [ngf, 96, 96]
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias =False),

            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)



class DiscriminatorNet(nn.Module):
    """定义判别器神经网络"""
    def __init__(self, opt: Config) -> None:
        super().__init__()
        self.ndf = opt.ndf
        self.discriminator = nn.Sequential(
            # input: [3, 96, 96]
            # output: [ndf*2, 16, 16]
            nn.Conv2d(in_channels=3, out_channels= self.ndf, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.GELU(),

            # input: [ndf, 32, 32]
            # output: [ndf*2, 16, 16]
            nn.Conv2d(in_channels= self.ndf, out_channels= self.ndf * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.GELU(),

            # input: [ndf*2, 16, 16]
            # output: [ndf*4, 8, 8]
            nn.Conv2d(in_channels= self.ndf * 2, out_channels= self.ndf *4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.GELU(),

            # input: [ndf*4, 8, 8]
            # output: [ndf*8, 4, 4]
            nn.Conv2d(in_channels= self.ndf *4, out_channels= self.ndf *8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf *8),
            nn.GELU(),

            # input: [ndf*8, 4, 4]
            # output: [1, 1, 1]
            nn.Conv2d(in_channels= self.ndf *8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x).view(-1)