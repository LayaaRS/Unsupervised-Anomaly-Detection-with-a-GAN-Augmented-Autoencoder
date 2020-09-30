import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_normalization import SpectralNorm

##################  Generator, Encoder, Discriminator for Leukemia dataset size 3 * 220 * 220

class Generator_Leukemia(nn.Module):

    def __init__(self, latent_size):
        super(Generator_Leukemia, self).__init__()
        self.latent_size = latent_size

        self.output_bias = nn.Parameter(torch.zeros(3, 220, 220), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 5, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 4, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 4, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 2, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        # output = torch.sigmoid(output + self.output_bias)
        # print ('Generator output size: ', output.shape)
        return output


class Encoder_Leukemia(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder_Leukemia, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 5, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 5, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 5, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.latent_size, 1, stride=1, bias=True)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        # print ('Encoder output size: ', output.shape)
        return output


class Discriminator_xz_Leukemia(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1):
        super(Discriminator_xz_Leukemia, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size

        self.infer_x = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 5, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 4, stride=3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, 3, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.logit = nn.Conv2d(1024, self.output_size, 1, stride=1, bias=True)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        itermediate_layer = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        logit = self.logit(itermediate_layer)
        # print (itermediate_layer.shape)
        # print (logit.squeeze().shape)
        return logit.squeeze(), itermediate_layer


class Discriminator_xx_Leukemia(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_Leukemia, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 32, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(32, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(128, 128, 5, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True)      
        )

        self.joint_xx_linear = nn.Sequential(
            SpectralNorm(nn.Linear(4608, 1)))

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        itermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(itermediate_layer.view(itermediate_layer.shape[0], -1))
        return logit.squeeze(), itermediate_layer.squeeze()


class Discriminator_zz_Leukemia(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_Leukemia, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )
    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        itermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


################################ MNIST dataset (1, 28, 28)

class Generator_MNIST(nn.Module):

    def __init__(self, latent_size):
        super(Generator_MNIST, self).__init__()
        self.latent_size = latent_size

        self.output_bias = nn.Parameter(torch.zeros(1, 28, 28), requires_grad=True)
        self.Linear = nn.Sequential(
            nn.Linear(self.latent_size, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 7*7*128, bias=True),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(),
            )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(7*7*128, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Linear(input.view(input.shape[0], -1))
        output = torch.unsqueeze(output, 2).unsqueeze(3)
        output = self.main(output)
        return output


class Encoder_MNIST(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder_MNIST, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 32, 3, stride=1, bias=False)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),            
        )
        self.main2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(32, 64, 3, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
            )

        self.main3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
            )

        self.main4 = nn.Sequential(
            SpectralNorm(nn.Linear(3200, self.latent_size, bias=True))
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3.view(x3.shape[0], -1))
        output = output.unsqueeze(2).unsqueeze(3)
        return output


class Discriminator_xz_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xz_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.infer_x = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 4, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
        )

        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Linear(2112, 1024, bias=True)),
            nn.LeakyReLU(inplace=True),
        )

        self.logit = SpectralNorm(nn.Linear(1024, 1, bias=True))

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        itermediate_layer = self.infer_joint(torch.cat([output_x.view(output_x.shape[0], -1), output_z.view(output_z.shape[0], -1)], dim=1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


class Discriminator_xx_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True)            
        )

        self.joint_xx_linear = nn.Sequential(
            SpectralNorm(nn.Linear(2048, 1))
            )

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        itermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(itermediate_layer.view(itermediate_layer.shape[0], -1))
        return logit.squeeze(), itermediate_layer.squeeze()


class Discriminator_zz_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )
    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        itermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


##################################### CIFAR10 dataset (3, 32, 32)

class Generator_CIFAR10(nn.Module):

    def __init__(self, latent_size):
        super(Generator_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.2

        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 512, 4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 5, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 3, 4, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Encoder_CIFAR10(nn.Module):

    def __init__(self, latent_size, noise=False, do_spectral_norm=True):
        super(Encoder_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.2

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value , inplace=True),            
        )
        self.main2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value , inplace=True)
            )

        self.main3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value , inplace=True)
            )

        self.main4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, self.latent_size, 4, stride=1, bias=True, padding=1)),
            # nn.BatchNorm2d(self.latent_size)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output


class Discriminator_xz_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xz_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.infer_x = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(256, 512, 4, stride=3, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1024, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            )

    def forward(self, x, z):
        x = x
        z = z
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        itermediate_layer = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


class Discriminator_xx_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True)            
        )

        self.joint_xx_linear = nn.Sequential(
            SpectralNorm(nn.Linear(3200, 1))
            )

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        itermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(itermediate_layer.view(itermediate_layer.shape[0], -1))
        return logit.squeeze(), itermediate_layer.squeeze()


class Discriminator_zz_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )
    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        itermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer

