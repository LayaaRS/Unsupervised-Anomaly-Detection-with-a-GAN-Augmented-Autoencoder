import torch
import torch.nn as nn


class Discriminator_Leukemia(nn.Module):
    def __init__(self, latent_size, dropout=0.2, output_size=2):
        super(Discriminator_Leukemia, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1

        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(2, stride=2),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            # nn.MaxPool2d(2, stride=2),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(2, stride=2),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(2, stride=2),
        ))

        self.final = nn.Linear(30976, self.output_size, bias=True)      
        self.init_weights()
        
    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)

        output = self.convs[-1](output_pre)
        output = self.final(output.view(output.shape[0], -1))
        return output.squeeze(), output_pre

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Generator_Leukemia(nn.Module):  
    def __init__(self, latent_size):
        super(Generator_Leukemia, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.1
        
        self.output_bias = nn.Parameter(torch.zeros(3, 220, 220), requires_grad=True)
        self.deconv = nn.ModuleList()
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 2048, kernel_size=5, stride=1, bias=True),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(2048, 1024, 5, stride=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(1024, 512, 5, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(512, 512, 5, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(512, 256, 5, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(256, 128, 5, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),   
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(128, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(64, 64, 4, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv = nn.Sequential(    
            nn.Conv2d(64, 3, 1, stride=1, bias=True),
            nn.Tanh()
        )
        
        self.init_weights()

    def forward(self, noise):
        x = noise
        for layer in self.deconv:
            x = layer(x)
        x = self.conv(x)
        # output = torch.sigmoid(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Encoder_Leukemia(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_Leukemia, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.1
        self.conv = nn.ModuleList()

        if noise:
            self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()
        
        self.conv.append(nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))
        self.conv.append(nn.Sequential(    
            nn.Conv2d(64, 128, 3, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))
        self.conv.append(nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(2, stride=2),
        ))
        self.conv.append(nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))

        self.conv.append(nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(2, stride=2),
        ))

        self.conv.append(nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        ))

        self.linear = nn.Linear(123904, self.latent_size, bias=True)
        
        self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv:
            output = layer(output)

        output = self.linear(output.view(output.shape[0], -1))   
        output = self.nonlinear(output)
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        # output = self.nonlinear(output)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Discriminator_CIFAR10(nn.Module):
    def __init__(self, latent_size, dropout=0.3, output_size=2):
        super(Discriminator_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=3, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.final = nn.Sequential(
            nn.Linear(512, self.output_size, bias=True)
            )
        
        self.init_weights()
        
    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output = self.convs[-1](output_pre)
        output = self.final(output.view(output.shape[0], -1))
        return output.squeeze(), output_pre

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Generator_CIFAR10(nn.Module):  
    def __init__(self, latent_size):
        super(Generator_CIFAR10, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.2
        self.deconv = nn.ModuleList()
        
        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
        
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, stride=2, bias=True, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(512, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(256, 128, 5, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(128, 3, 4, stride=2, bias=True),
            nn.Tanh()
        ))
        
        self.init_weights()

    def forward(self, noise):
        x = noise
        for layer in self.deconv:
            x = layer(x)
        # output = torch.sigmoid(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Encoder_CIFAR10(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_CIFAR10, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.2
        self.conv = nn.ModuleList()

        if noise:
            self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()
        
        self.conv.append(nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True), 
        ))
        self.conv.append(nn.Sequential(    
            nn.Conv2d(128, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            nn.Conv2d(512, self.latent_size, 4, stride=1, bias=True, padding=1),
        ))

        self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Discriminator_MNIST(nn.Module):
    def __init__(self, latent_size, dropout=0.2, output_size=2):
        super(Discriminator_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.convs.append(nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))

        self.final1 = nn.Linear(6400, 1024, bias=True)
        self.final2 = nn.Linear(1024, self.output_size, bias=True)

        self.init_weights()
        
    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output = self.convs[-1](output_pre)
        output = self.final1(output.view(output.shape[0], -1))
        output = self.final2(output)
        # output = torch.sigmoid(output)        
        return output.squeeze(), output_pre

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Generator_MNIST(nn.Module):  
    def __init__(self, latent_size):
        super(Generator_MNIST, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.1
        self.deconv = nn.ModuleList()
        
        self.output_bias = nn.Parameter(torch.zeros(1, 28, 28), requires_grad=True)
        
        self.Linear = nn.Sequential(
            nn.Linear(self.latent_dim, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 7*7*128, bias=True),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(),
            )
        
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(7*7*128, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(64, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(64, 1, 4, stride=2, bias=True),
            nn.Tanh(),
        ))
        
        self.init_weights()

    def forward(self, noise):
        x = noise
        x = self.Linear(x.view(x.shape[0], -1))
        x = torch.unsqueeze(x, 2).unsqueeze(3)
        for layer in self.deconv:
            x = layer(x)
        # output = torch.sigmoid(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Encoder_MNIST(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_MNIST, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.1
        self.conv = nn.ModuleList()

        # if noise:
        #   self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()
        
        self.conv.append(nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, bias=True),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(    
            nn.Conv2d(64, 128, 3, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))

        self.linear = nn.Linear(6400, self.latent_size, bias=True)

        self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv:
            output = layer(output)

        output = self.linear(output.view(output.shape[0], -1))
        output = self.nonlinear(output)
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        # output = self.nonlinear(output)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
