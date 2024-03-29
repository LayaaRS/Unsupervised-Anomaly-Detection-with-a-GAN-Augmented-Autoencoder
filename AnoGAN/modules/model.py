import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, latent_size, dropout=0.2, output_size=2):
        super(Discriminator, self).__init__()
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

        # self.final = nn.Conv2d(512, self.output_size, 1, stride=1, bias=True)
        self.final = nn.Linear(30976, self.output_size, bias=True)
        
        self.init_weights()
        
    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output = self.convs[-1](output_pre)
        output = self.final(output.view(output.shape[0], -1))
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


class Generator(nn.Module):  
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.1
        
        self.output_bias = nn.Parameter(torch.zeros(3, 220, 220), requires_grad=True)
        self.deconv = nn.ModuleList()
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 256, kernel_size=5, stride=1,bias=True),
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
            nn.ConvTranspose2d(64, 32, 5, stride=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),   
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(32, 32, 5, stride=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.deconv.append(nn.Sequential(    
            nn.ConvTranspose2d(32, 32, 4, stride=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv = nn.Sequential(    
            nn.Conv2d(32, 3, 1, stride=1, bias=True),
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
