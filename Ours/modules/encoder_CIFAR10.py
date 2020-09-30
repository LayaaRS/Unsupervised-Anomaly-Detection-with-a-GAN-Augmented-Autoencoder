import torch
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self, latent_size, dropout=0.3, noise=False):
		super(Encoder, self).__init__()
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
            nn.LeakyReLU(self.leaky_value , inplace=True), 
		))
		self.conv.append(nn.Sequential(    
			nn.Conv2d(128, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value , inplace=True)
		))
		self.conv.append(nn.Sequential(
			nn.Conv2d(256, 512, 4, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value , inplace=True)
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
			if isinstance(m , nn.Conv2d):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
			elif isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
			elif isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)