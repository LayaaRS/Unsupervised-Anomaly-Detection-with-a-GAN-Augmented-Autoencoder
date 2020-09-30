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
		output = self.final1(output.view(output.shape[0],-1))
		output = self.final2(output)
		# output = torch.sigmoid(output)		
		return output.squeeze(), output_pre

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