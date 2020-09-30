import torch
import torch.nn as nn


class Generator(nn.Module):  
	def __init__(self, latent_size):
		super(Generator, self).__init__()
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
		x = self.Linear(x.view(x.shape[0],-1))
		x = torch.unsqueeze(x, 2).unsqueeze(3)
		for layer in self.deconv:
			x = layer(x)
		# output = torch.sigmoid(x)
		return x
	
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