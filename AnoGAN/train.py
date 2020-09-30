import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import numpy as np
import argparse
import datetime
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt

import modules
from utils import dataset
from modules import generator, discriminator


device = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--trainset_path', default='./data', help='path to the train data')
parser.add_argument('-ts', '--testset_path', default='./data', help='path to the tets data')
parser.add_argument('-lr', '--learning_rate', default=1e-4, help='learning rate')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-ep', '--epochs', type=int, default=25, help='number of epochs')
parser.add_argument('-z', '--latent_size', type=int, default=15, help='latent size')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='daropout')
parser.add_argument('-ds', '--dataset', default='Leukemia', help='which dataset the model is running on')
parser.add_argument('-dz', '--distribution', default='Uniform', help='choose the distribution you want to z selected from')

args = parser.parse_args()

normalize_CIFAR10 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_CIFAR10 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
							  normalize_CIFAR10])

normalize_MNIST = torchvision.transforms.Normalize((0.5, ), (0.5, ))
transform_MNIST = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
							  normalize_MNIST])

normalize_Leukemia = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_Leukemia = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
								torchvision.transforms.RandomHorizontalFlip(),
								torchvision.transforms.RandomVerticalFlip(),
								torchvision.transforms.RandomCrop(220),
								torchvision.transforms.ToTensor(),
								normalize_Leukemia])
norm_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
								normalize_Leukemia])
test_transform_Leukemia = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
								torchvision.transforms.FiveCrop(220),
								torchvision.transforms.Lambda(lambda crops: torch.stack([norm_transform(crop)
								for crop in crops]))])

if args.dataset == 'MNIST':
	train_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=True, transform=transform_MNIST, target_transform=None, download=False)
	test_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=False, transform=transform_MNIST, target_transform=None, download=False)
	idx = torch.as_tensor(train_MNIST.targets) != 1
	dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==1)[0])
	dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx!=1)[0])
	idx = torch.as_tensor(test_MNIST.targets) != 1
	dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==1)[0])
	dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx!=1)[0])
	trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
	testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
	testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
	print ('MNIST train: ', len(trainloader), 'MNIST test normal:', len(testloader_normal), 'MNIST test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'CIFAR10':
	train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=transform_CIFAR10, target_transform=None, download=False)
	test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False, transform=transform_CIFAR10, target_transform=None, download=False)
	idx = torch.tensor(train_CIFAR10.targets) != 1
	dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==1)[0])
	dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx!=1)[0])
	idx = torch.tensor(test_CIFAR10.targets) != 1
	dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==1)[0])
	dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx!=1)[0])
	trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
	testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
	testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
	print ('CIFAR10 train: ', len(trainloader), 'CIFAR10 test normal: ', len(testloader_normal), 'CIFAR10 test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'Leukemia':
	train_Leukemia = dataset.ImageFolderPaths(args.trainset_path, transform = transform_Leukemia) 
	test_Leukemia = dataset.ImageFolderPaths(args.testset_path, transform = test_transform_Leukemia)
	trainloader = torch.utils.data.DataLoader(train_Leukemia, batch_size= args.batch_size, shuffle=True, num_workers= 0)
	testloader = torch.utils.data.DataLoader(test_Leukemia, batch_size= 1, shuffle=True, num_workers= 1)



torch.manual_seed(100)
learning_rate = args.learning_rate
beta1 = 0.5
beta2 = 0.999
num_epochs = args.epochs
latent_size = args.latent_size
distribution = args.distribution

dis_criterion = nn.CrossEntropyLoss()

if args.dataset == 'CIFAR10':
	gen = generator_CIFAR10.Generator(latent_size)
	dis = discriminator_CIFAR10.Discriminator(latent_size)
elif args.dataset == 'MNIST':
	gen = generator_MNIST.Generator(latent_size)
	dis = discriminator_MNIST.Discriminator(latent_size)
else:
	gen = generator.Generator(latent_size)
	dis = discriminator.Discriminator(latent_size)  


dis.to(device)
gen.to(device)


optimizer_d = optim.Adam(dis.parameters(), 
						 lr=learning_rate, 
						 betas=(beta1, beta2), weight_decay=0.1)
optimizer_g = optim.Adam(gen.parameters(), 
						  lr=learning_rate,
						  betas=(beta1, beta2), weight_decay=0.5)

# scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.1, last_epoch=-1)
# scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.1, last_epoch=-1)


for epoch in range(num_epochs):
	if epoch == 0:
		print ('training starts')
	dis.train()
	gen.train()
	for i, batch in enumerate(trainloader):
		x = batch[0].to(device)

		valid_label = Variable(torch.ones((x.size(0)), dtype=torch.long), requires_grad=False).to(device)
		fake_label = Variable(torch.zeros((x.size(0)), dtype=torch.long), requires_grad=False).to(device)
		
		# Generator
		optimizer_g.zero_grad()
		
		if distribution == 'Gaussian':
			z_random = torch.randn(x.shape[0], latent_size, 1, 1).to(device)
		elif distribution == 'Uniform':
			z_random = (torch.rand(x.shape[0], latent_size, 1, 1) * 2 -1).to(device)

		x_fake = gen(z_random)
		dis_fake,_ = dis(x_fake)
		loss_gen = dis_criterion(dis_fake, valid_label)
		loss_gen.backward()
		optimizer_g.step()
	

		# Discriminator
		optimizer_d.zero_grad()
		dis_real,_ = dis(x)
		loss_real = dis_criterion(dis_real, valid_label)
		dis_fake,_ = dis(x_fake.detach())
		loss_fake = dis_criterion(dis_fake, fake_label)
		loss_dis = (loss_fake + loss_real)
		loss_dis.backward()
		optimizer_d.step()

	print("epoch: %d, discriminator loss: %.2f, generator loss: %.2f" %(epoch, loss_dis.item(), loss_gen.item()))    
		
print ('End of training')

filepath = str(args.dataset) + str(num_epochs) + 'epochs' + str(distribution) + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

torch.save(gen.state_dict(), './models/' + filepath + 'G.pt')
torch.save(dis.state_dict(), './models/' + filepath + 'D.pt')