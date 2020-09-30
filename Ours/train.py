import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.utils as vutils

import time
import numpy as np
import argparse
import datetime
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt

import modules
from utils import dataset
from modules import encoder_ALL, generator_ALL, discriminator_ALL
from modules import encoder_CIFAR10, generator_CIFAR10, discriminator_CIFAR10
from modules import encoder_MNIST, generator_MNIST, discriminator_MNIST


device = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--trainset_path', default='./data', help='path to the train data')
parser.add_argument('-ts', '--testset_path', default='./data', help='path to the tets data')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-ep', '--epochs', type=int, default=25, help='number of epochs')
parser.add_argument('-z', '--latent_size', type=int, default=100, help='latent size')
parser.add_argument('-d', '--dropout', type=float, default=0.3, help='daropout')
parser.add_argument('-ef', '--encoder_first_train', type=int, default=0, help='number of epochs encoder first train')
parser.add_argument('-ds', '--dataset', default='Leukemia', help='which dataset the model is running on')
parser.add_argument('-sm','--save_model_dir', default='./models/')
parser.add_argument('-cl', '--class_label_anomalous', type=int, default=0, help='select which class label should be selected as the anomalous')
parser.add_argument('-dz', '--distribution', default='Gaussian', help='choose the distribution you want to z selected from')

args = parser.parse_args()
# args.save_image_dir = args.save_model_dir + 'images'

print (args.class_label_anomalous)

normalize_CIFAR10 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_CIFAR10 = torchvision.transforms.Compose([torchvision.transforms.Resize((36, 36)),
								torchvision.transforms.RandomVerticalFlip(),
								torchvision.transforms.RandomCrop(32),
								torchvision.transforms.ToTensor(),
							  	normalize_CIFAR10])

normalize_MNIST = torchvision.transforms.Normalize((0.5, ), (0.5, ))
transform_MNIST = torchvision.transforms.Compose([torchvision.transforms.Resize((35, 35)),
								torchvision.transforms.RandomVerticalFlip(),
								torchvision.transforms.RandomCrop(28),
								torchvision.transforms.ToTensor(),
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
	idx = torch.as_tensor(train_MNIST.targets) != torch.tensor(args.class_label_anomalous)
	dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==True)[0])
	dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==False)[0])
	idx = torch.as_tensor(test_MNIST.targets) != torch.tensor(args.class_label_anomalous)
	dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==True)[0])
	dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==False)[0])
	trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
	testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
	testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
	print ('MNIST train: ', len(trainloader), 'MNIST test normal:', len(testloader_normal), 'MNIST test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'CIFAR10':
	train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=transform_CIFAR10, target_transform=None, download=False)
	test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False, transform=transform_CIFAR10, target_transform=None, download=False)
	idx = torch.tensor(train_CIFAR10.targets) == torch.tensor(args.class_label_anomalous)
	dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==True)[0])
	dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==False)[0])
	idx = torch.tensor(test_CIFAR10.targets) == torch.tensor(args.class_label_anomalous)
	dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==True)[0])
	dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==False)[0])
	trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
	testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
	testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
	print ('CIFAR10 train: ', len(trainloader), 'CIFAR10 test normal: ', len(testloader_normal), 'CIFAR10 test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'Leukemia':
	train_Leukemia = dataset.ImageFolderPaths(args.trainset_path, transform = transform_Leukemia) 
	test_Leukemia = dataset.ImageFolderPaths(args.testset_path, transform = test_transform_Leukemia)
	trainloader = torch.utils.data.DataLoader(train_Leukemia, batch_size= args.batch_size, shuffle=True, num_workers= 0)
	testloader = torch.utils.data.DataLoader(test_Leukemia, batch_size= 1, shuffle=True, num_workers= 1)



start_time = time.time()
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


learning_rate = args.learning_rate
beta1 = 0.9
beta2 = 0.999
num_epochs = args.epochs
latent_size = args.latent_size
distribution = args.distribution
dropout = args.dropout

dis_criterion = nn.CrossEntropyLoss()
aen_criterion = nn.MSELoss()  

if args.dataset == 'CIFAR10':
	enc = encoder_CIFAR10.Encoder(latent_size, dropout)
	gen = generator_CIFAR10.Generator(latent_size)
	dis = discriminator_CIFAR10.Discriminator(latent_size, dropout)
elif args.dataset == 'MNIST':
	enc = encoder_MNIST.Encoder(latent_size, dropout)
	gen = generator_MNIST.Generator(latent_size)
	dis = discriminator_MNIST.Discriminator(latent_size, dropout)
else:
	enc = encoder_new.Encoder(latent_size)
	gen = generator_new.Generator(latent_size)
	dis = discriminator_new.Discriminator(latent_size)  


dis.to(device)
enc.to(device)
gen.to(device)


optimizer_d = optim.Adam(dis.parameters(), 
						 lr=learning_rate, 
						 betas=(beta1, beta2), weight_decay=0.5)
optimizer_g = optim.Adam(gen.parameters(), 
						  lr=learning_rate,
						  betas=(beta1, beta2), weight_decay=0.5)
optimizer_e = optim.Adam(enc.parameters(),
						  lr=learning_rate,
						  betas=(beta1, beta2), weight_decay=1e-3)

gen_train = False
print ('training starts')
for epoch in range(num_epochs):
	dis.train()
	gen.train()
	enc.train()
	for i, batch in enumerate(trainloader):
		x = batch[0].to(device)

		valid_label = Variable(torch.ones((x.size(0)), dtype=torch.long), requires_grad=False).to(device)
		fake_label = Variable(torch.zeros((x.size(0)), dtype=torch.long), requires_grad=False).to(device)

		# encoder
		optimizer_e.zero_grad()
		optimizer_g.zero_grad()
		encoded = enc(x)
		x_hat = gen(encoded)
		loss_aen = aen_criterion(x_hat, x)
		loss_aen.backward()
		optimizer_e.step()
		optimizer_g.step()
		
		# Generator
		if epoch >= args.encoder_first_train:
			gen_train = True
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

			# if epoch % 10 == 0 and epoch >= args.encoder_first_train:
			# 	vutils.save_image(x_fake.cpu().data[:16, ], '%s/fake_%d.png' % (args.save_image_dir, epoch))

	if gen_train == True:    
		print("epoch: %d, discriminator loss: %.2f, generator loss: %.2f, encoder loss: %.2f" %(epoch, loss_dis.item(), loss_gen.item(), loss_aen.item()))
	

end_time = time.time() - start_time
print ('End of training')
print ('time: ', end_time)

## Final model save
filepath = str(args.save_model_dir) + str(args.dataset) + str(num_epochs) + 'epochs' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
torch.save(gen.state_dict(), filepath + 'G')
torch.save(enc.state_dict(), filepath + 'E')
torch.save(dis.state_dict(), filepath + 'D')