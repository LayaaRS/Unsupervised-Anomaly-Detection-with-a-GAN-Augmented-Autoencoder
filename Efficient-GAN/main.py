import argparse
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model import *
import os
import datetime
import dataset
import numpy as np


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', default='Leukemia', help='The datset you want the model to be trained on')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-ep', '--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-z', '--latent_size', type=int, default=15, help='latent size')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('-sm','--save_model_dir', default='results/')
parser.add_argument('-dz', '--distribution', default='Gaussian', help='choose the distribution you want to z selected from')
parser.add_argument('-cl', '--class_label_anomalous', default=0, type=int, help='select which class label should be selected as the anomalous')
opt = parser.parse_args()

torch.manual_seed(0)
batch_size = opt.batch_size
lr = 1e-4
latent_size = opt.latent_size
num_epochs = opt.epochs
cuda_device = "0"


# opt.save_image_dir = opt.save_model_dir + 'images'

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


normalize_CIFAR10 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_CIFAR10 = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomVerticalFlip(),
                                torchvision.transforms.ToTensor(),
                                normalize_CIFAR10])

normalize_MNIST = torchvision.transforms.Normalize((0.5, ), (0.5, ))
transform_MNIST = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomVerticalFlip(),
                                torchvision.transforms.ToTensor(),
                                normalize_MNIST])

normalize_Leukemia = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_Leukemia = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomVerticalFlip(),
                                torchvision.transforms.RandomCrop(220),
                                torchvision.transforms.ToTensor(),
                                normalize_Leukemia])

if opt.dataset == 'MNIST':
    train_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=True, transform=transform_MNIST, target_transform=None, download=False)
    test_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=False, transform=transform_MNIST, target_transform=None, download=False)
    idx = torch.as_tensor(train_MNIST.targets) == torch.tensor(opt.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==False)[0])
    idx = torch.as_tensor(test_MNIST.targets) == torch.tensor(opt.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=opt.batch_size, drop_last=True, shuffle=True,)
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    print (len(trainloader))
elif opt.dataset == 'CIFAR10':
    train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=transform_CIFAR10, target_transform=None, download=False)
    test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False, transform=transform_CIFAR10, target_transform=None, download=False)
    idx = torch.tensor(train_CIFAR10.targets) == torch.tensor(opt.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==False)[0])
    idx = torch.tensor(test_CIFAR10.targets) == torch.tensor(opt.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=opt.batch_size, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    print (len(trainloader))
elif opt.dataset == 'Leukemia':
    dset_train = dataset.ImageFolderPaths('./data/Leukemia/train/', transform = transform_Leukemia)
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True)

if opt.dataset == 'MNIST':
    # ##### for 1*28*28 images of MNIST dataset  
    netE = tocuda(Encoder_MNIST(latent_size, True))
    netG = tocuda(Generator_MNIST(latent_size))
    netD = tocuda(Discriminator_MNIST(latent_size, 0.2, 1))
elif opt.dataset == 'CIFAR10':
    ##### for 3*32*32 images of CIFAR10 dataset
    netE = tocuda(Encoder_CIFAR10(latent_size, True))
    netG = tocuda(Generator_CIFAR10(latent_size))
    netD = tocuda(Discriminator_CIFAR10(latent_size, 0.2, 1))
else:
    ##### for 3*220*220 images of Leukemia dataset
    netE = tocuda(Encoder_Leukemia(latent_size, True))
    netG = tocuda(Generator_Leukemia(latent_size))
    netD = tocuda(Discriminator_Leukemia(latent_size, 0.2, 1))


netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):

    i = 0
    for (data, target) in trainloader:

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)

        if loss_g.item() < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 10 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item())

        # if i % 50 == 0:
        #     vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
        #     vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (opt.save_image_dir))

        i += 1

    if epoch % 100 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (opt.save_model_dir, epoch))
    if epoch > 30:
        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))


filepath = str(opt.dataset) + str(num_epochs) + 'epochs' + str(opt.distribution) + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

torch.save(netG.state_dict(), opt.save_model_dir + filepath + 'G.pt')
torch.save(netE.state_dict(), opt.save_model_dir + filepath + 'E.pt')
torch.save(netD.state_dict(), opt.save_model_dir + filepath + 'D.pt')

