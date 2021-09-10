import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import numpy as np
import argparse
import datetime
from sklearn import metrics

from utils import dataset
from modules import model


device = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--trainset_path', default='./data', help='path to the train data')
parser.add_argument('-ts', '--testset_path', default='./data', help='path to the tets data')
parser.add_argument('-rs', '--random_seed', default=0, type=int, help='random seed')
parser.add_argument('-dlr', '--dis_learning_rate', default=0.25, type=float, help='dis learning rate')
parser.add_argument('-glr', '--gen_learning_rate', default=0.00005, type=float, help='gen learning rate')
parser.add_argument('-zlr', '--z_learning_rate', default=0.00005, type=float, help='z learning rate')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-ep', '--epochs', type=int, default=25, help='number of epochs')
parser.add_argument('-epz', '--epochs_z', type=int, default=5, help='number of epochs for noise')
parser.add_argument('-z', '--latent_size', type=int, default=15, help='latent size')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='daropout')
parser.add_argument('-cl', '--class_label', default=0, type=int, help='normal/anomalous class label')
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
transform_Leukemia = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.RandomVerticalFlip(),
                                                     torchvision.transforms.RandomCrop(220),
                                                     torchvision.transforms.ToTensor(),
                                                     normalize_Leukemia])
norm_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 normalize_Leukemia])
test_transform_Leukemia = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                          torchvision.transforms.FiveCrop(220),
                                                          torchvision.transforms.Lambda(lambda 
                                                          crops: torch.stack([norm_transform(crop)
                                                                              for crop in crops]))])

if args.dataset == 'CIFAR10':
    train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True,
                                                 transform=transform_CIFAR10,
                                                 target_transform=None, download=False)
    test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False,
                                                transform=transform_CIFAR10,
                                                target_transform=None, download=False)

    idx = torch.tensor(train_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)

    idx = torch.tensor(test_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==False)[0])
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(testloader_anomalous, batch_size=1, shuffle=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)

elif args.dataset == 'MNIST':
    train_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=True,
                                             transform=transform_MNIST,
                                             target_transform=None, download=False)
    test_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=False,
                                            transform=transform_MNIST,
                                            target_transform=None, download=False)

    idx = torch.as_tensor(train_MNIST.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, drop_last=True, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)

    idx = torch.as_tensor(test_MNIST.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==False)[0])
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(testloader_anomalous, batch_size=1, shuffle=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)
elif args.dataset == 'Leukemia':
    train_Leukemia = dataset.ImageFolderPaths(args.trainset_path, transform=transform_Leukemia) 
    test_Leukemia = dataset.ImageFolderPaths(args.testset_path, transform=test_transform_Leukemia)
    trainloader = torch.utils.data.DataLoader(train_Leukemia, batch_size=args.batch_size, shuffle=True, num_workers= 0)
    testloader = torch.utils.data.DataLoader(test_Leukemia, batch_size=1, shuffle=True, num_workers=1)


torch.manual_seed(args.random_seed)
dis_learning_rate = args.dis_learning_rate
gen_learning_rate = args.gen_learning_rate
z_learning_rate = args.z_learning_rate
beta1 = 0.5
beta2 = 0.999
num_epochs = args.epochs
latent_size = args.latent_size
distribution = args.distribution

dis_criterion = nn.CrossEntropyLoss()
test_criterion = nn.MSELoss()

if args.dataset == 'CIFAR10':
    gen = model.Generator(latent_size)
    dis = model.Discriminator(latent_size)
elif args.dataset == 'MNIST':
    gen = model.Generator(latent_size)
    dis = model.Discriminator(latent_size)
else:
    gen = model.Generator(latent_size)
    dis = model.Discriminator(latent_size)  


dis.to(device)
gen.to(device)


optimizer_d = optim.Adam(dis.parameters(), 
                         lr=dis_learning_rate, 
                         betas=(beta1, beta2), weight_decay=0.1)
optimizer_g = optim.Adam(gen.parameters(), 
                         lr=gen_learning_rate,
                         betas=(beta1, beta2), weight_decay=0.5)

# scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.1, last_epoch=-1)
# scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.1, last_epoch=-1)


for epoch in range(num_epochs):
    if epoch == 0:
        print('training starts')
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
        dis_fake, _ = dis(x_fake)
        loss_gen = dis_criterion(dis_fake, valid_label)
        loss_gen.backward()
        optimizer_g.step()
    
        # Discriminator
        optimizer_d.zero_grad()
        dis_real, _ = dis(x)
        loss_real = dis_criterion(dis_real, valid_label)
        dis_fake, _ = dis(x_fake.detach())
        loss_fake = dis_criterion(dis_fake, fake_label)
        loss_dis = (loss_fake + loss_real)
        loss_dis.backward()
        optimizer_d.step()

    print("epoch: %d, discriminator loss: %.2f, generator loss: %.2f" 
          %(epoch, loss_dis.item(), loss_gen.item()), flush=True)
        
print('End of training', flush=True)

filepath = str(args.dataset) + str(num_epochs) + 'epochs' + str(distribution) \
           + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

torch.save(gen.state_dict(), './models/' + filepath + 'G.pt')
torch.save(dis.state_dict(), './models/' + filepath + 'D.pt')


def query_noise(query_im, dis, gen, lam):
    query = query_im.cuda()
    noise = torch.randn((query.shape[0], args.latent_size, 1, 1), requires_grad=True, device=device)
    optim_test = optim.Adam([noise], betas=(beta1, beta2), lr=args.z_learning_rate)
    for i in range(args.epochs_z):
        optim_test.zero_grad()
        sample = gen(noise)
        real, internal_real = dis(query)
        fake, internal_fake = dis(sample)
        resloss = test_criterion(sample, query)
        discloss = test_criterion(internal_real, internal_fake)
        loss_test = lam * (discloss) + (1 - lam) * resloss
        loss_test.backward()
        optim_test.step()
    return loss_test, resloss, discloss


score_neg = torch.zeros((len(testloader_normal), 1)).cuda()
score_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()

if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    img_shape = (-1, 1, 32, 32)
elif args.dataset == 'CIFAR10':
    img_shape = (-1, 3, 32, 32)
c_neg = c_pos = 0
score = 0
lam = 0.8

c_neg = c_pos = 0
for step, (images, labels) in enumerate(testloader_normal, 0):
    images = images.view(img_shape).to(device)
    dis.eval()
    gen.eval()
    loss_test, resloss, discloss = query_noise(images, dis, gen, lam)
    
    score_neg[c_neg] = loss_test.detach()
    c_neg += 1

for step, (images, labels) in enumerate(testloader_anomalous, 0):
    images = images.view(img_shape).to(device)
    dis.eval()
    gen.eval()
    loss_test, resloss, discloss = query_noise(images, dis, gen, lam)
    score_pos[c_pos] = loss_test.detach()
    c_pos += 1

print('mean negative: %0.4f, std negative: %0.4f' %(torch.mean(score_neg), torch.std(score_neg)), flush=True)
print('mean positive: %0.4f, std positive: %0.4f' %(torch.mean(score_pos), torch.std(score_pos)), flush=True)

x1 = score_neg.cpu().numpy()
x2 = score_pos.cpu().numpy()
data = {'Normal': x1, 'Anomalous': x2}

FP = TP = []
neg_pre_wrong = 0
for i in range(len(score_neg)):
    if score_neg[i] > torch.mean(score_neg):
        neg_pre_wrong += 1

pos_pre_wrong = 0
for i in range(len(score_pos)):
    if score_pos[i] <= torch.mean(score_neg):
        pos_pre_wrong += 1
tp = (len(score_pos) - pos_pre_wrong)
fn = pos_pre_wrong
fp = neg_pre_wrong
tn = len(score_neg) - neg_pre_wrong
anomalous = torch.ones((len(score_pos), 1))
normal = torch.zeros((len(score_neg), 1))
y = torch.cat((anomalous, normal), 0)
scores = torch.cat((score_pos, score_neg), 0)
fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
auc = metrics.auc(fpr, tpr)
print('AUC', auc, flush=True)
