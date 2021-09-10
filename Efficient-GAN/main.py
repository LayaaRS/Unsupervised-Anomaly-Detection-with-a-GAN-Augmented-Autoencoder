import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.utils as vutils

import os
import argparse
import datetime
import numpy as np
from sklearn import metrics

from utils import dataset
from modules import model


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', default='Leukemia', help='The datset you want the model to be trained on')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-rs', '--random_seed', default=0, type=int, help='random seed')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-ep', '--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-z', '--latent_size', type=int, default=15, help='latent size')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='daropout')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('-sm', '--save_model_dir', default='results/')
parser.add_argument('-lm', '--lambda_val', default=0.8, type=float, help='lambda of AS')
parser.add_argument('-dz', '--distribution', default='Gaussian', help='choose the distribution you want to z selected from')
parser.add_argument('-cl', '--class_label_anomalous', default=0, type=int, help='select which class label should be selected as the anomalous')


args = parser.parse_args()

torch.manual_seed(args.random_seed)
batch_size = args.batch_size
num_epochs = args.epochs
cuda_device = "0"

# args.save_image_dir = args.save_model_dir + 'images'

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def tocuda(x):
    if args.use_cuda:
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

if args.dataset == 'MNIST':
    train_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=True, transform=transform_MNIST, target_transform=None, download=False)
    test_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=False, transform=transform_MNIST, target_transform=None, download=False)
    idx = torch.as_tensor(train_MNIST.targets) == torch.tensor(args.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==False)[0])
    idx = torch.as_tensor(test_MNIST.targets) == torch.tensor(args.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, drop_last=True, shuffle=True,)
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    print('MNIST train: ', len(trainloader), 'MNIST test normal:', len(testloader_normal), 'MNIST test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'CIFAR10':
    train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=transform_CIFAR10, target_transform=None, download=False)
    test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False, transform=transform_CIFAR10, target_transform=None, download=False)
    idx = torch.tensor(train_CIFAR10.targets) == torch.tensor(args.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==False)[0])
    idx = torch.tensor(test_CIFAR10.targets) == torch.tensor(args.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    print('CIFAR10 train: ', len(trainloader), 'CIFAR10 test normal: ', len(testloader_normal), 'CIFAR10 test anomalous: ', len(testloader_anomalous))
elif args.dataset == 'Leukemia':
    train_Leukemia = dataset.ImageFolderPaths('./data/Leukemia/train/', transform=transform_Leukemia) 
    test_Leukemia = dataset.ImageFolderPaths('./data/Leukemia/test/', transform=test_transform_Leukemia)
    trainloader = torch.utils.data.DataLoader(train_Leukemia, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_Leukemia, batch_size=1, shuffle=True, num_workers=1)

if args.dataset == 'MNIST':
    # for 1*28*28 images of MNIST dataset  
    enc = tocuda(model.Encoder_MNIST(args.latent_size, True))
    gen = tocuda(model.Generator_MNIST(args.latent_size))
    dis = tocuda(model.Discriminator_MNIST(args.latent_size, 0.2, 1))
elif args.dataset == 'CIFAR10':
    # for 3*32*32 images of CIFAR10 dataset
    enc = tocuda(model.Encoder_CIFAR10(args.latent_size, True))
    gen = tocuda(model.Generator_CIFAR10(args.latent_size))
    dis = tocuda(model.Discriminator_CIFAR10(args.latent_size, 0.2, 1))
else:
    # for 3*220*220 images of Leukemia dataset
    enc = tocuda(model.Encoder_Leukemia(args.latent_size, True))
    gen = tocuda(model.Generator_Leukemia(args.latent_size))
    dis = tocuda(model.Discriminator_Leukemia(args.latent_size, 0.2, 1))


enc.apply(weights_init)
gen.apply(weights_init)
dis.apply(weights_init)

optimizerG = optim.Adam([{'params': enc.parameters()},
                         {'params': gen.parameters()}], lr=args.learning_rate, betas=(0.5, 0.999))
optimizerD = optim.Adam(dis.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(args.epochs):
    i = 0
    for (data, target) in trainloader:

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (args.epochs - epoch) / args.epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (args.epochs - epoch) / args.epochs)))

        if epoch == 0 and i == 0:
            gen.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, args.latent_size, 1, 1)))
        d_fake = gen(z_fake)

        z_real, _, _, _ = enc(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :args.latent_size], z_real[:, args.latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, args.latent_size)))

        output_z = mu + epsilon * sigma

        output_real, _ = dis(d_real + noise1, output_z.view(batch_size, args.latent_size, 1, 1))
        output_fake, _ = dis(d_fake + noise2, z_fake)

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
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), 
                  "G loss :", loss_g.item(), "D(x) :", output_real.mean().item(),
                  "D(G(x)) :", output_fake.mean().item())

        # if i % 50 == 0:
        #     vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (args.save_image_dir))
        #     vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (args.save_image_dir))

        i += 1

    if epoch % 100 == 0:
        torch.save(gen.state_dict(), './%s/netG_epoch_%d.pth' % (args.save_model_dir, epoch))
        torch.save(enc.state_dict(), './%s/netE_epoch_%d.pth' % (args.save_model_dir, epoch))
        torch.save(dis.state_dict(), './%s/netD_epoch_%d.pth' % (args.save_model_dir, epoch))
    if epoch > 30:
        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (args.save_image_dir, epoch))

filepath = str(args.dataset) + str(num_epochs) + 'epochs' + str(args.distribution) + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

torch.save(gen.state_dict(), args.save_model_dir + filepath + 'G.pt')
torch.save(enc.state_dict(), args.save_model_dir + filepath + 'E.pt')
torch.save(dis.state_dict(), args.save_model_dir + filepath + 'D.pt')

score_neg = torch.zeros((len(testloader_normal), 1)).cuda()
score_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()

if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    img_shape = (-1, 1, 32, 32)
elif args.dataset == 'CIFAR10':
    img_shape = (-1, 3, 32, 32)
c_neg = c_pos = 0
score = 0

c_neg = c_pos = 0
for step, (images, labels) in enumerate(testloader_normal, 0):
    images = images.view(img_shape).cuda()
    dis.eval()
    gen.eval()
    enc.eval()
    x_real_test = images.cuda()
    E_x, _, _, _ = enc(x_real_test)
    G_z = gen(E_x[:, :100]).detach()
    real, internal_real = dis(x_real_test, E_x[:, :100])
    fake, internal_fake = dis(G_z, E_x[:, :100])
    
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(internal_real - internal_fake))
    loss_test = ((1 - args.lambda_val) * resloss + args.lambda_val * discloss)
    score_neg[c_neg] = loss_test.detach()
    c_neg += 1

for step, (images, labels) in enumerate(testloader_anomalous, 0):
    images = images.view(img_shape).cuda()
    dis.eval()
    gen.eval()
    enc.eval()
    x_real_test = images.cuda()
    E_x, _, _, _ = enc(x_real_test)
    G_z = gen(E_x[:, :100]).detach()
    real, internal_real = dis(x_real_test, E_x[:, :100])
    fake, internal_fake = dis(G_z, E_x[:, :100])
    
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(internal_real - internal_fake))
    loss_test = ((1 - args.lambda_val) * resloss + args.lambda_val * discloss)
    score_pos[c_pos] = loss_test.detach()
    c_pos += 1

print('mean negative: %0.4f, std negative: %0.4f' 
      %(torch.mean(score_neg), torch.std(score_neg)), flush=True)
print('mean positive: %0.4f, std positive: %0.4f' 
      %(torch.mean(score_pos), torch.std(score_pos)), flush=True)

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
