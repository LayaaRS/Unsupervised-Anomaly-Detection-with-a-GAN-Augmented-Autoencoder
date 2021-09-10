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
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-bs', '--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('-ep', '--num_epochs', type=int, default=25, help='number of epochs')
parser.add_argument('-z', '--latent_size', type=int, default=100, help='latent size')
parser.add_argument('-d', '--dropout', type=float, default=0.3, help='daropout')
parser.add_argument('-ef', '--encoder_first_train', type=int, default=0, help='number of epochs encoder first train')
parser.add_argument('-ds', '--dataset', default='Leukemia', help='which dataset the model is running on')
parser.add_argument('-sm', '--save_model_dir', default='./models/')
parser.add_argument('-cl', '--class_label', type=int, default=0, help='select which class label should be selected as the anomalous')
parser.add_argument('-lm', '--lambda_val', default=0.1, type=float, help='lambda of AS')
parser.add_argument('-bt', '--beta_val', default=1, type=float, help='beta of AS')
parser.add_argument('-dz', '--distribution', default='Gaussian', help='choose the distribution you want to z selected from')

args = parser.parse_args()
# args.save_image_dir = args.save_model_dir + 'images'


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
    trainloader = torch.utils.data.DataLoader(train_Leukemia, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_Leukemia, batch_size=1, shuffle=True, num_workers=1)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

beta1 = 0.9
beta2 = 0.999

dis_criterion = nn.CrossEntropyLoss()
aen_criterion = nn.MSELoss()  

if args.dataset == 'CIFAR10':
    enc = model.Encoder(args.latent_size, args.dropout)
    gen = model.Generator(args.latent_size)
    dis = model.Discriminator(args.latent_size, args.dropout)
elif args.dataset == 'MNIST':
    enc = model.Encoder(args.latent_size, args.dropout)
    gen = model.Generator(args.latent_size)
    dis = model.Discriminator(args.latent_size, args.dropout)
elif args.dataset == 'Leukemia':
    enc = model.Encoder(args.latent_size)
    gen = model.Generator(args.latent_size)
    dis = model.Discriminator(args.latent_size)  


dis.to(device)
enc.to(device)
gen.to(device)


optimizer_d = optim.Adam(dis.parameters(), 
                         lr=args.learning_rate, 
                         betas=(beta1, beta2), weight_decay=0.5)
optimizer_g = optim.Adam(gen.parameters(), 
                         lr=args.learning_rate,
                         betas=(beta1, beta2), weight_decay=0.5)
optimizer_e = optim.Adam(enc.parameters(),
                         lr=args.learning_rate,
                         betas=(beta1, beta2), weight_decay=1e-3)

gen_train = False
print('training starts')
for epoch in range(args.num_epochs):
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

            if args.distribution == 'Gaussian':
                z_random = torch.randn(x.shape[0], args.latent_size, 1, 1).to(device)
            elif args.distribution == 'Uniform':
                z_random = (torch.rand(x.shape[0], args.latent_size, 1, 1) * 2 - 1).to(device)
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

            # if epoch % 10 == 0 and epoch >= args.encoder_first_train:
            #   vutils.save_image(x_fake.cpu().data[:16, ], '%s/fake_%d.png' % (args.save_image_dir, epoch))

    if gen_train == True:
        print("epoch: %d, discriminator loss: %.2f, generator loss: %.2f, encoder loss: %.2f" 
              %(epoch, loss_dis.item(), loss_gen.item(), loss_aen.item()), flush=True)
    
print('End of training', flush=True)

# Final model save
filepath = str(args.save_model_dir) + str(args.dataset) + str(args.num_epochs) + 'epochs' + \
               datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
torch.save(gen.state_dict(), filepath + 'G')
torch.save(enc.state_dict(), filepath + 'E')
torch.save(dis.state_dict(), filepath + 'D')


lam = args.lambda_val
betha = args.beta_val
loss_neg = torch.zeros((len(testloader_normal), 1)).cuda()
loss_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()

if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    img_shape = (-1, 1, 28, 28)
elif args.dataset == 'CIFAR10':
    img_shape = (-1, 3, 32, 32)
c_neg = c_pos = 0

for step, (images, labels) in enumerate(testloader_normal, 0):
    images = images.view(img_shape)
    dis.eval()
    gen.eval()
    enc.eval()
    x_real_test = images.cuda()
    E_x = enc(x_real_test)
    G_z = gen(E_x)
    
    E_G_z = enc(G_z)
    
    real, internal_real = dis(x_real_test)
    fake, internal_fake = dis(G_z)
    
    latentloss = torch.mean(torch.abs(E_x - E_G_z))
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(internal_real-internal_fake))
    loss_test = (1 - lam) * resloss + lam * discloss + betha * latentloss
    loss_neg[c_neg] = loss_test.detach()
    c_neg += 1
    
for step, (images, labels) in enumerate(testloader_anomalous, 0):
    images = images.view(img_shape)
    dis.eval()
    gen.eval()
    enc.eval()
    x_real_test = images.cuda()
    E_x = enc(x_real_test)
    G_z = gen(E_x)
    
    E_G_z = enc(G_z)
    
    real, internal_real = dis(x_real_test)
    fake, internal_fake = dis(G_z)
    
    latentloss = torch.mean(torch.abs(E_x - E_G_z))
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(internal_real-internal_fake))
    loss_test = (1 - lam) * resloss + lam * discloss + betha * latentloss
    loss_pos[c_pos] = loss_test.detach()
    c_pos += 1

print('mean negative: %0.4f, std negative: %0.4f' 
      %(torch.mean(loss_neg), torch.std(loss_neg)), flush=True)
print('mean positive: %0.4f, std positive: %0.4f' 
      %(torch.mean(loss_pos), torch.std(loss_pos)), flush=True)

x1 = loss_neg.cpu().numpy()
x2 = loss_pos.cpu().numpy()

FP = TP = []
neg_pre_wrong = 0
for i in range(len(loss_neg)):
    if loss_neg[i] > torch.mean(loss_neg):
        neg_pre_wrong += 1

pos_pre_wrong = 0
for i in range(len(loss_pos)):
    if loss_pos[i] <= torch.mean(loss_neg):
        pos_pre_wrong += 1
tp = (len(loss_pos) - pos_pre_wrong)
fn = pos_pre_wrong
fp = neg_pre_wrong
tn = len(loss_neg) - neg_pre_wrong
precision = tp / (tp + fp)
# recall / sensitivity / True Positive Rate
recall = tp / (tp + fn)
# False Positive Rate / 1 - Specificity
fp_rate = fp / (fp + tn)
specificity = tn / (tn + fp)
f1 = 2 * ((precision * recall)/(precision + recall))
accuracy = (tp + tn) / (tp + tn + fp + fn)
anomalous = torch.ones((len(loss_pos), 1))
normal = torch.zeros((len(loss_neg), 1))
y = torch.cat((anomalous, normal), 0)
scores = torch.cat((loss_pos, loss_neg), 0)
fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
prec, rec, _ = metrics.precision_recall_curve(y.cpu(), scores.cpu())
auc = metrics.auc(fpr, tpr)
print('AUC', auc, flush=True)
