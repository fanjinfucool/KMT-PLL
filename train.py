import os
import os.path
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np
import time

from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from cifar_models import convnet, resnet
from datasets.mnist import mnist
from datasets.fashion import fashion
from datasets.kmnist import kmnist
from datasets.cifar10 import cifar10
from datasets.fmnist import fmnist
from utils.utils_k import *
from kvit import k_max_vit
import torch,gc

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(0);
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
    prog='KMT-PLL demo file.',
    usage='Demo with partial labels.',
    description='A simple demo file with MNIST dataset.',
    epilog='end',
    add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('-bs', help='batch size', type=int, default=64)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'kmnist', 'cifar10'], required=False)
parser.add_argument('-model', help='model name', type=str, default='mlp',
                    choices=['linear', 'mlp', 'convnet', 'resnet'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial',
                    choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)

parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
parser.add_argument('-alpha', help='alpha', type=float, default=1e-4)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load dataset
if args.ds == 'mnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = mnist(root='./mnist/',
                          download=True,
                          train_or_not=True,
                          transform=transforms.Compose(
                              [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                          partial_type=args.partial_type,
                          partial_rate=args.partial_rate
                          )
    test_dataset = mnist(root='./mnist/',
                         download=True,
                         train_or_not=False,
                         transform=transforms.Compose(
                             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                         partial_type=args.partial_type,
                         partial_rate=args.partial_rate
                         )

if args.ds == 'fmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = fmnist(root='./FashionMNIST/',
                           download=True,
                           train_or_not=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate
                           )
    test_dataset = fmnist(root='./FashionMNIST/',
                          download=True,
                          train_or_not=False,
                          transform=transforms.Compose(
                              [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                          partial_type=args.partial_type,
                          partial_rate=args.partial_rate
                          )

if args.ds == 'fashion':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = fashion(root='./fashion/',
                            download=True,
                            train_or_not=True,
                            transform=transforms.Compose(
                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate
                            )
    test_dataset = fashion(root='./fashionmnist/',
                           download=True,
                           train_or_not=False,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate
                           )

if args.ds == 'kmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = kmnist(root='./kmnist/',
                           download=True,
                           train_or_not=True,
                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate
                           )
    test_dataset = kmnist(root='./kmnist/',
                          download=True,
                          train_or_not=False,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                          partial_type=args.partial_type,
                          partial_rate=args.partial_rate
                          )

if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar10(root='./cifar10/',
                            download=True,
                            train_or_not=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                               (0.2023, 0.1994, 0.2010)), ]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate
                            )
    test_dataset = cifar10(root='./cifar10/',
                           download=True,
                           train_or_not=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                              (0.2023, 0.1994, 0.2010)), ]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate
                           )

# learning rate
lr_plan = [args.lr] * args.ep
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir,
                         ('KmaxVIT' + args.ds + '_' + args.partial_type + '_' + str(args.partial_rate) + '.txt'))


# calculate accuracy


def evaluate(loader, model, am_center):
    model.eval()
    correct = 0
    total = 0
    for data, _, labels, _ in loader:
        images = data.to(device).float()
        labels = labels.to(device).float()
        output1,_ = model(images, am_center)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc

def binarize_class(y):
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)

    return label


def main():
    # print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.nw,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)


    """Compute cluster center"""
    center = cluster_center(np.array(train_loader.dataset.train_data).reshape(-1, 28*28), np.array(train_loader.dataset.train_final_labels), axis=1)




    print('building model...')

    net = k_max_vit(num_features, num_classes)
    net.to(device)
    print(net.parameters)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    time_start = time.time()

    am_center = torch.tensor(center).float().to(device)

    best_test_acc = 0
    print('training...')

    for epoch in range(0, args.ep):
        net.train()
        adjust_learning_rate(optimizer, epoch)

        for i, (data, labels, trues, indexes) in enumerate(train_loader):
            images = Variable(data).to(device).float()
            labels = Variable(labels).to(device).float()
            trues = trues.to(device).float()
            output, am_center1 = net(images, am_center)

            am_center = am_center - args.alpha * (am_center - am_center1)

            # dis_label = center_sim(data, am_center)
            dis_label = disCosine(data, am_center)

            loss, new_label = partial_loss(output, labels, trues, dis_label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # update weights

            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()

            del images, labels, trues, output, am_center1, loss, new_label, dis_label
            gc.collect()
            torch.cuda.empty_cache()

            # am_center = updat_center(data, new_label, am_center, 0.01)

        print('evaluating model...')
        train_acc = evaluate(train_loader, net, am_center)
        test_start = time.time()
        test_acc = evaluate(test_loader, net, am_center)
        test_end = time.time()

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(
            ' TRAIN: epoch = {:.4f}\t : Training Acc.: = {:.4f}\t, Test Acc.: = {:.4f}\t, Test Time.: = {:.4f}\n'.format(
                epoch, train_acc, test_acc, test_end - test_start))

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(
                round(test_acc, 4)) + '\n')

    time_end = time.time()
    print('totally cost', time_end - time_start)
    print('Beat test acc.:= {:.4f}\t'.format(best_test_acc))






if __name__ == '__main__':
    main()
