import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import argparse
from model import Resnet

from tqdm import tqdm
import time
import sys

class Trainer:
    def __init__(self, num_workers = 2,cuda = False,opt = 'sgd',epochs = 5):
        self.num_workers = num_workers
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if cuda else 'cpu'
        self.model = Resnet().to(self.device)
        print('Resnet18  model summary\n\n')
        summary(self.model,(3,32,32))
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.opt = opt

        if opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
        elif opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.1,
                                weight_decay=5e-4)
        elif opt == 'sgdn':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4, nesterov= True)
        elif opt == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.1,
                                weight_decay=5e-4)
        elif opt == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.1,
                                weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    
    def load_data(self):
        print('\n\nLoading Dataset\n\n')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=self.num_workers)
        
    def train(self):
        print('\n\nTraining\n\n')
        print('No. of workers:',self.num_workers)
        print('Cuda:',self.device)
        print('Optimizer:',self.opt)

        # sys.stderr.write('No. of workers: {}\n'.format(self.num_workers))
        # sys.stderr.write('Cuda: {}\n'.format(self.device))
        # sys.stderr.write('Optimizer: {}\n'.format(self.opt))

        self.model.train()
        start = time.perf_counter()
        data_total = 0
        train_total = 0
        for ep in range(self.epochs):
            # with tqdm(enumerate(self.trainloader),total=len(self.trainloader),desc = 'Epoch {}/{}'.format(ep+1,self.epochs),position = 0,dynamic_ncols=True) as tep:
            train_loss = 0
            correct = 0
            total = 0

            data_time = 0
            train_time = 0
            other_time = 0
            total_time = 0

            total_start = time.perf_counter()
            # data_start = time.perf_counter()

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # data_time_ = time.perf_counter()-data_start
                # data_time += data_time_

                train_start = time.perf_counter()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_time += time.perf_counter()-train_start
                # train_time += train_time_

                temp_start = time.perf_counter()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                other_time += time.perf_counter()-temp_start

                # tep.set_postfix({'loss': train_loss/(batch_idx+1), 'acc': 100.*correct/total, 'data': data_time_, 'train': train_time_})
                # tep.set_postfix_str('loss={:.4f}, acc={:.2f}, data-time={:.4f} s, train-time={:.4f} s'.format(train_loss/(batch_idx+1),100.*correct/total,data_time_,train_time_))
                # data_start = time.perf_counter()
                
            total_time = time.perf_counter() - total_start

            data_time = total_time - train_time - other_time

            data_total += data_time
            train_total += train_time

            print('Epoch {}/{}: loss={:.4f} | acc={:.2f} || Total time -> epoch: {:.04f} s | data: {:.04f} s | train: {:.04f} s\n'.format(ep+1,self.epochs,train_loss/(batch_idx+1),100.*correct/total,total_time,data_time,train_time))
            # sys.stderr.write('Epoch {}/{}: loss={:.4f} | acc={:.2f} || Total time -> epoch: {:.04f} s | data: {:.04f} s | train: {:.04f} s\n'.format(ep+1,self.epochs,train_loss/(batch_idx+1),100.*correct/total,total_time,data_time,train_time))
        total_time = time.perf_counter()-start
        print('After {} epochs: loss={:.4f} | acc={:.2f} || Total time {:.04f} s | Avg Time -> epoch: {:.04f} s | data: {:.04f} s | train: {:.04f} s'.format(self.epochs,train_loss/(batch_idx+1),100.*correct/total,total_time,total_time/self.epochs,data_total/self.epochs,train_total/self.epochs))
        # sys.stderr.write('After {} epochs: loss={:.4f} | acc={:.2f} || Total time {:.04f} s | Avg Time -> epoch: {:.04f} s | data: {:.04f} s | train: {:.04f} s'.format(self.epochs,train_loss/(batch_idx+1),100.*correct/total,total_time,total_time/self.epochs,data_total/self.epochs,train_total/self.epochs))    


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--workers', default=0, type=int, help='no. of workers')
parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
parser.add_argument('--cuda', action="store_true", help='Cuda device')
parser.add_argument('--epochs', default=5, type=int, help='no. of epochs')
args = parser.parse_args()

trainer = Trainer(num_workers=args.workers,cuda=args.cuda,opt=args.opt.lower(),epochs=args.epochs)
trainer.load_data()
trainer.train()
