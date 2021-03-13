import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18, resnet152

import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

def build_dataloader():
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = torchvision.datasets.CIFAR10(root='./dummy/cifar10', train=True, download=True, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, sampler=train_sampler, num_workers=10, pin_memory=False)

    test_dataset = torchvision.datasets.CIFAR10(root='./dummy/cifar10', train=False, download=True, transform=test_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, sampler=test_sampler, num_workers=10, pin_memory=False)
    return train_loader, train_sampler, test_loader, test_sampler

def main_worker(gpu, ngpus_per_node, main, config):
    cudnn.benchmark = True
    print(f'Use GPU: {gpu}')
    config['gpu'] = gpu
    dist.init_process_group(backend='nccl', init_method='tcp://9.0.199.130:23457', world_size=ngpus_per_node, rank=gpu)
    main(config)

def spawn_workers(main, config):
    ngpus_per_node = 4
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, config))

def deploy_model(config):
    model = resnet152()
    model.fc = nn.Linear(model.fc.in_features, 10)
    torch.cuda.set_device(config['gpu'])
    model.cuda(config['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']], find_unused_parameters=True)
    return model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(epoch, model, train_loader, criterion, optimizer, config):
    for i, data in enumerate(train_loader):
        batch = data[0].cuda()
        label = data[1].cuda()
        output = model(batch)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.item(), batch.shape[0])
        top1.update(prec1.item(), batch.shape[0])
        top5.update(prec5.item(), batch.shape[0])

        if config['gpu'] == 1:
            print('-' * 20 + 'Train' + '-' * 20)
            print(f'Epoch: [{epoch}][{i+1}/{len(train_loader)}]')
            print(f'Loss: {losses.avg:.5f}')
            print(f'Top1: {top1.avg:.2f}%, Top5: {top5.avg:.2f}%')

def test(epoch, model, test_loader, criterion, config):
    loss_lst, top1_lst, top5_lst = [], [], []

    for i, data in enumerate(test_loader):
        batch = data[0].cuda()
        label = data[1].cuda()
        output = model(batch)
        loss = criterion(output, label)

        loss.backward()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.item(), batch.shape[0])
        top1.update(prec1.item(), batch.shape[0])
        top5.update(prec5.item(), batch.shape[0])


        if config['gpu'] == 0:
            print('-' * 20 + 'Test' + '-' * 20)
            print(f'Epoch: [{epoch}][{i+1}/{len(test_loader)}]')
            print(f'Loss: {losses.avg:.5f}')
            print(f'Top1: {top1.avg:.2f}%, Top5: {top5.avg:.2f}%')

        
        with open(f"./dummy/result/{config['gpu']}.txt", 'w') as f:
            f.write(f'{losses.avg} {top1.avg} {top5.avg}\n')
    
    if config['gpu'] == 0:
        for i in range(4):
            data = open(f"./dummy/result/{config['gpu']}.txt", 'r').readline().strip().split(" ")
            data = [float(x) for x in data]
            loss_lst.append(data[0])
            top1_lst.append(data[1])
            top5_lst.append(data[2])

        print('-' * 20 + 'Global Result' + '-' * 20)
        print(f'Epoch: [{epoch}][{len(test_loader)}/{len(test_loader)}]')
        print(f'Loss: {np.mean(loss_lst):.5f}')
        print(f'Top1: {np.mean(top1_lst):.2f}%, Top5: {np.mean(top5_lst):.2f}%')
        

def main(config):
    model = deploy_model(config)
    num_epoch = config['epoch']
    lr = config['lr']

    train_loader, train_sampler, test_loader, test_sampler = build_dataloader()
    train_sampler.set_epoch(config['epoch'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for epoch in range(num_epoch):
        train(epoch + 1, model, train_loader, criterion, optimizer, config)
        test(epoch + 1, model, test_loader, criterion, config)

if __name__ == '__main__':
    config = {'epoch': 1000000000000, 'lr': 0.01, 'gpu': -1}
    spawn_workers(main, config)