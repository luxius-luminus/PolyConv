from __future__ import print_function
import os
import time
import argparse

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import ModelNet40
from model import DGCNN
from util import cal_loss, IOStream,rotate_point_cloud,jitter_point_cloud

import sample_arch

# Training settings
parser = argparse.ArgumentParser(description='PolyConvNet Traning For Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn',
                    choices=['dgcnn'],
                    help='Model to use, [dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40',
                    choices=['modelnet40'])
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch for training')
parser.add_argument('--test_batch_size', type=int, default=16, 
                    help='Size of batch for testing')
parser.add_argument('--epochs', type=int, default=250, 
                    help='number of epochs to train ')
parser.add_argument('--use_adam', action="store_true",
                    help='Use Adam or not(SGD instead)')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='learning rate (default: 0.001; 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, 
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, 
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='', 
                    help='Pretrained model path')
parser.add_argument('--rotate', type=bool, default=False,
                    help='Pretrained model path')
parser.add_argument('--arch', type=str, default='only_0_res',
                    help='PolyConvNet architecture to adopt')
parser.add_argument('--lrschdl', type=str, default='cos',
                    help='learning rate schedule, currently using cosine annealing')
args = parser.parse_args()

cur_time = time.gmtime(time.time())
args.exp_name = str(args.arch) + '_' + args.exp_name + '_' + "%02d%02d_%02d%02d%02d"%(cur_time.tm_mon,cur_time.tm_mday,cur_time.tm_hour,cur_time.tm_min,cur_time.tm_sec)
if args.rotate:
  args.exp_name += '_r'

arch = eval("sample_arch.%s" % (args.arch))

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp *.py checkpoints'+'/'+args.exp_name+'/')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = DGCNN(args).to(device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print('Training architecture %s ...'%arch)
    args.use_sgd = not args.use_adam
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.lrschdl == 'cos':
      eta_min = args.lr if args.use_sgd else args.lr/100
      scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=eta_min)
    else:
        raise Exception("Not implemented")

    
    criterion = cal_loss
    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            if args.rotate:
              rotated_data = rotate_point_cloud(data.numpy())
              data = torch.from_numpy(jitter_point_cloud(rotated_data)).type_as(data)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data,ops=arch)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
          for data, label in test_loader:
              if args.rotate:
                rotated_data = rotate_point_cloud(data.numpy())
                data = torch.from_numpy(jitter_point_cloud(rotated_data)).type_as(data)
              data, label = data.to(device), label.to(device).squeeze()
              data = data.permute(0, 2, 1)
              batch_size = data.size()[0]
              logits = model(data,ops=arch)
              loss = criterion(logits, label)
              preds = logits.max(dim=1)[1]
              count += batch_size
              test_loss += loss.item() * batch_size
              test_true.append(label.cpu().numpy())
              test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    with torch.no_grad():
      for data, label in test_loader:
          if args.rotate:
            rotated_data = rotate_point_cloud(data.numpy())
            data = torch.from_numpy(jitter_point_cloud(rotated_data)).type_as(data)

          data, label = data.to(device), label.to(device).squeeze()
          data = data.permute(0, 2, 1)
          batch_size = data.size()[0]
          logits = model(data,ops=arch)
          preds = logits.max(dim=1)[1]
          test_true.append(label.cpu().numpy())
          test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
