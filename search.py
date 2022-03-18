import os
import sys
import time
import glob
import logging
import argparse

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import util_search as utils
from data import ModelNet40
from model import DGCNN
from util import cal_loss, IOStream,rotate_point_cloud,jitter_point_cloud
from mcts import *

parser = argparse.ArgumentParser(description='Searching PolyConv via MCTS on Modelnet40')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, 
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=200, 
                    help='number of episode to search ')
parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, 
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, 
                    help='Num of nearest neighbors to use')
parser.add_argument('--rotate', type=bool, default=False,
                    help='Pretrained model path')
parser.add_argument('--arch', type=str, default=0,
                    help='PolyConvNet architecture to adopt')
parser.add_argument('--save', type=str, default='EXP', 
                    help='experiment name')
parser.add_argument('--select_c', type=float, default=0.6,
                    help='hyperparameter of the UCT formula in MCTS')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.save == 'EXP':
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
os.mkdir(os.path.join(args.save,'alphas'))
os.mkdir(os.path.join(args.save,'weights'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


tree_nodes = dict()

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = cal_loss
  device = torch.device("cuda" if args.cuda else "cpu")
  model = DGCNN(args).to(device)
  model = nn.DataParallel(model)

  if args.use_sgd:
      print("Use SGD")
      optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
  else:
      print("Use Adam")
      optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

  train_queue = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
  valid_queue = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                           batch_size=args.test_batch_size, shuffle=True, drop_last=False)


  scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)

  #global tree_nodes
  root = Node(arch=(-1,-1))
  root.set_num(len(tree_nodes))
  tree_nodes[root.num]=root
  archs_list = []

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, 
    criterion, optimizer, lr, epoch,root,archs_list,args.select_c)

    # validation
    leaf_node, trail = Sample(root,tree_nodes,c=0.0)
    arch = [x[0] for x in trail]
    valid_acc, valid_obj = infer(valid_queue, model, criterion,arch=arch,epoch=epoch)
    logging.info('valid_acc %f', valid_acc)

    torch.save({'root':root,
                'tree_nodes':tree_nodes},
                os.path.join(args.save,'alphas',f'tree_at_epoch{epoch}.pt'))
    utils.save(model, os.path.join(args.save, 'weights',
                    f'weigths_epoch{epoch}.pt'))


def train(train_queue, valid_queue, model, criterion,
optimizer,lr,epoch,root,archs_list,select_c):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  max_tree_depth = 20
  cur_node = root

  count = 0.0
  train_pred = []
  train_true = []
  model.train()
  params = model.parameters()
  device = torch.device("cuda" if args.cuda else "cpu")
  for step, (input, target) in enumerate(train_queue):
    if args.rotate:
      rotated_data = rotate_point_cloud(input.numpy())
      input = torch.from_numpy(jitter_point_cloud(rotated_data)).type_as(input)
    input, target = input.to(device), target.to(device).squeeze()
    input = input.permute(0, 2, 1)
    batch_size = input.size()[0]
    cur_node = root

    n = input.size(0)

    leaf_node, trail = Selection(cur_node,tree_nodes,select_c)
    optimizer.zero_grad()
    arch = [x[0] for x in trail]
    logits = model(input,ops=arch)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    preds = logits.max(dim=1)[1]
    count += batch_size
    train_true.append(target.cpu().numpy())
    train_pred.append(preds.detach().cpu().numpy())

    acc = float(prec1)/100.0
    BackProp(leaf_node,tree_nodes,acc,1.0)

    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

  train_true = np.concatenate(train_true)
  train_pred = np.concatenate(train_pred)

  outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                           objs.avg,
                                                                           metrics.accuracy_score(
                                                                               train_true, train_pred),
                                                                           metrics.balanced_accuracy_score(
                                                                               train_true, train_pred))
  logging.info(outstr)
  return top1.avg, objs.avg

def infer(valid_queue, model, criterion, **kargs):
  arch = kargs['arch']
  epoch = kargs['epoch']
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  test_loss = 0.0
  count = 0.0
  model.eval()
  test_pred = []
  test_true = []

  device = torch.device("cuda" if args.cuda else "cpu")
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if args.rotate:
        rotated_data = rotate_point_cloud(input.numpy())
        input = torch.from_numpy(jitter_point_cloud(rotated_data)).type_as(input)
      input, target = input.to(device), target.to(device).squeeze()
      input = input.permute(0, 2, 1)
      batch_size = input.size()[0]

      logits = model(input,ops=arch)

      loss = criterion(logits, target)
      preds = logits.max(dim=1)[1]

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)
      test_true.append(target.cpu().numpy())
      test_pred.append(preds.detach().cpu().numpy())

  test_true = np.concatenate(test_true)
  test_pred = np.concatenate(test_pred)
  test_acc = metrics.accuracy_score(test_true, test_pred)
  avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
  outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                        objs.avg,
                                                                        test_acc,
                                                                        avg_per_class_acc)

  logging.info(outstr)
  return top1.avg, objs.avg

if __name__ == '__main__':
  main() 
