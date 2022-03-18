#from __future__ import print_function
import os
import sys
import numpy as np
import torch
import argparse

from mcts import DeriveSelection as Selection

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--epoch', type=int, default=200, help='')
parser.add_argument('--exp', type=str, default='exactly_as_enas_1', help='')
parser.add_argument('--c', type=float, default=0.5, help='')
parser.add_argument('--max_sample', action='store_true', default=False,
                      help='determinsitc sampling')
parser.add_argument('--print', action='store_true', default=False, help='print\
                      statistics of the nodes')
args = parser.parse_args()


exp=args.exp
epoch=args.epoch-1

path='%s/alphas/tree_at_epoch%d.pt'%(exp,epoch)
print('#',path)
chp=torch.load(path)
tree_nodes=chp['tree_nodes']
root=chp['root']
name = exp.split('/')[-1].replace('.','dot')

if args.max_sample:
  node,trail=Selection(root,tree_nodes,c=0.0,max_sample=args.max_sample,verbose=args.print)
  tokens = '%4d,'*len(trail)%tuple([item[1].arch[1] for item in trail])
  print(f'{name:s}_epoch{epoch}_sample_max=[{tokens:s}]')
else:
  for i in range(10):
    node,trail=Selection(root,tree_nodes,c=args.c,max_sample=args.max_sample,verbose=args.print)
    tokens = '%4d,'*len(trail)%tuple([item[1].arch[1] for item in trail])
    print(f'{name:s}_epoch{epoch}_sample{i}=[{tokens:s}]')
