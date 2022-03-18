import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import ConvChannels, ConvNeighbours,ConvCos

def get_cos(x,k,idx):
    with torch.no_grad():
      q_p = get_graph_feature(x[:,:,0:3], k=k,idx=idx)[:,:,:,0:3]
      mod = q_p.pow(2).sum(dim=3,keepdim=True).sqrt()+1e-8
      q_p = q_p / mod
      q_p = q_p.pow(2)
    return q_p

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature_4input(x, k=20, idx=None):
    x = x.permute(0,2,1)
    batch_size = x.size(0)
    num_points = x.size(2)
    #x = x.contiguous().view(batch_size, -1, num_points)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x[:,0:3,:], k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    #feature = torch.cat((feature[:,:,:,0:3]-x[:,:,:,0:3], x[:,:,:,3:]), dim=3)
    feature = torch.cat((feature[:,:,:,0:3]-x[:,:,:,0:3], x), dim=3)
  
    return feature

def get_graph_feature(x, k=20, idx=None):
    x = x.permute(0,2,1)
    batch_size = x.size(0)
    num_points = x.size(2)
    #x = x.contiguous().view(batch_size, -1, num_points)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3)
  
    return feature

#simple form
def create_ops(C1,C2,K):
    return nn.ModuleList([
      #fcC then max/sum K
      ConvChannels(nn.Sequential(
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.MaxPool2d((1,K)),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      ConvChannels(nn.Sequential(
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.AvgPool2d((1,K)),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      #maxK then fcC
      ConvChannels(nn.Sequential(
                      nn.MaxPool2d((1,K)),
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      #fcK then max/sum C
      ConvNeighbours(nn.Sequential(
                      nn.Conv2d(
                        in_channels=K,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.MaxPool2d((1,C1)),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      ConvNeighbours(nn.Sequential(
                      nn.Conv2d(
                        in_channels=K,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.AvgPool2d((1,C1)),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      #maxC then fcK
      ConvNeighbours(nn.Sequential(
                      nn.MaxPool2d((1,C1)),
                      nn.Conv2d(
                        in_channels=K,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      #fcK then fcC / fcC then fcK
      ConvChannels(nn.Sequential(
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C1,
                        kernel_size=(1,K),
                        groups=C1,bias=False),
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      ConvChannels(nn.Sequential(
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False),
                      nn.Conv2d(
                        in_channels=C2,
                        out_channels=C2,
                        kernel_size=(1,K),
                        groups=C2,bias=False),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                      )),
      #all at once.
      ConvChannels(nn.Sequential(
                      nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=(1,K),bias=False),
                      nn.BatchNorm2d(C2, momentum = 0.9),
                      nn.ReLU(),
                    )),
      #fcC then fc_by_cos
      ConvCos(C1,C2,K),
      ])

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        init_C = 6
        self.channel_list = [64]*2+[128]*1+[256]
        channels = self.channel_list

        self.op_dict = nn.ModuleDict()
        for layer in range(len(channels)):
          C1 = 6+3 if layer == 0 else C2*2
          C2 = channels[layer]
          self.op_dict['ops%d'%layer] = create_ops(C1,C2,self.k)
          self.op_dict['ops%d_W'%layer] = create_ops(init_C,C2,self.k)
          self.op_dict['reduce%d'%layer] = nn.Linear(6 if layer == 0 else channels[layer-1],C2,bias=False)


        self.conv_pre_pool = nn.Sequential(nn.Conv1d(sum(channels), args.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, **kargs):
        ops = kargs['ops']
      
        batch_size = x.size(0)
        x0 = x.permute(0,2,1)

        inputs = []
        inputs.append(x0)

        p = get_graph_feature(x0[:,:,0:3], k=self.k)
        for layer in range(len(self.channel_list)):
          idx = knn(inputs[layer].permute(0,2,1), k = self.k)
          x = get_graph_feature_4input(inputs[layer], k = self.k) if layer == 0 else get_graph_feature(inputs[layer], k = self.k)
          cos = get_cos(x0, self.k, idx) if ops[layer*2] == 9 else None

          _x = self.op_dict['ops%d'%layer][ops[layer*2]]((x,cos)).squeeze(2)
          _W = self.op_dict['ops%d_W'%layer][ops[layer*2+1]//2](p).squeeze(2)
          _x = _x*_W
          if ops[layer*2+1]%2 == 1:
            _x += self.op_dict['reduce%d'%layer](inputs[layer])
          inputs.append(_x)

        x = torch.cat(tuple(inputs[1:]), dim=2).permute(0,2,1)
        x = self.conv_pre_pool(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
