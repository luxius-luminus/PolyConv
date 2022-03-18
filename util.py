import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from typing import Callable, Union, Tuple

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        R = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def ConvCos(C1,C2,K, make_contiguous = False):
    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.C2 = C2
            self.conv1 =  nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False)
            self.conv2 =  nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False)
            self.conv3 =  nn.Conv2d(
                        in_channels=C1,
                        out_channels=C2,
                        kernel_size=1,bias=False)
            self.pool = nn.AvgPool2d((1,K))
            self.bn = nn.BatchNorm2d(C2, momentum = 0.9)
            self.relu = nn.ReLU()

        def forward(self,*args):
            x = args[0][0]
            cos = args[0][1]
            x = x.permute(0,3,1,2)
            x = torch.stack((self.conv1(x),self.conv2(x),self.conv3(x)),dim=4)
            #x = (x* torch.from_numpy(cos.unsqueeze(1).cpu().numpy()).cuda().repeat(1,self.C2,1,1,1)).sum(dim=4,keepdim=False)
            x = (x* cos.unsqueeze(1).repeat(1,self.C2,1,1,1).detach()).sum(dim=4,keepdim=False)
            #W = torch.from_numpy(cos.unsqueeze(1).repeat(1,self.C2,1,1,1).cpu().numpy()).cuda()
            #print(cos.unsqueeze(1).repeat(1,self.C2,1,1,1).size())
            #x = (x* torch.from_numpy(torch.ones_like(cos.unsqueeze(1).cpu()).numpy()).cuda().repeat(1,self.C2,1,1,1)).sum(dim=4,keepdim=False)
            #x = torch.matmul(x,cos).squeeze(4).sum(dim=3,keepdim=True)
            x = self.pool(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

def ConvChannels(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, *args):
            x = args[0][0] if isinstance(args[0],tuple) else args[0]
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

def ConvNeighbours(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along neighbours. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, *args):
            x = args[0][0] if isinstance(args[0],tuple) else args[0]
            x = x.permute(0,2,1,3)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
