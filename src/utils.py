import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X

def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def distanceMap(coords):
    XTX = coords@coords.t()
    d   = torch.diag(XTX).unsqueeze(0)
    D   = torch.relu(d + d.t() - 2*XTX)
    return D

def getBatchData(coordss, elementss, charges=[]):

    n = max([p.shape[0] for p in elementss])
    k = len(coordss)
    if len(charges)>0:
        M = 2*elementss[0].shape[1] + 1 + 2
    else:
        M = 2 * elementss[0].shape[1] + 1

    inputs = torch.zeros(k,M,n,n)
    Mask   = torch.zeros(k,1,n,n)
    for i in range(k):
        if len(charges)>0:
            input = getData(coordss[i], elementss[i], charges[i])
        else:
            input = getData(coordss[i], elementss[i])
        m     = input.shape[-1]
        inputs[i,:,:m,:m] = input
        Mask[i, 0, :m,:m] = 1.0
    return inputs, Mask


def getData(coords, elements, charge=[]):
    D = distanceMap(coords).unsqueeze(0)
    elements = elements.t().unsqueeze(1)
    G = (elements - elements.transpose(1, 2))
    A = 0.5*(elements + elements.transpose(1, 2))
    E = torch.cat([A, G], dim=0)
    if len(charge)>0:
        gc = (charge.unsqueeze(0) - charge.unsqueeze(1)).unsqueeze(0)
        ac = 0.5*(charge.unsqueeze(0) + charge.unsqueeze(1)).unsqueeze(0)
        C = torch.cat([gc,ac],dim=0)
        input = torch.cat([D,E,C],dim=0).unsqueeze(0)
    else:
        input = torch.cat([D, E], dim=0).unsqueeze(0)
    return input

class resNet(nn.Module):

    def __init__(self, nNin, nopen, nhid, nout, nlayer, h=0.1):
        super(resNet, self).__init__()

        self.h = h
        stdv = 1e-2
        self.Kopen  = nn.Parameter(torch.randn(nopen, nNin,5,5) * stdv)
        self.K      = nn.Parameter(torch.randn(nlayer, nhid, nopen, 5,5) * stdv)
        self.G      = nn.Parameter(torch.randn(nlayer, nopen, nhid, 5,5) * stdv)
        self.W      = nn.Parameter(torch.randn(nout,    nopen, 1, 1) * stdv)

    def forward(self, x, Mask=1.0):

        # x = [B, C, N, N]
        # Opening layer
        x = Mask*conv2(x,self.Kopen)

        nlayers = self.K.shape[0]
        for i in range(nlayers):
            Ki = self.K[i]
            Gi = self.G[i]

            dx  = Mask*conv2(x,Ki)
            dx  = tv_norm(dx)
            dx  = torch.relu(dx)
            dx  = Mask*conv2(dx,Gi)

            x  = x + self.h*dx

        x = conv2(x, self.W)
        x = torch.mean(x,dim=[2,3])
        return x


