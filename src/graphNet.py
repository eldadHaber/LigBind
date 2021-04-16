import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import graphOps as GO


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


#def doubleLayer(x, K1, K2):
#    x = F.conv1d(x, K1.unsqueeze(-1))
#    #x = F.layer_norm(x, x.shape)
#    x = tv_norm(x)
#    x = torch.relu(x)
#    x = F.conv1d(x, K2.unsqueeze(-1))
#    return x


class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, varlet=False):
        super(graphNetwork, self).__init__()

        self.h = h
        self.varlet = varlet
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)

        if varlet:
            Nfeatures = 2 * nopen
        else:
            Nfeatures = 3 * nopen

        Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
        Idt = torch.eye(nopen,nhid).unsqueeze(0)
        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)

        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
        Idt = torch.eye(nopen,nhid).unsqueeze(0)
        IdTensor = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)

        self.KN1 = nn.Parameter(IdTensor * stdvp)
        self.KN2 = nn.Parameter(IdTensort * stdvp)

    def edgeConv(self, xe, K):

        if K.dim() == 2:
            xe = F.conv1d(xe, K.unsqueeze(-1))
        else:
            xe = conv1(xe, K)
        return xe

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        x = tv_norm(x)
        x = torch.tanh(x)
        x = self.edgeConv(x, K2)

        return x

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):

            gradX = Graph.nodeGrad(xn)
            intX  = Graph.nodeAve(xn)
            if self.varlet:
                dxe = torch.cat([intX, gradX], dim=1)
            else:
                dxe = torch.cat([intX, xe, gradX], dim=1)

            dxe = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            if self.varlet:
                xe = xe + self.h * dxe
                flux = xe
            else:
                flux = dxe

            divE = Graph.edgeDiv(flux)
            aveE = Graph.edgeAve(flux, method='ave')

            if self.varlet:
                dxn = torch.cat([aveE, divE], dim=1)
            else:
                dxn = torch.cat([aveE, xn, divE], dim=1)

            dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])

            xn = xn - self.h * dxn
            if self.varlet == False:
                xe = xe - self.h * dxe

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))

        return xn, xe




Test = False
if Test:
    nNin = 20
    nEin = 3
    nNopen = 32
    nEopen = 16
    nEhid = 128
    nNclose = 3
    nEclose = 2
    nlayer = 18
    model = graphNetwork(nNin, nEin, nNopen, nEopen, nEhid, nNclose, nEclose, nlayer)

    L = 55
    xn = torch.zeros(1, nNin, L)
    xn[0, :, 23] = 1
    xe = torch.ones(1, nEin, L, L)

    G = GO.dense_graph(L)

    xnout, xeout = model(xn, xe, G)

