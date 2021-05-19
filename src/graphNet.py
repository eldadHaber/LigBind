import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
import graphOps as GO


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


class graphNetwork(nn.Module):

    def __init__(self, nNin, nopen, nNclose, nlayer, h=0.1):
        super(graphNetwork, self).__init__()

        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        #self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
        #self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        #nopen      = 3*nopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen

        nhid = 5*nopen
        Id  = (torch.eye(nhid,5*nopen)).unsqueeze(0) #+ 1e-3*torch.randn(nhid,5*nopen)).unsqueeze(0)
        Idt = (torch.eye(5*nopen, nhid)).unsqueeze(0) # + 1e-3*torch.randn(5*nopen,nhid)).unsqueeze(0)

        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen)*1e-2)
        self.Kw = nn.Parameter(torch.ones(nopen,1))

    def doubleLayer(self, x, K1, K2):

        x = torch.tanh(x)
        x = F.conv1d(x, K1.unsqueeze(-1))  # self.edgeConv(x, K1)
        x = tv_norm(x)
        x = torch.tanh(x)
        x = F.conv1d(x, K2.unsqueeze(-1))
        x = torch.tanh(x)

        return x

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe =  [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        #xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        #xn = torch.cat([xn,Graph.edgeDiv(xe), Graph.edgeAve(xe)], dim=1)

        nlayers = self.KE1.shape[0]

        xnold = xn
        for i in range(nlayers):

            # Compute the distance in real space
            x3    = F.conv1d(xn, self.KNclose.unsqueeze(-1))
            w     = Graph.edgeLength(x3)
            w     = self.Kw@w
            w     = w/(torch.std(w)+1e-4)
            w     = torch.exp(-(w**2))
            #w     = torch.ones(xe.shape[2], device=xe.device)

            gradX   = Graph.nodeGrad(xn,w)
            intX    = Graph.nodeAve(xn,w)
            xgradX  = gradX*intX
            gradXsq = gradX*gradX
            xSq     = intX*intX

            dxe = torch.cat([gradX, intX, xgradX, gradXsq, xSq], dim=1)
            dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            divE = Graph.edgeDiv(dxe[:,:self.nopen,:],w)
            aveE = Graph.edgeAve(dxe[:,self.nopen:2*self.nopen,:],w)
            aveB = Graph.edgeAve(dxe[:,2*self.nopen:3*self.nopen, :], w)
            aveI = Graph.edgeAve(dxe[:,3*self.nopen:4*self.nopen, :], w)
            aveS = Graph.edgeAve(dxe[:,4*self.nopen:, :], w)

            tmp  = xn.clone()
            xn   = 2*xn - xnold - self.h * (divE + aveE + aveB + aveI + aveS)
            #xn = xn - self.h * (divE + aveE + aveB + aveI + aveS)

            xnold = tmp

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xn = torch.cat((torch.relu(xn),torch.relu(-xn)),dim=1)

        return xn


#===============================================================

def vectorLayer(V, wx, wy, wz):
    nvecs = V.shape[1]//3
    Vx    = F.conv1d(V[:,nvecs,:],wx.unsqueeze(1))
    Vy    = F.conv1d(V[:,nvecs:2*nvecs,:],wy.unsqueeze(1))
    Vz    = F.conv1d(V[:, 2*nvecs:,:],wz.unsqueeze(1))
    Vout  = torch.cat((Vx,Vy,Vz),dim=1)

    return Vout

def vectorFeatures(V, Graph):

    # get invariant vector features
    V = Graph.nodeGrad(V)
    nvecs = V.shape[1] // 3
    Vx = V[:, nvecs, :]
    Vy = V[:, nvecs:2 * nvecs, :]
    Vz = V[:, 2 * nvecs:, :]

    # Length
    L = Vx**2 + Vy**2 + Vz**2

    return L


class graphNetworkEqvrnt(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, const=False):
        super(graphNetworkEqvrnt, self).__init__()

        self.const = const
        self.h = h
        stdv = 1.0 #1e-2
        stdvp = 1.0 # 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
        self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        nopen      = 3*nopen  # [xn; Av*xe; Div*xe]
        self.nopen = nopen
        Nfeatures  = 2*nopen+1 # [G*xn; Av*xn]

        Id  = torch.eye(nhid,Nfeatures).unsqueeze(0)
        Idt = torch.eye(Nfeatures,nhid).unsqueeze(0)
        IdTensor  = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)
        self.KE1 = nn.Parameter(IdTensor * stdvp)
        self.KE2 = nn.Parameter(IdTensort * stdvp)

        self.Kw1 = nn.Parameter(torch.ones(nopen,1))
        self.Kw2 = nn.Parameter(torch.ones(3,Nfeatures))

    def doubleLayer(self, x, K1, K2):

        x = torch.tanh(x)
        x = F.conv1d(x, K1.unsqueeze(-1))  # self.edgeConv(x, K1)
        x = tv_norm(x)
        x = torch.tanh(x)
        x = F.conv1d(x, K2.unsqueeze(-1))
        x = torch.tanh(x)

        return x


    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe =  [B, C, E]
        # Opening layer
        Nnodes = xn.shape[2]
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        xn = torch.cat([xn,Graph.edgeDiv(xe), Graph.edgeAve(xe)], dim=1)

        #Coords = (torch.tensor([1,0,0]).unsqueeze(1)@torch.arange(Nnodes).unsqueeze(0)).unsqueeze(0) + 0.1*torch.randn(1,3,Nnodes)
        Coords = torch.zeros(1, 3, Nnodes)
        j = 0
        for i in range(1,Nnodes):
            Coords[0,j,i] = Coords[0,j,i-1] + 3.8
            j = np.mod(j + 1,3)

        if self.const:
            Coords = proj(Coords, torch.eye(3, 3), n=1000)

        CoordsOld = Coords

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):

            # Compute the distance in real space
            w     = Graph.edgeLength(xn)
            w     = self.Kw1@w
            w     = w/(torch.std(w)+1e-4)
            w     = torch.tanh(w)

            # Node to edge
            gradX = Graph.nodeGrad(xn,w)
            intX = Graph.nodeAve(xn,w)
            d    = Graph.edgeLength(Coords).unsqueeze(0)

            # Edge attributes
            dxe = torch.cat([gradX, intX, d], dim=1)
            dxe  = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            # Update Coordinates
            w3         = self.Kw2@dxe
            w3         = w3/(torch.std(w3)+1e-4)
            w3         = torch.tanh(w3)
            gradCoords = Graph.nodeGrad(Coords, w3)
            aveCoords  = Graph.edgeAve(gradCoords, w3)

            tmp    = Coords
            Coords = CoordsOld + 2*self.h*aveCoords
            CoordsOld = tmp


            divE = Graph.edgeDiv(dxe[:,:self.nopen,:],w)
            aveE = Graph.edgeAve(dxe[:,self.nopen:-1,:],w)

            xn = xn - self.h * (divE + aveE)

            if self.const:
                Coords = proj(Coords, torch.eye(3, 3), n=50)


        if self.const:
            c = constraint(Coords)
            if c.abs().mean() > 0.1:
                Coords = proj(Coords, torch.eye(3, 3), n=500)


        return Coords, xn, xe



#
# u_t = f(u,(Ku_x)_x, uu_x, u^2, u_x^2, theta)
#