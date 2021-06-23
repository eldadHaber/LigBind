import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

from src import graphOps as GO
from src import utils
from src import graphNet as GN
#from src import graphOps as GO
#from src import utils
#from src import graphNet as GN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#lig = torch.load('../../../data/intmin_docking_data/1f02_subset_train.pt')
#lig = torch.load('../../../data/mpro_docking_data/mpro_lig_select.pt')
#pro = torch.load('../../../data/mpro_docking_data/mpro_rec_select.pt')
#lig = torch.load('../../../data/intmin_docking_data/1f02_train.pt')
lig = torch.load('../../../data/intmin_docking_data/all_200.pt')
pro = torch.load('../../../data/intmin_docking_data/all_rec_info.pt')


def getLigData(lig,IND):

    #X      = lig['coords']
    #atype  = lig['atom_types']
    #charge = lig['charge']
    #score  = lig['score']
    #dof    = lig['torsdof']
    #atomc  = lig['atom_connect']
    #btype  = lig['bond_type']

    n = len(IND)
    II = torch.zeros(0).long()
    JJ = torch.zeros(0).long()
    XE = torch.zeros(0,6)
    XN = torch.zeros(0,19)
    NL = torch.zeros(n, dtype=torch.long)
    score = torch.zeros(n)
    receptor = ['0']*n
    cnt = 0
    for j in range(n):

        i = IND[j]
        score[j] = torch.tensor([float(lig['scores'][i])])

        I = (lig['atom_connect'][i][:,0]-1).long()
        J = (lig['atom_connect'][i][:,1]-1).long()

        II = torch.cat((II,I+cnt))
        JJ = torch.cat((JJ,J+cnt))

        #xe = torch.tensor(lig['bonds'][i][:,2], dtype=torch.long)
        #xe = lig['bond_type'][i].long()
        #oo = torch.zeros(N,6)
        #xe = torch.cat((xe,xe,oo),dim=0)
        n  = I.shape[0]
        xe = torch.zeros(n,6)

        XE = torch.cat((XE,xe),dim=0)

        xn = lig['atom_types'][i]
        cn = lig['charges'][i].unsqueeze(1)

        xn = torch.cat((xn,5*cn),dim=1)
        XN = torch.cat((XN, xn), dim=0)
        NL[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

        receptor[j] = lig['receptor'][i]

    XE = XE.t().unsqueeze(0)
    XN = XN.t().unsqueeze(0)

    return II, JJ, XN, XE.float(), score, NL, receptor


def getPocketData(P,IND, conf):

    n = len(IND)
    II = torch.zeros(0)
    JJ = torch.zeros(0)
    XE = torch.zeros(0)
    XN = torch.zeros(0,18)

    NP    = torch.zeros(n, dtype=torch.long)
    cnt = 0
    for j in range(n):
        i = IND[j]

        X = P[conf]['coords'][i]
        D = utils.distanceMap(X)
        D = D / D.std()
        D = torch.exp(-2 * D)
        D = torch.triu(D,0)

        # Choose nsparse neiboughrs
        nsparse = 9
        vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
        nd = D.shape[0]
        I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
        I = I.view(-1)
        J = indices.view(-1).type(torch.LongTensor)

        II = torch.cat((II,I+cnt))
        JJ = torch.cat((JJ,J+cnt))

        xn = P[conf]['atom_types'][i]
        XN = torch.cat((XN,xn),dim=0)
        xe = D[I, J]
        XE = torch.cat((XE,xe))

        cn = P[conf]['charges'][i].unsqueeze(1)
        xn = torch.cat((xn, 5*cn), dim=1)

        NP[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

    XN = XN.t()

    IJ1 = torch.cat((II, JJ)).unsqueeze(1)
    IJ2 = torch.cat((JJ, II)).unsqueeze(1)
    IJ = torch.cat((IJ1, IJ2), dim=1)
    IJ = IJ.unique(dim=0)
    II = IJ[:,0]
    JJ = IJ[:,1]
    return II.long(), JJ.long(), XN.unsqueeze(0), XE.unsqueeze(0).unsqueeze(0), NP


def computeScore(XNOutP, XNOutL, NP, NL, bias=0):

    cntP = 0
    cntL = 0
    n = len(NL)
    compScore = torch.zeros(n)
    for i in range(n):
        xnOutP = XNOutP[:,:,cntP:cntP+NP[i]]
        xnOutL = XNOutL[:,:,cntL:cntL+NL[i]]
        bindingScore = bias+torch.dot(torch.mean(xnOutP, dim=2).squeeze(), torch.mean(xnOutL, dim=2).squeeze())
        compScore[i] = bindingScore
        cntP = cntP + NP[i]
        cntL = cntL + NL[i]

    return compScore

# Setup the network for ligand and its parameters

nNin    = 19
nopen   = 32
nNclose = 8
nlayer  = 6

modelL = GN.graphNetwork(nNin, nopen, nNclose, nlayer)
modelL.to(device)


total_params = sum(p.numel() for p in modelL.parameters())
print('Number of parameters  for ligand', total_params)

# network for the protein
# Setup the network for protein and its parameters
nNin = 18
nopen = 16
nNclose = 8
nlayer = 3

modelP = GN.graphNetwork(nNin, nopen, nNclose, nlayer)
modelP.to(device)
#
total_params = sum(p.numel() for p in modelP.parameters())
print('Number of parameters  for pocket', total_params)
#

#IL, JL, XNL, XEL, score, NL, receptor = getLigData(lig,[2])
#IP, JP, XNP, XEP, NP                  = getPocketData(pro,[0],receptor[0])

# running the model for the protein
#nNodesP = XNP.shape[2]
#GP = GO.graph(IP, JP, nNodesP)
#xnOutP = modelP(XNP, XEP, GP)


# running the model for the ligand
#nNodesL = XNL.shape[2]
#GL = GO.graph(IL, JL, nNodesL)
#xnOutL = modelL(XNL, XEL, GL)

#
#s = computeScore(xnOutP, xnOutL, NP, NL)


#### Start Training ####

bias = nn.Parameter(torch.ones(1)-6.6126)

#optimizer = optim.Adam([{'params': modelP.K1Nopen, 'lr': lrO},
#                        {'params': modelP.K2Nopen, 'lr': lrO},
#                        {'params': modelP.KE1, 'lr': lrE1},
#                        {'params': modelP.KE2, 'lr': lrE2},
#                        {'params': modelP.KNclose, 'lr': lrC},
#                        {'params': modelP.Kw, 'lr': lrW},
#                        {'params': modelL.K1Nopen, 'lr': lrO},
#                        {'params': modelL.K2Nopen, 'lr': lrO},
#                        {'params': modelL.KE1, 'lr': lrE1},
#                        {'params': modelL.KE2, 'lr': lrE2},
#                        {'params': modelL.KNclose, 'lr': lrC},
#                        {'params': modelL.Kw, 'lr': lrW}])

ndata = 32
batchSize = 16
lrO = 1e-2
lrC = 1e-2
lrW = 1e-2
lrE1 = 1e-2
lrE2 = 1e-2
lrbias = 1e-1

optimizer = optim.Adam([{'params': modelL.K1Nopen, 'lr': lrO},
                       {'params': modelL.K2Nopen, 'lr': lrO},
                       {'params': modelL.KE1, 'lr': lrE1},
                        {'params': modelL.KE2, 'lr': lrE2},
                       {'params': modelL.KNclose, 'lr': lrC},
                       {'params': modelL.Kw, 'lr': lrW},
                       {'params': bias, 'lr': lrbias}])


#optimizer = optim.LBFGS([modelL.K1Nopen, modelL.K2Nopen, modelL.KE1, modelL.KE2, modelL.KNclose, modelL.Kw, H],
#                        lr=1, max_iter=100)

epochs = 30

hist = torch.zeros(epochs)

bestLoss = 1e11
#bestModelP = modelP
bestModelL = modelL


for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    for i in range(ndata//batchSize):

        optimizer.zero_grad()
        IND = torch.arange(i * batchSize, (i + 1) * batchSize)
        # Get the lig network
        IL, JL, xnL, xeL, truescore, NL, receptor = getLigData(lig, IND)
        nNodesL = xnL.shape[2]
        GL = GO.graph(IL, JL, nNodesL)
        optimizer.zero_grad()
        xnOutL = modelL(xnL, xeL, GL)


        # get protein data
        unique_rec = set(receptor)
        xnOutP = torch.zeros(1, xnOutL.shape[1],0)
        NP = torch.zeros_like(NL)
        cnt = 0
        for kk in receptor: #unique_rec:
            IP, JP, XNP, XEP, NPi = getPocketData(pro, [0], kk)
            GP = GO.graph(IP, JP, NPi)
            xnOutPi = modelP(XNP, XEP, GP)
            xnOutP = torch.cat((xnOutP,xnOutPi), dim=2)
            NP[cnt] = NPi
            cnt += 1

        predScore = computeScore(xnOutP, xnOutL, NP, NL, bias)

        loss = F.mse_loss(predScore, truescore)
        loss.backward()
#            print(loss.item())
#            return loss

        #optimizer.step(closure)
        optimizer.step()

        gC  = modelL.KNclose.grad.norm().item()
        gE1 = modelL.KE1.grad.norm().item()
        gE2 = modelL.KE2.grad.norm().item()
        gO1 = modelL.K1Nopen.grad.norm().item()
        gO2 = modelL.K2Nopen.grad.norm().item()
        gw  = modelL.Kw.grad.norm().item()

        #aloss = closure()

        aloss += loss.detach()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = torch.sqrt(aloss / nprnt)
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E"
                  % (j, i, aloss, gC, gE1, gE2, gO1, gO2, gw), flush=True)
            aloss = 0.0

            if aloss < bestLoss:
                #bestModelP = modelP
                bestModelL = modelL
                bestLoss   = aloss

    # Validation

    # with torch.no_grad():
    #     IND = torch.arange(5000, 6000)
    #     # Get the lig data
    #     IL, JL, xnL, xeL, truescore, NL = getLigData(lig, IND)
    #     nNodesL = xnL.shape[2]
    #     GL = GO.graph(IL, JL, nNodesL)
    #
    #     xnOutL = modelL(xnL, xeL, GL)
    #     xnOutP = torch.ones(xnOutL.shape)
    #     predScore = computeScore(xnOutP, xnOutL, NL, NL, H)
    #
    #     loss = F.mse_loss(predScore, truescore)  # /F.mse_loss(truescore*0, truescore)
    #     print("%2d   %10.3E" % (j, torch.sqrt(loss).item()))
    #     print("=========================================================")


#IND = torch.arange(ndata)
#IND = torch.arange(5000,6000)

# Get the lig data
