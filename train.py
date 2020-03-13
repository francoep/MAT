#!/usr/bin/env python3

#TODO: clean the script from changing into the src directory and having everything being mapped from there.

import os
import pandas as pd
import torch
#from tqdm import tqdm
import numpy as np
from rdkit.Chem import AllChem as Chem
os.chdir('src')

import argparse
import matplotlib.pyplot as plt
from featurization.data_utils import load_data_from_df, construct_loader
from transformer import make_model

parser=argparse.ArgumentParser(desctiption='Train and test MAT model on datasets')
parser.add_argument('--train',type=str,required=True,help='Train data filename. Assumed to be a csv: smile,y')
parser.add_argument('--test',type=str,required=True,help='Test data filename. Assumed to be a csv: smile,y')
parser.add_argument('--pretrain',action='store_true',help='Flag to use the pretrained weights. If set, will use. Assumed to be pretrained_weights.pt')
parser.add_argument('--figdir',type=str,required=True,help='Absolutepath to the directory for the figures')
parser.add_argument('--savemodel',type=bool, action='store_true',help='Flag to save the trained model. The filename will be Traindata.model')

args=parser.parse_args()

#todo -- implement changing hyper params -- Sections to support indicated by a comment of #1#

trainX, trainy=load_data_from_df('../'+args.train,one_hot_formal_charge=True)
batch_size=8
data_loader=construct_loader(trainX,testy,batch_size)

d_atom = trainX[0][0].shape[1]

#1#
model_params= {
    'd_atom': d_atom,
    'd_model': 1024,
    'N': 8,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.33,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': 0.0,
    'aggregation_type': 'mean'
}

model=make_model(**model_params)

if args.pretrain:
    pretrained_name = '../pretrained_weights.pt'  # This file should be downloaded first (See README.md).
    pretrained_state_dict = torch.load(pretrained_name)
    model_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if 'generator' in name:
             continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

torch.cuda.empty_cache()
model.cuda()

#1#
criterion=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

losses=[]

#1#
#Training loop
for epoch in range(500):
    for batch in data_loader:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        
        
        loss=criterion(y_pred,y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#saving the trained model
if args.savemodel:
    torch.save(model.state_dict(),args.train.replace('.csv','.model'))

#now that the training is complete -- we need to output the training figures
epoch_mean_losses=[]
for i in range(int(len(losses)/len(data_loader))):
    tmp=[]
    for j in range(int(len(data_loader))):
        tmp.append(losses[int(j+i*len(losses)/len(data_loader))])
    epoch_mean_losses.append(np.mean(tmp))

if not os.path.isdir(args.figdir):
    os.mkdir(args.figdir)

plt.plot(epoch_mean_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.savefig(args.figdir+'/training_losses_split'+args.test.split('_test')[1][0]+'.png')

plt.plot(epoch_mean_losses[-100:])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss -- last 100 epochs')
plt.savefig(args.figdir+'/training_losses_split'+args.test.split('_test')[1][0]+'.png')

#after the plots were made -- we need to evaluate the test_set
testX, testy=load_data_from_df(args.train,one_hot_formal_charge=True)
testdata_loader=construct_loader(testX,testy,batch_size)

gold=np.array([])
preds=np.array([])
for batch in testdata_loader:
    adjacency_matrix, node_features, distance_matrix, y = batch
    gold=np.append(gold,y.tolist())
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
    preds=np.append(preds,y_pred.tolist())

print('R2',np.corrcoef(preds,gold)[0][1]**2)
print('RMSE',np.sqrt(np.mean(preds-gold)**2))