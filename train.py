#!/usr/bin/env python3

#TODO: clean the script from changing into the src directory and having everything being mapped from there.

import os
import sys
import pandas as pd
import torch
#from tqdm import tqdm
import numpy as np
#from rdkit.Chem import AllChem as Chem

#TODO -- change this behavior
os.chdir('src')
print(os.getcwd())
sys.path.append(os.getcwd())
import argparse
#import matplotlib.pyplot as plt  removing the plotting behavior -- instead just saving out the data
from featurization.data_utils import load_data_from_df, construct_loader
from transformer import make_model
import pickle



def logcosh_loss(output,target):
    '''
    Log of the cosh of predicted-target tensors.
    '''
    loss=torch.sum(torch.log(torch.cosh(output-target)))
    return loss


parser=argparse.ArgumentParser(description='Train and test MAT model on datasets')
parser.add_argument('--train',type=str,required=True,help='Train data filename. Assumed to be a csv: smile,y')
parser.add_argument('--test',type=str,required=True,help='Test data filename. Assumed to be a csv: smile,y')
parser.add_argument('--pretrain',action='store_true',help='Flag to use the pretrained weights. If set, will use. Assumed to be pretrained_weights.pt')
parser.add_argument('--datadir',type=str,required=True,help='Absolutepath to the directory for the data from training and testing the model. Saved filenames will be <train>_<loss>_<optim>_<optimHP>_<epochs>_<train|test|testmetrics>.pi')
parser.add_argument('--savemodel', action='store_true',help='Flag to save the trained model. The filename will be <traindata>_trained.model')
parser.add_argument('-e','--epochs', type=int, default=500,help='Number of epochs to train the model for. Defaults to 500')
parser.add_argument('-l','--loss',type=str,default='mse',help='Loss Function to use: mse, mae, huber, or logcosh')
parser.add_argument('-o','--optimizer',type=str,default='sgd',help='Optimizer for training the model: sgd, or adam')

args=parser.parse_args()

#TODO -- implement 'quantile' loss funtion
assert args.loss in set(['mse','mae','huber','logcosh']) and args.optimizer in set(['sgd','adam'])

trainname_for_output=args.train.split('/')[-1].replace('.csv','_')

#todo -- implement changing hyper params -- Sections to support indicated by a comment of #1#

trainX, trainy=load_data_from_df('../'+args.train,one_hot_formal_charge=True)
batch_size=8
data_loader=construct_loader(trainX,trainy,batch_size)

d_atom = trainX[0][0].shape[1]

# Since we are using the provided pretrained weights -- these are locked in place
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

#TODO -- extra loss functions -- quantile?
if args.loss == 'mse':
    criterion=torch.nn.MSELoss(reduction='sum')
elif args.loss=='mae':
    criterion=torch.nn.L1Loss(reduction='sum')
elif args.loss=='huber':
    criterion=torch.nn.SmoothL1Loss(reduction='sum')
elif args.loss=='logcosh':
    criterion=None


#TODO -- extra optimizers -- SGD (lr, momemtum,weight_decay), ADAM(lr, beta1, beta2, eps, weight_decay)  & implement their hyper-params into running.
if args.optimizer=='sgd':
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9,weight_decay=0,dampening=0,nesterov=False)
elif args.optimizer=='adam':
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

losses=[]

#Training loop
for epoch in range(args.epochs):
    for batch in data_loader:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        
        
        if criterion==None and args.loss=='logcosh':
            loss=logcosh_loss(y_pred,y)
        else:
            loss=criterion(y_pred,y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#saving the trained model
if args.savemodel:
    torch.save(model.state_dict(),trainname_for_output+'trained.model')

#now that the training is complete -- we need to output the training losses
epoch_mean_losses=[]
for i in range(int(len(losses)/len(data_loader))):
    tmp=[]
    for j in range(int(len(data_loader))):
        tmp.append(losses[int(j+i*len(losses)/len(data_loader))])
    epoch_mean_losses.append(np.mean(tmp))

#making the directory for the data
if not os.path.isdir(args.datadir):
    os.mkdir(args.datadir)

#saving the training loss -- <train>_<loss>_<optim>_<optimHP>_<epochs>_<train|test|testmetrics>.pi
#TODO -- adding the hyper parameters for the optimizer
with open(args.datadir+'/'+trainname_for_output+args.optimizer+'_'+str(args.epochs)+'_trainloss.pi','wb') as outfile:
    pickle.dump(epoch_mean_losses,outfile)

#we need to evaluate the test_set
testX, testy=load_data_from_df('../'+args.test,one_hot_formal_charge=True)
testdata_loader=construct_loader(testX,testy,batch_size)

gold=np.array([])
preds=np.array([])
model.eval()

for batch in testdata_loader:
    adjacency_matrix, node_features, distance_matrix, y = batch
    gold=np.append(gold,y.tolist())
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
    preds=np.append(preds,y_pred.tolist())

with open(args.datadir+'/'+trainname_for_output+args.optimizer+'_'+str(args.epochs)+'_test.pi','wb') as outfile:
    pickle.dump((gold,preds),outfile)

r2=np.corrcoef(preds,gold)[0][1]**2
rmse=np.sqrt(np.mean(preds-gold)**2)

with open(args.datadir+'/'+trainname_for_output+args.optimimzer+'_'+str(args.epochs)+'_testmetrics.txt','w') as outfile:
    outfile.write(f'R2: {r2}\n')
    outfile.write(f'RMSE: {rmse}\n')