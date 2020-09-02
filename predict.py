#!/usr/bin/env python3

# This script will predict a saved model. Note -- we assume the saved model is GPU-enabled.
# Input -- saved model file, test file. Output -- CSV with <SMILE>,<True>,<Pred>
#  TODO -- make GPU/CPU switch
#  TODO -- add timer?

import os
import sys
import pandas as pd
import torch
import numpy as np

#we assume that you are running the model from the main section of this github repository
sys.path.append(os.getcwd())
sys.path.append('src')

import argparse
import time
from transformer import make_model
from featurization.data_utils import load_data_from_df, construct_loader
import pickle

parser=argparse.ArgumentParser(description='Predict MAT model on a given test set')
parser.add_argument('-m','--model',type=str,required=True,help='Trained torch model file')
parser.add_argument('-i','--input',type=str,required=True,help='File to evaluate. Assumed format is "<SMILE>,<solubility>"')
parser.add_argument('-o','--output',type=str,required=True,help='File for Predictions. Format is "<SMILE>,<True>,<Predicted>"')
parser.add_argument('--stats',default=False,action='store_true',help='Flag to print the R2, RMSE, and the time to perform the evaluation.')
parser.add_argument('--twod',default=False,action='store_true',help='Flag to use 2D conformers for distance matrix.')
args=parser.parse_args()

if args.stats:
	start=time.time()

#loading the data
X,gold=load_data_from_df(args.input,one_hot_formal_charge=True,use_data_saving=False,two_d_only=args.twod)
data_loader=construct_loader(X,gold,batch_size=8,shuffle=False)

if args.stats:
	print('Loading data time:',time.time()-start)
	start=time.time()

#constructing the model
##TODO -- Add functionality to read the model's definition from a file!!
d_atom=X[0][0].shape[1]
model_params={
	'd_atom': d_atom,
    'd_model': 512,
    'N': 16,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.0,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': 0.1,
    'aggregation_type': 'mean'
}

model=make_model(**model_params)
state_dict=torch.load(args.model)
model_state_dict=model.state_dict()
for name, param in state_dict.items():
	if 'generator' in name:
		continue
	if isinstance(param,torch.nn.Parameter):
		param=param.data
	model_state_dict[name].copy_(param)
model.eval()
model.cuda()

if args.stats:
	print('Model construction time:',time.time()-start)
	start=time.time()

##getting the predictions
preds=np.array([])
ys=np.array([])
for batch in data_loader:
	adjacency_matrix, node_features,distance_matrix,y=batch
	batch_mask=torch.sum(torch.abs(node_features),dim=-1) != 0
	pred=model(node_features,batch_mask,adjacency_matrix,distance_matrix,None)
	preds=np.append(preds,pred.tolist())
	ys=np.append(ys,y.tolist())

if args.stats:
	print('Model Prediction time:',time.time()-start)
	print('RMSE:',np.sqrt(np.mean((preds-ys)**2)))
	print('R2:',np.corrcoef(preds,ys)[0][1]**2)

with open(args.output,'w') as outfile:
	outfile.write('smile,true,pred\n')
	lines=open(args.input).readlines()
	lines=lines[1:]
	preds=preds.tolist()
	assert len(lines)==len(preds)

	for l,p in zip(lines,preds):
		outfile.write(l.rstrip()+','+str(p))