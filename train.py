#!/usr/bin/env python3

import os
import sys
import pandas as pd
import torch
#from tqdm import tqdm
import numpy as np
import wandb
import time

#we assume that you are running the model from the main section of this github repository
sys.path.append(os.getcwd())
sys.path.append('src')

import argparse
#import matplotlib.pyplot as plt  removing the plotting behavior -- instead just saving out the data
from transformer import make_model
import pickle


parser=argparse.ArgumentParser(description='Train and test MAT model on datasets. Use either --trainfile  & --testfile OR --prefix & --fold. --trainfile will be preferred if both are set.')
parser.add_argument('--trainfile',type=str,default='',help='Specify training data file for model. Requires the use of --testfile.')
parser.add_argument('--testfile',type=str,default='',help='Spefify testing data file for model. Used in conjunction with --trainfile.')
parser.add_argument('--prefix',type=str,default='',help='Prefix for the train and test data. Assumed to follow <prefix>_train<fold>.csv. Requires --testfile.')
parser.add_argument('--fold',type=str,default='',help='Fold for the datafiles. Used in conjunction with --prefix')
parser.add_argument('--pretrain',action='store_true',default=False,help='Flag to use the pretrained weights. If set, will use. Assumed to be pretrained_weights.pt')
parser.add_argument('--datadir',type=str,default='sweep',help='Absolutepath to the directory for the data from training and testing the model (Def: sweep). Saved filenames will be <prefix>_<fold>_e<epochs>_<loss>_<optimizer>_lr<lr>_m<momentum>_wd<weightdecay>_<trainlosses|trainepochlosses|testdic>.pi')
parser.add_argument('--savemodel', action='store_true',default=False,help='Flag to save the trained model. The filename will be <prefix>_<fold>_trained.model')
parser.add_argument('-e','--epochs', type=int, default=500,help='Number of epochs to train the model for. Defaults to 500')
parser.add_argument('-l','--loss',type=str,default='mse',help='Loss Function to use: mse, mae, huber, or logcosh.')
parser.add_argument('-o','--optimizer',type=str,default='sgd',help='Optimizer for training the model: sgd, or adam.')
parser.add_argument('--lr',type=float,default=1e-4, help='Learning rate for the Optimizer. Defaults to 1e-4.')
parser.add_argument('--momentum',type=float,default=0.9,help='Momentum for SGD optimizer. Defaults to 0.9')
parser.add_argument('--weight_decay',type=float,default=0,help='L2 pentalty for Optimizer. Defaults to 0')
parser.add_argument('--dampening',type=float,default=0,help='Dampening for momentum in SGD Optimizer. Defaults to 0')
parser.add_argument('--beta1',type=float,default=0.9,help='Beta1 for ADAM optimizer. Defaults to 0.9')
parser.add_argument('--beta2',type=float,default=0.999,help='Beta2 for ADAM optimizer. Defaults to 0.999')
parser.add_argument('--epsilon',type=float,default=1e-08,help='Epsilon for ADAM optimizer. Defaults to 1e-08.')
parser.add_argument('--dropout',type=float,default=0,help='Applying Dropout to model weights when training')
parser.add_argument('--ldist',type=float,default=0.33,help='Lambda for model attention to the distance matrix. Defaults to 0.33 (even between dist, attention, and graph features)')
parser.add_argument('--lattn',type=float,default=0.33,help='Lambda for model attention to the attention matrix. Defaults to 0.33 (even between dist, attenttion, and graph features)')
parser.add_argument('--Ndense',type=int,default=1,help='Number of Dense blocks in FeedForward section. Defaults to 1')
parser.add_argument('--heads',type=int,default=16,help='Number of attention heads in MultiHeaded Attention. **Needs to evenly divide dmodel Defaults to 16.')
parser.add_argument('--dmodel',type=int,default=1024,help='Dimension of the hidden layer for the model. Defaults to 1024.')
parser.add_argument('--nstacklayers',type=int,default=8,help='Number of stacks in the Encoder layer. Defaults to 8')
parser.add_argument('--cpu',action='store_true',default=False,help='Flag to have model be CPU only.')
parser.add_argument('--wandb',action='store_true',default=False,help='Flag if using Weights and Biases to log.')
parser.add_argument('--twod',action='store_true',default=False,help='Flag to only use 2D conformers for making the distance matrix.')
parser.add_argument('--skip_train',action='store_true',help='Flag to skip training, and jump right into evaluations.')
parser.add_argument('--seed',type=int,default=420,help='Random seed for training the models.')

args=parser.parse_args()

if args.cpu:
    from featurization.cpu_data_utils import load_data_from_df, construct_loader
else:
    from featurization.data_utils import load_data_from_df, construct_loader
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

assert args.loss in set(['mse','mae','huber','logcosh']) and args.optimizer in set(['sgd','adam'])


#checking that the correct train/test file pointers are set correctly.
if args.trainfile:
    assert (bool(args.testfile)),'Need to set --testfile when trainfile is set'
elif args.prefix:
    assert (bool(args.fold)),'Need to set --fold when prefix is set'
else:
    print('Need to set either trainfile/testfile OR prefix/fold!')
    sys.exit(1)

#now that the input from the arguments is sanitized, we can proceed with the script
if args.trainfile:
    trainfile=args.trainfile
    testfile=args.testfile
    if args.cpu:
        outf_prefix=f'aqsol_test_ind_cpu_drop{args.dropout}_ldist{args.ldist}_lattn{args.lattn}_Ndense{args.Ndense}_heads{args.heads}_dmodel{args.dmodel}_nsl{args.nstacklayers}'
    else:
        outf_prefix=f'aqsol_test_ind_drop{args.dropout}_ldist{args.ldist}_lattn{args.lattn}_Ndense{args.Ndense}_heads{args.heads}_dmodel{args.dmodel}_nsl{args.nstacklayers}'
else:
    trainfile=args.prefix+'_train'+args.fold+'.csv'
    testfile=args.prefix+'_test'+args.fold+'.csv'
    namep=args.prefix.split('/')[-1]
    if args.cpu:
        outf_prefix=f'{namep}_cpu_{args.fold}_drop{args.dropout}_ldist{args.ldist}_lattn{args.lattn}_Ndense{args.Ndense}_heads{args.heads}_dmodel{args.dmodel}_nsl{args.nstacklayers}'
    else:
        outf_prefix=f'{namep}_{args.fold}_drop{args.dropout}_ldist{args.ldist}_lattn{args.lattn}_Ndense{args.Ndense}_heads{args.heads}_dmodel{args.dmodel}_nsl{args.nstacklayers}'

#wandb things
if args.wandb:
    #wandb.init(project='MAT',name=outf_prefix)  #this was from running the sweeps
    wandb.init(project='mat_independent_set_tests',name=outf_prefix)

print('Trainfile:',trainfile)
print('Testfile:',testfile)
print('Outfile Prefix:',outf_prefix)
print('Loading train and test data')

#setting the specified random seed.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#loading the training & testing data
batch_size=8
trainX, trainy=load_data_from_df(trainfile,one_hot_formal_charge=True,two_d_only=args.twod)
data_loader=construct_loader(trainX,trainy,batch_size)
testX, testy=load_data_from_df(testfile,one_hot_formal_charge=True,two_d_only=args.twod)
testdata_loader=construct_loader(testX,testy,batch_size)


d_atom = trainX[0][0].shape[1]

model_params= {
    'd_atom': d_atom,
    'd_model': args.dmodel,
    'N': args.nstacklayers,
    'h': args.heads,
    'N_dense': args.Ndense,
    'lambda_attention': args.lattn, 
    'lambda_distance': args.ldist,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': args.dropout,
    'aggregation_type': 'mean'
}

print('Making Model')

model=make_model(**model_params)
param_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters:',param_count)

if args.wandb:
    wandb.watch(model,'all')
    wandb.log({'Parameters':param_count},step=0)

if args.pretrain:
    print('Using Pretrained Weights')
    pretrained_name = 'pretrained_weights.pt'  # This file should be downloaded first (See README.md).
    pretrained_state_dict = torch.load(pretrained_name)
    model_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if 'generator' in name:
             continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

if args.cpu:
    model.cpu()
else:
    torch.cuda.empty_cache()
    model.cuda()
model.train()

#TODO -- extra loss functions -- quantile?
if args.loss == 'mse':
    criterion=torch.nn.MSELoss(reduction='mean')
elif args.loss=='mae':
    criterion=torch.nn.L1Loss(reduction='mean')
elif args.loss=='huber':
    criterion=torch.nn.SmoothL1Loss(reduction='mean')
elif args.loss=='logcosh':
    criterion=None


#Selecting Optimizer
if args.optimizer=='sgd':
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)#,dampening=args.dampening,nesterov=args.nesterov)
elif args.optimizer=='adam':
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)#betas=(args.beta1,args.beta2),eps=args.epsilon,amsgrad=args.amsgrad)#default adam has lr=0.001

losses=[]

if not args.skip_train:
    print('Training')
    #Training loop
    iteration=0
    for epoch in range(args.epochs):
        epoch_preds=np.array([])
        epoch_gold=np.array([])
        for batch in data_loader:
            iteration+=1
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

            #accumulate the epoch training datas
            epoch_gold=np.append(epoch_gold,y.tolist())
            epoch_preds=np.append(epoch_preds,y_pred.tolist())
            
            if criterion==None and args.loss=='logcosh':
                loss=logcosh_loss(y_pred,y)
            else:
                loss=criterion(y_pred,y)
            losses.append(loss.item())
            if args.wandb:
                wandb.log({"Train Loss":loss.item()},step=iteration)

            optimizer.zero_grad()
            loss.backward()
            
            #implementing gradient clipping -- First trying a clip of 10
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)

            optimizer.step()

            if not args.cpu:
                torch.cuda.empty_cache()

            if iteration%1000==0:
                #we evaluate the test set.
                model.eval()
                gold=np.array([])
                preds=np.array([])

                for t_batch in testdata_loader:
                    t_adjacency_matrix, t_node_features, t_distance_matrix, t_y = t_batch
                    gold=np.append(gold,t_y.tolist())
                    t_batch_mask = torch.sum(torch.abs(t_node_features), dim=-1) != 0
                    t_y_pred = model(t_node_features, t_batch_mask, t_adjacency_matrix, t_distance_matrix, None)
                    preds=np.append(preds,t_y_pred.tolist())
                    if not args.cpu:
                        torch.cuda.empty_cache()

                test_rmse=np.sqrt(np.mean((preds-gold)**2))
                test_r2=np.corrcoef(preds,gold)[0][1]**2
                model.train()
                if args.wandb:
                    wandb.log({"Test RMSE":test_rmse,"Test R2":test_r2},step=iteration)

        #end of 1 epoch -- time to log the stats
        train_rmse, train_r2=(np.sqrt(np.mean((epoch_preds-epoch_gold)**2)), np.corrcoef(epoch_preds,epoch_gold)[0][1]**2)
        if args.wandb:
            wandb.log({"Train Epoch RMSE":train_rmse,"Train Epoch R2":train_r2},step=iteration)

    #now that the training is complete -- we need to output the training losses
    epoch_mean_losses=[]
    for i in range(int(len(losses)/len(data_loader))):
        tmp=[]
        for j in range(int(len(data_loader))):
            tmp.append(losses[int(j+i*len(losses)/len(data_loader))])
        epoch_mean_losses.append(np.mean(tmp))

    print('Training Complete!')
    #saving the trained model
    if args.savemodel:
        print('Saving Model:',args.datadir+'/'+outf_prefix+'_trained.model')
        torch.save(model.state_dict(),args.datadir+'/'+outf_prefix+'_trained.model')

    #making the directory for the data
    if not os.path.isdir(args.datadir):
        os.mkdir(args.datadir)

    #saving the training losses
    with open(args.datadir+'/'+outf_prefix+'_trainlosses.pi','wb') as outfile:
        pickle.dump(losses,outfile)
    with open(args.datadir+'/'+outf_prefix+'_trainepochlosses.pi','wb') as outfile:
        pickle.dump(epoch_mean_losses,outfile)

#final evaluations
print('Final Evaluations!')
print('Training Set:')
model.eval()

#first the training set
gold=np.array([])
preds=np.array([])
t0=time.time()
train_times=[]
for batch in data_loader:
    t1=time.time()
    adjacency_matrix, node_features, distance_matrix, y = batch
    tload=time.time()-t1
    gold=np.append(gold,y.tolist())
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
    tpred=time.time()-t1
    preds=np.append(preds,y_pred.tolist())
    if not args.cpu:
        torch.cuda.empty_cache()

    train_times.append((tload,tpred))

ttime=time.time()-t0
print('Overall Time: ',ttime)
if args.wandb:
    wandb.log({'Train Eval Time':ttime},step=iteration+1)

r2=np.corrcoef(preds,gold)[0][1]**2
rmse=np.sqrt(np.mean((preds-gold)**2))
if args.wandb:
    wandb.log({"Train Epoch RMSE":rmse,"Train Epoch R2":r2},step=iteration+1)

#we need to evaluate the test_set
print('Test Set:')
gold=np.array([])
preds=np.array([])
t0=time.time()
test_times=[]
for batch in testdata_loader:
    t1=time.time()
    adjacency_matrix, node_features, distance_matrix, y = batch
    tload=time.time()-t1
    gold=np.append(gold,y.tolist())
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
    tpred=time.time()-t1
    preds=np.append(preds,y_pred.tolist())
    if not args.cpu:
        torch.cuda.empty_cache()

    test_times.append((tload,tpred))

ttime=time.time()-t0
print('Overall Time: ',ttime)
if args.wandb:
    wandb.log({'Test Eval Time':ttime},step=iteration+1)

r2=np.corrcoef(preds,gold)[0][1]**2
rmse=np.sqrt(np.mean((preds-gold)**2))

testdic={
    'predicted':preds,
    'target':gold,
    'RMSE':rmse,
    'R2':r2,
    'Traintimes':train_times,
    'Testtimes':test_times
}

print('Test RMSE:',rmse)
print('Test R2  :',r2)

if args.wandb:
    wandb.log({"Test RMSE":rmse,"Test R2":r2},step=iteration+1)

with open(args.datadir+'/'+outf_prefix+'_testdic.pi','wb') as outfile:
    pickle.dump(testdic,outfile)
