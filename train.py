from __future__ import print_function
from datetime import date
import os
import sys
import argparse
import random
import pickle
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from utils import maskedMSE, maskedMSETest

from torch_geometric.data import Data, DataLoader

from stp_r_model import STP_R_Net
from stp_g_model import STP_G_Net
from stp_gr_model import STP_GR_Net
from mtp_gr_model import MTP_GR_Net

from stp_gr_dataset import STP_GR_Dataset
from mtp_gr_dataset import MTP_GR_Dataset

import math
import time

def train_a_model(model_to_tr, num_ep=1):
    model_to_tr.train()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    train_running_loss = 0.0
    for i, data in enumerate(trainDataloader):
        # down-sampling data
        data.x =  data.x[:, ::2, :] 
        data.y =  data.y[:,4::5,:] 
    
        optimizer.zero_grad()
        # forward + backward + optimize
        fut_pred = model_to_tr(data.to(args['device']))
        
        op_mask = torch.ones(data.y.shape)
        train_l = maskedMSE(fut_pred, data.y, op_mask)

        train_l.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 10)
        optimizer.step()
        train_running_loss += train_l.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('ep {}, {} batches, {} - {}'.format( num_ep, i + 1, 'maskedMSE', round(train_running_loss / 1000, 4)))
            train_running_loss = 0.0
    scheduler.step()
    return round(train_running_loss / (i%1000), 4)

def val_a_model(model_to_val):
    model_to_val.eval()
    lossVals = torch.zeros(10)
    counts = torch.zeros(10)
    with torch.no_grad():
        print('Testing no grad')
        # val_running_loss = 0.0
        for i, data in enumerate(valDataloader):
            # down-sampling data
            data.x =  data.x[:, ::2, :] 
            data.y =  data.y[:,4::5,:] 

            # predict
            fut_pred = model_to_val(data.to(args['device']))

            # calculate loss
            fut_pred = fut_pred.permute(1,0,2)
            ff = data.y.permute(1,0,2)
            
            op_mask = torch.ones(ff.shape)
            l, c = maskedMSETest(fut_pred, ff, op_mask)

            lossVals +=l.detach()
            counts += c.detach()

    print(torch.pow(lossVals / counts,0.5) *0.3048)   

    return torch.pow(lossVals / counts,0.5) *0.3048

def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    

    # command line arguments
    pp = pprint.PrettyPrinter(indent=1)
    def parse_args(cmd_args):
        """ Parse arguments from command line input
        """
        parser = argparse.ArgumentParser(description='Training parameters')
        parser.add_argument('-g', '--gnn', type=str, default='GAT', help="the GNN to be used")
        parser.add_argument('-r', '--rnn', type=str, default='GRU', help="the RNN to be used")
        parser.add_argument('-m', '--modeltype', type=str, default='GR', help="the model type [R, G, GR]")
        parser.add_argument('-b', '--histlength', type=int, default=30, help="length of history 10, 30, 50")
        parser.add_argument('-f', '--futlength', type=int, default=10, help="length of future 50")
        parser.add_argument('-k', '--gpu', type=str, default='0', help="the GPU to be used")
        parser.add_argument('-i', '--number', type=int, default=0, help="run times of the py script")

        parser.set_defaults(render=False)
        return parser.parse_args(cmd_args)

    # Parse arguments
    cmd_args = sys.argv[1:]
    cmd_args = parse_args(cmd_args)

    ## Network Arguments
    args = {}
    args['run_i'] = cmd_args.number
    args['random_seed'] = 1
    args['input_embedding_size'] = 16 # if args['single_or_multiple'] == 'single_tp' else 32
    args['encoder_size'] = 32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['decoder_size'] = 64 # if args['single_or_multiple'] == 'single_tp' else 128 # 128 256
    args['dyn_embedding_size'] = 32 # if args['single_or_multiple'] == 'single_tp' else 64 # 64 128
    args['train_epoches'] = 50
    args['num_gat_heads'] = 3
    args['concat_heads'] = True # False # True
    
    args['in_length'] = cmd_args.histlength
    args['out_length'] = cmd_args.futlength
    
    args['single_or_multiple'] = 'multiple_tp' # or multiple_tp single_tp
    args['date'] = date.today().strftime("%b-%d-%Y")
    args['batch_size'] = 16 if args['single_or_multiple'] == 'single_tp' else 128
    args['net_type'] = cmd_args.modeltype
    args['enc_rnn_type'] = cmd_args.rnn # LSTM GRU
    args['gnn_type'] = cmd_args.gnn # GCN GAT
    
    device = torch.device('cuda:{}'.format(cmd_args.gpu) if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # set random seeds
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    if device != 'cpu':
        print('running on {}'.format(device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        print('seed setted! {}'.format(args['random_seed']))
    




    # Initialize network
    if args['net_type'] == 'GR':
        if args['single_or_multiple'] == 'single_tp':
            print('loading {} model'.format(args['net_type']))
            train_net = STP_GR_Net(args)
        elif args['single_or_multiple'] == 'multiple_tp':
            print('loading {} model'.format(args['net_type']))
            train_net = MTP_GR_Net(args)
    elif args['net_type'] == 'R':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_R_Net(args)
    elif args['net_type'] == 'G':
        print('loading {} model'.format(args['net_type']))
        train_net = STP_G_Net(args)
    else:
        print('\nselect a proper model type!\n')
    train_net.to(args['device'])

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))
    print(train_net)
    pp.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))
    
    # for name, param in train_net.named_parameters():
    #     if param.requires_grad:
    #         print(name,':',param.size())

    ## Initialize optimizer and the the 
    optimizer = torch.optim.Adam(train_net.parameters(),lr=0.001) 
    scheduler = MultiStepLR(optimizer, milestones=[1], gamma=1.0)
    if args['single_or_multiple'] == 'multiple_tp':
        optimizer = torch.optim.Adam(train_net.parameters(),lr=0.004) # lr 0.0035, batch_size=4 or 8.
        scheduler = MultiStepLR(optimizer, milestones=[1,2,3,6,20,30], gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6], gamma=1.0)
    
    if args['single_or_multiple'] == 'single_tp':
        train_set = STP_GR_Dataset(data_path='/home/xy/stp_data_2021_train', scenario_names=[
                                                                                    'stp0750am-0805am', 
                                                                                    'stp0805am-0820am',
                                                                                    'stp0820am-0835am', 
                                                                                    ]) 
        val_set = STP_GR_Dataset(data_path='/home/xy/stp_data_2021_val', scenario_names=[
                                                                                    'stp0750am-0805am', 
                                                                                    'stp0805am-0820am',
                                                                                    'stp0820am-0835am', 
                                                                                    ]) 
    elif args['single_or_multiple'] == 'multiple_tp':
        train_set = MTP_GR_Dataset(data_path='/home/xy/gat_mtp_data_0805am_Train/') # HIST_1w FUT_1w HIST FUT
        val_set = MTP_GR_Dataset(data_path='/home/xy/gat_mtp_data_0805am_Test/')
    
    torch.set_num_threads(4)
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    valDataloader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 1000.0
    for ep in range(1, args['train_epoches']+1):
        train_loss_ep = train_a_model(train_net, num_ep=ep)
        val_loss_ep = val_a_model(train_net)

        Val_LOSS.append(val_loss_ep)
        Train_LOSS.append(train_loss_ep)

        ## save model
        if val_loss_ep[-1]<min_val_loss:
            save_model_to_PATH = './models_itsc2021/{}_{}_{}_{}_h{}f{}_d{}_{}.tar'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                                 args['in_length'], args['out_length'], '3s', args['batch_size'])
            torch.save(train_net.state_dict(), save_model_to_PATH)
            min_val_loss = val_loss_ep[-1]

        with open('./models_itsc2021/{}-{}-{}-{}-h{}f{}-TRAINloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                            args['in_length'], args['out_length'], '3s', args['batch_size']), "w") as file:
            file.write(str(Train_LOSS))
        with open('./models_itsc2021/{}-{}-{}-{}-h{}f{}-VALloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                   args['in_length'], args['out_length'], '3s', args['batch_size']), "w") as file:
            file.write(str(Val_LOSS))
        save_obj_pkl(args, save_model_to_PATH.split('.tar')[0])
    
    torch.save(train_net.state_dict(), save_model_to_PATH)