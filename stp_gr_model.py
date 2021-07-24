import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from stp_g_model import STP_G_Net
class STP_GR_Net(STP_G_Net):
    def __init__(self, args):
        super(STP_GR_Net, self).__init__(args)
        self.args = args
        # # Input embedding layer
        # self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        # # Encoder LSTM
        # # self.enc_rnn = torch.nn.LSTM(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # # # Vehicle dynamics embedding
        # self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # # GAT layers
        # self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # # fully connected
        # self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        # # Decoder LSTM
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        # # Output layers:
        # self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        # # Activations:
        # self.leaky_relu = torch.nn.LeakyReLU(0.1)
    
    # def LSTM_Encoder(self, Hist):
    #     """ Encode sequential features of all considered vehicles 
    #         Hist: history trajectory of all vehicles
    #     """
    #     _, Hist_Enc = self.enc_rnn(self.ip_emb(Hist))
    #     Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
    #     return Hist_Enc
    
    # def GAT_Interaction(self, hist_enc, edge_idx, target_index):
    #     node_matrix = hist_enc
    #     # print('hist_enc {}'.format(hist_enc))
    #     # print('node_matrix {}'.format(node_matrix))
    #     # print('edge_idx {}'.format(edge_idx))
    #     # GAT conv
    #     gat_feature = self.gat_conv1(node_matrix, edge_idx)
    #     gat_feature = self.gat_conv2(gat_feature, edge_idx)
    #     # print('gat_feature : {}'.format(gat_feature.shape))

    #     # get target node's GAT feature
    #     # print('gat_feature {}'.format(gat_feature.shape))
    #     # print('target_index {}'.format(target_index.shape))
    #     target_gat_feature = gat_feature[target_index]

    #     GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

    #     return GAT_Enc
    
    def forward(self, data_pyg):
        
        # get target vehicles' index first
        ########################################################################
        # for single TP
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        # elif self.args['single_or_multiple'] == 'multiple_tp':
        #     target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0:data_pyg.num_target_v[i]]) for i in range(data_pyg.num_graphs)]
        #     target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n single TP or multiple TP? \n\n')
        ########################################################################
       
        # get target vehicles' index first
        ########################################################################
        # for multi TP
        # target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0:data_pyg.num_target_v[i]]) for i in range(data_pyg.num_graphs)]
        # target_index = torch.cat(target_index, dim=0)
        ########################################################################
        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        # Interaction
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)

        # get the lstm features of target vehicles
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]

        # Combine Individual and Interaction features
        enc = torch.cat((fwd_tar_LSTM_Enc, fwd_tar_GAT_Enc), 1)
        # Decode
        fut_pred = self.decode(enc)
        return fut_pred

    # def decode(self,enc):
    #     # print(enc.shape)
    #     enc = enc.unsqueeze(1)
    #     # print('enc : {}'.format(enc.shape))
    #     # print(enc.shape)
    #     enc = enc.repeat(1, self.args['out_length'], 1)
    #     # print('enc : {}'.format(enc.shape))
    #     # print(enc.shape)
    #     # enc = enc.permute(1,0,2)
    #     h_dec, _ = self.dec_rnn(enc)
    #     # print('h_dec shape {}'.format(h_dec.shape))
    #     # h_dec = h_dec.permute(1, 0, 2)
    #     fut_pred = self.op(h_dec)
    #     # print(fut_pred.shape)
    #     # fut_pred = fut_pred.permute(1, 0, 2)
    #     # print()
    #     return fut_pred