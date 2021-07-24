import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from stp_base_model import STP_Base_Net
class STP_G_Net(STP_Base_Net):
    def __init__(self, args):
        super(STP_G_Net, self).__init__(args)
        self.args = args
        
        # GAT layers
        self.gat_conv1 = GATConv(self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        self.gat_conv2 = GATConv(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size']+self.args['encoder_size'], self.args['encoder_size'], heads=self.args['num_gat_heads'], concat=self.args['concat_heads'], dropout=0.0)
        # fully connected
        self.nbrs_fc = torch.nn.Linear(int(self.args['concat_heads'])*(self.args['num_gat_heads']-1)*self.args['encoder_size'] + self.args['encoder_size'], 1*self.args['encoder_size'])
        
        # Decoder LSTM
        self.dec_rnn = torch.nn.LSTM(1*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
    
    def GAT_Interaction(self, hist_enc, edge_idx, target_index):
        node_matrix = hist_enc
        # print('hist_enc {}'.format(hist_enc))
        # print('node_matrix {}'.format(node_matrix))
        # print('edge_idx {}'.format(edge_idx))
        # GAT conv
        gat_feature = self.gat_conv1(node_matrix, edge_idx)
        gat_feature = self.gat_conv2(gat_feature, edge_idx)
        # print('gat_feature : {}'.format(gat_feature.shape))

        # get target node's GAT feature
        # print('gat_feature {}'.format(gat_feature.shape))
        # print('target_index {}'.format(target_index.shape))
        target_gat_feature = gat_feature[target_index]

        GAT_Enc = self.leaky_relu(self.nbrs_fc(target_gat_feature))

        return GAT_Enc
    
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
            print('\n\n Only single_tp is supported in R model? \n\n')
        ########################################################################
       
        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        # Interaction
        fwd_tar_GAT_Enc = self.GAT_Interaction(fwd_Hist_Enc, data_pyg.edge_index.long(), target_index)

        # Decode
        fut_pred = self.decode(fwd_tar_GAT_Enc)
        return fut_pred
    