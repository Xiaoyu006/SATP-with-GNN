import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from stp_base_model import STP_Base_Net
class STP_R_Net(STP_Base_Net):
    def __init__(self, args):
        super(STP_R_Net, self).__init__(args)
        self.dec_rnn = torch.nn.LSTM(self.args['dyn_embedding_size'], self.args['decoder_size'], 2, batch_first=True)

    def forward(self, data_pyg):
        
        # get target vehicles' index first
        ########################################################################
        # for single TP
        if self.args['single_or_multiple'] == 'single_tp':
            target_index = [torch.flatten((data_pyg.batch==i).nonzero()[0]) for i in range(data_pyg.num_graphs)]
            target_index = torch.cat(target_index, dim=0)
        else:
            print('\n\n Only single_tp is supported in R model? \n\n')
        ########################################################################
       
        # Encode
        fwd_Hist_Enc = self.LSTM_Encoder(data_pyg.x)
        
        # get the lstm features of target vehicles
        fwd_tar_LSTM_Enc = fwd_Hist_Enc[target_index]

        # Decode
        fut_pred = self.decode(fwd_tar_LSTM_Enc)
        return fut_pred