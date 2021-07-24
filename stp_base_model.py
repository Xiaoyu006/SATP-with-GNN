import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

class STP_Base_Net(torch.nn.Module):
    ''' 
        Shared layers:
            self.ip_emb
            self.enc_rn
            self.dyn_emb
            self.op
            self.leaky_relu

            self.LSTM_Encoder
            self.decode
        '''
    def __init__(self, args):
        super(STP_Base_Net, self).__init__()
        self.args = args
        
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.args['input_embedding_size'])
        # Encoder LSTM
        self.enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # Decoder LSTM
        self.dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        # Output layers:
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
    
    def LSTM_Encoder(self, Hist):
        """ Encode sequential features of all considered vehicles 
            Hist: history trajectory of all vehicles
        """
        _, Hist_Enc = self.enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
        Hist_Enc = self.leaky_relu(self.dyn_emb(self.leaky_relu(Hist_Enc.view(Hist_Enc.shape[1],Hist_Enc.shape[2]))))
        return Hist_Enc
    
    def forward(self, data_pyg):
        raiseNotImplementedError("forward is not implemented in STP_Base_Net!")
        
    def decode(self,enc):
        # print(enc.shape)
        enc = enc.unsqueeze(1)
        # print('enc : {}'.format(enc.shape))
        # print(enc.shape)
        enc = enc.repeat(1, self.args['out_length'], 1)
        # print('enc : {}'.format(enc.shape))
        # print(enc.shape)
        # enc = enc.permute(1,0,2)
        h_dec, _ = self.dec_rnn(enc)
        # print('h_dec shape {}'.format(h_dec.shape))
        # h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        # print(fut_pred.shape)
        # fut_pred = fut_pred.permute(1, 0, 2)
        # print()
        return fut_pred