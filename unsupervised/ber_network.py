import os
import random

import numpy as np
import torch
from torch import nn

from unsupervised.autoencoder import Decoder

ACTIVATION = nn.ReLU()


class DecoderBer(nn.Module):
    def __init__(self,
                 code_dim,
                 hidden_dim=20,
                 output_dim=25,
                 drop_prob=0.1,
                 norm_layer=nn.BatchNorm1d,
                 activation=ACTIVATION,
                 ):
        super(DecoderBer, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.output_dim = output_dim
        self.dec_a = Decoder(code_dim, hidden_dim=hidden_dim, output_dim=output_dim, drop_prob = drop_prob, norm_layer=norm_layer, activation=activation)
        self.dec_b = Decoder(code_dim, hidden_dim=hidden_dim, output_dim=output_dim, drop_prob = drop_prob, norm_layer=norm_layer, activation=activation)

    def forward(self, x, y):
        """
        Concats bio and batch codes, and does forward propagation to obtain recon
        """
        mask0 = y.int() == 0
        mask1 = y.int() == 1
        # prediction = torch.zeros((y.shape[0], self.output_dim))
        # x = self.modulation1(x, style)
        y_0 = 1 * torch.tanh(0.1 * self.dec_a(x[mask0]))
        y_1 = 1 * torch.tanh(0.1 * self.dec_b(x[mask1]))
        # prediction[mask0] = y_0
        # prediction[mask1] = y_1

        return y_0, y_1

    def save(self, path):
        state_dict = self.state_dict()
        weights_path = os.path.join(path, "decode_ber.pt")
        torch.save(state_dict, weights_path)


class Net(nn.Module):
    def __init__(self,
                 encoder,
                 decoder
                 ):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x,y):
        code = self.encoder(x)
        return self.decoder(code,y)

    def save(self, path):
        state_dict = self.state_dict()
        weights_path = os.path.join(path, "net.pt")
        torch.save(state_dict, weights_path)

    def from_pretrain(self,path):
        weight_path = os.path.join(path,"net.pt")
        state_dict = torch.load(weight_path)
        self.load_state_dict(state_dict)
