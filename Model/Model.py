import numpy as np
import torch
import torch.nn as nn

class ExpertLearningModel(nn.Module):
    def __init__(self, transformer, resnet, lstm, fnn, device, batch_size):
        super(ExpertLearningModel, self).__init__()
        self.transformer = transformer
        self.resnet = resnet
        self.lstm = lstm

        self.lstm_hidden, self.lstm_c = self.lstm.init_hidden(batch_size)

        self.fnn = fnn

        self.device = device


    def forward(self, x, is_train=True):
        # input shape [batch, sequence, 20]
        x = self.transformer(x)
        # output of transformer [batch, sequence, 20]

        x = x.transpose(1, 2)
        # resnet take [batch, 20, sequence]
        x = self.resnet(x)
        # output shape is [batch, output_dim]

        x, hidden, c = self.lstm(x, (self.lstm_hidden, self.lstm_c))

        # only update the memory when it is training
        if is_train:
            self.lstm_hidden = hidden
            self.lstm_c = c

        x = self.fnn(x)

        return x
