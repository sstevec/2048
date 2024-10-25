import torch.nn as nn

class ExpertLearningModel(nn.Module):
    def __init__(self, embedder, transformer, resnet, lstm, fnn, device, batch_size):
        super(ExpertLearningModel, self).__init__()
        self.embedding = embedder
        self.transformer = transformer
        self.resnet = resnet
        self.lstm = lstm

        lstm_hidden, lstm_c = self.lstm.init_hidden(batch_size)

        self.fnn = fnn

        self.device = device

        self.lstm_hidden = lstm_hidden.to(device)
        self.lstm_c = lstm_c.to(device)


    def forward(self, x, is_train=True):
        # input shape [batch, sequence, input_dim]
        x = self.embedding(x)
        # output of embedding [batch, sequence, input_dim, embed_dim]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # get [batch, sequence, input_dim * embed_dim]

        # input shape [batch, sequence, new_input_dim]
        x = self.transformer(x)
        # output of transformer [batch, sequence, new_input_dim]

        x = x.transpose(1, 2)
        # resnet take [batch, new_input_dim, sequence]
        x = self.resnet(x)
        # output shape is [batch, output_dim_resnet]

        x, hidden, c = self.lstm(x, (self.lstm_hidden, self.lstm_c))

        # only update the memory when it is training
        if is_train:
            self.lstm_hidden = hidden.clone().detach()
            self.lstm_c = c.clone().detach()

        x = self.fnn(x)

        return x
