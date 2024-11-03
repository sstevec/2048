import torch.nn as nn

class ExpertLearningModel(nn.Module):
    def __init__(self, embedder, resnet, positional_encoding, transformer1, transformer2, fnn):
        super(ExpertLearningModel, self).__init__()
        self.embedding = embedder
        self.resnet = resnet
        self.positional_encoding = positional_encoding.permute(1, 2, 0)
        self.positional_encoding.requires_grad = False

        self.transformer1 = transformer1
        self.transformer2 = transformer2

        self.fnn = fnn

    def forward(self, x):
        # input shape [batch, input_dim]
        x = self.embedding(x)
        # output of embedding [batch, 16, embed_dim]
        b, _, e = x.shape
        x = x.reshape(b, 4, 4, e)

        x = x.permute(0, 3, 1, 2)

        # resnet take [batch, embed_dim, row, col]
        x = self.resnet(x)
        # output shape is [batch, out_dim, out_dim, hidden_channel]

        x = x + self.positional_encoding

        _, r, c, hc = x.shape
        x = x.reshape(b, r * c, hc)

        # input shape [batch, 9, hidden_dim]
        x = self.transformer1(x)

        x = self.transformer2(x)
        # the output here is [batch, 9, hidden_dim]

        x = self.fnn(x.reshape(b, -1))

        return x
