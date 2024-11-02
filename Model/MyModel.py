import torch.nn as nn

class ExpertLearningModel(nn.Module):
    def __init__(self, embedder, transformer1, resnet, transformer2, fnn):
        super(ExpertLearningModel, self).__init__()
        self.embedding = embedder
        self.resnet = resnet
        self.transformer1 = transformer1
        self.transformer2 = transformer2

        self.fnn = fnn

    def forward(self, x):
        # input shape [batch, sequence, input_dim]
        x = self.embedding(x)
        # output of embedding [batch, sequence, input_dim, embed_dim]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # get [batch, sequence, input_dim * embed_dim]
        x = x.transpose(1, 2)
        # resnet take [batch, new_input_dim, sequence]
        x = self.resnet(x)
        # output shape is [batch, output_dim_resnet]

        # input shape [batch, sequence, new_input_dim]
        x = self.transformer1(x)

        x = self.transformer2(x)
        # the output here is [batch, seq, hidden_dim]

        # we only interest in the last sequence, as the decision is primary made based on this one,
        # so we dropped the previous steps and only decode the last one
        x = self.fnn(x[:, -1, :])

        return x
