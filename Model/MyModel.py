import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLearningModel(nn.Module):
    def __init__(self, embedder, resnet, positional_encoding, transformer1, transformer2, fnn, actor_head, critic_head):
        super(ExpertLearningModel, self).__init__()
        self.embedding = embedder
        self.resnet = resnet
        self.positional_encoding = positional_encoding.permute(1, 2, 0)
        self.positional_encoding.requires_grad = False

        self.transformer1 = transformer1
        self.transformer2 = transformer2

        self.fnn = fnn

        self.actor_head = actor_head
        self.critic_head = critic_head

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

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value

    # Sample action is not deterministic, it encourages exploration
    def sample_action(self, x):
        logits, value = self.forward(x.unsqueeze(0))

        action_probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).squeeze(1)

        action_log_probs = F.log_softmax(logits, dim=-1)

        return action, action_log_probs, value