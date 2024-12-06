import torch.nn as nn
import torch
import torch.nn.functional as F

from Model import ResNet, ResidualBlock, CustomFNN


class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Discriminator, self).__init__()
        self.hidden_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.obs_resnet = ResNet(ResidualBlock, [2, 2, 2], 1, 64, self.hidden_dim)
        self.obs_flat = nn.Flatten()

        self.act_fnn = CustomFNN([action_dim, 64, self.hidden_dim], self.device)

        self.combined_fnn = CustomFNN(
            [(3 * 3 + 1) * self.hidden_dim, 4 * self.hidden_dim, 2 * self.hidden_dim, self.hidden_dim, 1],
            self.device)

    def forward(self, obs, action, policy_log_probs):
        # Reshape observation to (batch_size, 1, 4, 4)
        obs = obs.view(-1, 1, 4, 4)
        # Extract spatial features
        spatial_features = self.obs_resnet(obs)
        spatial_features = self.obs_flat(spatial_features)

        # Process action
        action_features = self.act_fnn(action.squeeze(1))

        combined = torch.cat([spatial_features, action_features], dim=1)
        f = self.combined_fnn(combined).squeeze(-1)

        # Compute the discriminator output
        d = torch.sigmoid(f - policy_log_probs)

        # Reward function, it is explicitly compute last, do not reuse rewards to compute d, it will mess up the backprop
        rewards = f - policy_log_probs

        return d, rewards
