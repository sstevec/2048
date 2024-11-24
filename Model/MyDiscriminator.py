import torch.nn as nn
import torch
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        # Convolutional branch for 4x4 board observation
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (4x4 -> 4x4), 16 filters
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (4x4 -> 4x4), 32 filters
            nn.ReLU(),
            nn.Flatten()  # Flatten to (batch_size, 32 * 4 * 4)
        )
        self.fc_obs = nn.Linear(32 * obs_dim, hidden_dim)  # Fully connected for spatial features

        # Fully connected layer for one-hot encoded action
        self.fc_action = nn.Linear(action_dim, hidden_dim)

        # Combined branch
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Output probability
            nn.Sigmoid()
        )

    def forward(self, obs, action):
        obs = obs.view(-1, 1, 4, 4)  # Reshape observation to (batch_size, 1, 4, 4)
        spatial_features = self.conv(obs)  # Extract spatial features
        spatial_features = self.fc_obs(spatial_features)  # Map to hidden_dim

        action_features = self.fc_action(action)  # Process action

        combined = torch.cat([spatial_features, action_features], dim=1)
        output = self.combined_fc(combined)  # Final classification
        return output
