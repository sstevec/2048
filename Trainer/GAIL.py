import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from Trainer import Config
from Env import Game2048
from Trainer.MyTrainer import Trainer
from Model.MyDiscriminator import Discriminator
from Trainer.Dataset import SequenceDataset

class GAIL:
    def __init__(self, args, env, policy_wrapper, discriminator_model):
        self.args = args
        self.env = env
        self.policy_network = policy_wrapper.model
        self.discriminator_model = discriminator_model

        self.disc_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.disc_lr)
        self.policy_optimizer = policy_wrapper.optimizer

        self.device = policy_wrapper.device

    def train(self, expert_dataset):
        self.policy_network.to(self.device)
        self.discriminator_model.to(self.device)

        for epoch in tqdm(range(self.args.num_epochs), desc="Epochs", position=0):
            # Step 1: Generate agent's trajectories
            self.policy_network.eval()
            agent_obs, agent_actions = [], []
            obs = self.env.reset()
            for _ in range(self.args.self_play_steps):
                # Sample action from the current policy
                action, _ = self.policy_network.sample_action(torch.from_numpy(obs).to(self.device).to(torch.long))

                # Step in the environment
                action = action.cpu().item()
                next_obs, _, done = self.env.step(action)

                # Store agent's trajectory
                agent_obs.append(obs)
                agent_actions.append(action)

                obs = next_obs
                if done:
                    obs = self.env.reset()

            # Convert lists to tensors
            agent_obs = torch.from_numpy(np.array(agent_obs)).float()
            agent_actions = torch.from_numpy(np.array(agent_actions))

            # Step 2: Train Discriminator
            self.discriminator_model.train()
            disc_total_loss = 0

            expert_obs, expert_actions = expert_dataset.random_sample(self.args.self_play_steps)
            for _ in range(self.args.disc_train_epochs):
                # Sample expert and agent batches
                expert_indices = torch.randint(0, len(expert_obs), (self.args.batch_size,))
                agent_indices = torch.randint(0, len(agent_obs), (self.args.batch_size,))

                expert_obs_batch = expert_obs[expert_indices].to(self.device)
                expert_actions_batch = expert_actions[expert_indices].squeeze()
                agent_obs_batch = agent_obs[agent_indices].to(self.device)
                agent_actions_batch = agent_actions[agent_indices]

                expert_actions_batch = F.one_hot(expert_actions_batch, num_classes=4).float().to(self.device)
                agent_actions_batch = F.one_hot(agent_actions_batch, num_classes=4).float().to(self.device)

                # Discriminator predictions
                expert_preds = self.discriminator_model(expert_obs_batch, expert_actions_batch)
                agent_preds = self.discriminator_model(agent_obs_batch, agent_actions_batch)

                # Discriminator loss: maximize log D(expert) + log(1 - D(agent))
                expert_loss = F.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds, device=self.device))
                agent_loss = F.binary_cross_entropy(agent_preds, torch.zeros_like(agent_preds, device=self.device))
                disc_loss = expert_loss + agent_loss

                # Update discriminator
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()

                disc_total_loss += disc_loss.item()
            # take the average loss
            disc_total_loss /= self.args.disc_train_epochs

            # Step 3: Update Policy with PPO using discriminator rewards
            self.discriminator_model.eval()
            self.policy_network.train()
            for _ in range(self.args.policy_update_steps):
                # Compute discriminator rewards for agent trajectories
                agent_obs = agent_obs.to(self.device)
                agent_actions = agent_actions.to(self.device)
                agent_rewards = -torch.log(
                    self.discriminator_model(agent_obs, F.one_hot(agent_actions, num_classes=4).float()) + 1e-8).squeeze()

                # Compute policy loss
                action_logits, values = self.policy_network(agent_obs.to(torch.long))
                action_probs = F.softmax(action_logits, dim=-1)
                selected_action_probs = action_probs[range(len(agent_actions)), agent_actions]

                advantages = agent_rewards - values.detach().squeeze()
                policy_loss = -(advantages * torch.log(selected_action_probs)).mean()  # Policy gradient
                value_loss = F.mse_loss(values.squeeze(), agent_rewards)  # Value function loss

                total_loss = policy_loss + value_loss
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()

            if (epoch + 1) % self.args.save_freq == 0:
                print(f"Epoch {epoch + 1}/{self.args.num_epochs} - Discriminator Loss: {disc_total_loss}")
                torch.save(self.policy_network.state_dict(), f'{self.args.checkpoint_dir}/policy_{epoch}.pth')
                torch.save(self.discriminator_model.state_dict(), f'{self.args.checkpoint_dir}/discriminator_{epoch}.pth')


if __name__ == '__main__':
    # get args
    args = Config.get_gail_args()

    # get policy model
    trainer = Trainer(args)
    # trainer.load_checkpoint("./checkpoint/35.pth")

    # get game env
    env = Game2048(fill_percent=0.3)

    # get discriminator model
    disc_model = Discriminator(16, 4)

    # get the GAIL trainer
    gail_trainer = GAIL(args, env, trainer, disc_model)

    # load the expert data
    expert_dataset = SequenceDataset("./Data/processed_data", 1, "chunk_", ".pt")

    gail_trainer.train(expert_dataset)