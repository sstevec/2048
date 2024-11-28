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

            best_score, precious_score = 0, 0
            agent_obs, agent_actions = [], []
            obs = self.env.reset()
            stuck_counter = 0
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

                # prevent the model stuck on one stage forever
                if precious_score == self.env.score:
                    stuck_counter += 1
                else:
                    precious_score = self.env.score
                    stuck_counter = 0

                if done or stuck_counter > 10:
                    score = self.env.score
                    if score > best_score:
                        best_score = score
                    obs = self.env.reset()
                    stuck_counter = 0
                    precious_score = 0

            # Convert lists to tensors
            agent_obs = torch.from_numpy(np.array(agent_obs)).float()
            agent_actions = torch.from_numpy(np.array(agent_actions))

            # Step 2: Train Discriminator
            self.discriminator_model.train()
            disc_total_loss = 0
            disc_train_steps = 0

            expert_obs, expert_actions = expert_dataset.random_sample(self.args.self_play_steps,
                                                                      int(best_score // 2) + 10)
            for _ in range(self.args.disc_train_epochs):
                # Sample expert and agent batches
                expert_indices = torch.randint(0, len(expert_obs), (self.args.batch_size,))
                agent_indices = torch.randint(0, len(agent_obs), (self.args.batch_size,))
                expert_obs_batch = expert_obs[expert_indices].to(self.device)
                expert_actions_batch = expert_actions[expert_indices].squeeze()
                agent_obs_batch = agent_obs[agent_indices].to(self.device)
                agent_actions_batch = agent_actions[agent_indices]

                # One-hot encoding actions
                expert_actions_batch = F.one_hot(expert_actions_batch, num_classes=4).float().to(self.device)
                agent_actions_batch = F.one_hot(agent_actions_batch, num_classes=4).float().to(self.device)

                # Discriminator predictions
                expert_preds = self.discriminator_model(expert_obs_batch, expert_actions_batch)
                agent_preds = self.discriminator_model(agent_obs_batch, agent_actions_batch)

                # Label smoothing
                smooth_labels_expert = torch.full_like(expert_preds, 0, device=self.device)
                smooth_labels_agent = torch.full_like(agent_preds, 1, device=self.device)

                # Discriminator loss
                expert_loss = F.binary_cross_entropy_with_logits(expert_preds, smooth_labels_expert)
                agent_loss = F.binary_cross_entropy_with_logits(agent_preds, smooth_labels_agent)
                disc_loss = expert_loss + agent_loss

                # Backpropagation and optimization
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.discriminator_model.parameters(), max_norm=1.0)
                self.disc_optimizer.step()

                # Track total loss for logging
                disc_total_loss += disc_loss.item()

                disc_train_steps += 1
                # if disc loss is too small, we will only train it for 1 step, and skip the rest
                if disc_loss.item() < 0.5:
                    break

            # take the average loss
            disc_total_loss /= disc_train_steps

            # Step 3: Update Policy with PPO using discriminator rewards
            self.discriminator_model.eval()
            self.policy_network.train()
            for _ in range(self.args.policy_update_steps):
                # Compute discriminator rewards for agent trajectories
                agent_obs = agent_obs.to(self.device)
                agent_actions = agent_actions.to(self.device)
                discriminator_outputs = self.discriminator_model(
                    agent_obs, F.one_hot(agent_actions, num_classes=4).float()
                )
                # Compute raw rewards from discriminator
                agent_rewards = -torch.log(discriminator_outputs + 1e-8).squeeze()

                # Forward pass through policy network
                action_logits, values = self.policy_network(agent_obs.to(torch.long))

                # Compute action probabilities and log-probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                log_action_probs = torch.log(action_probs + 1e-8)  # Log-safe softmax

                # Select the log probabilities for actions taken
                selected_log_probs = log_action_probs[range(len(agent_actions)), agent_actions]

                # Compute advantages: (rewards - value estimates)
                advantages = agent_rewards - values.detach().squeeze()

                # Policy loss: Advantage-weighted policy gradient with entropy regularization
                policy_loss = -(advantages * selected_log_probs).mean()

                # Value loss: Mean squared error between predicted values and rewards
                value_loss = F.mse_loss(values.squeeze(), agent_rewards)

                # Total loss (policy loss + value loss)
                total_loss = policy_loss + value_loss

                # Backpropagation and optimization
                self.policy_optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                self.policy_optimizer.step()

            print(f"Epoch {epoch + 1}/{self.args.num_epochs} - Discriminator Loss: {disc_total_loss} - "
                  f"Policy Loss: {total_loss} - Best score: {best_score}")
            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.policy_network.state_dict(), f'{self.args.checkpoint_dir}/policy_{epoch}.pth')
                torch.save(self.discriminator_model.state_dict(),
                           f'{self.args.checkpoint_dir}/discriminator_{epoch}.pth')


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
