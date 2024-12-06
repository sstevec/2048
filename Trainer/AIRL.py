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


class AIRL:
    def __init__(self, args, env, policy_wrapper, discriminator_model):
        self.args = args
        self.env = env
        self.policy_network = policy_wrapper.model
        self.discriminator_model = discriminator_model

        self.disc_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.disc_lr)
        self.policy_optimizer = policy_wrapper.optimizer

        self.device = policy_wrapper.device

    def collect_agent_traj(self):
        self.policy_network.eval()

        best_score, precious_score = 0, 0
        agent_obs, agent_actions, agent_log_probs, agent_value = [], [], [], []
        obs = self.env.reset()
        stuck_counter = 0

        start_index = 0
        # record each game length
        game_length = []
        for i in range(self.args.self_play_steps):
            # Sample action from the current policy
            action, action_log_probs, value = self.policy_network.sample_action(
                torch.from_numpy(obs).to(self.device).to(torch.long))

            # Step in the environment
            action = action.cpu().item()
            next_obs, _, done = self.env.step(action)

            # Store agent's trajectory
            agent_obs.append(obs)
            agent_actions.append(action)
            agent_log_probs.append(action_log_probs[0, action])
            agent_value.append(value.cpu().item())

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

                game_length.append(i + 1 - start_index)
                start_index = i + 1
                stuck_counter = 0
                precious_score = 0

        final_len = self.args.self_play_steps - start_index
        if final_len > 0:
            game_length.append(final_len)
        return agent_obs, agent_actions, agent_log_probs, agent_value, game_length, best_score

    def train_discriminator(self, expert_data, agent_obs, agent_actions, agent_log_probs, best_score):
        self.discriminator_model.train()

        expert_obs, expert_actions = expert_data.random_sample(self.args.self_play_steps,
                                                               int(best_score // 2) + 10)

        expert_actions_one_hot = F.one_hot(expert_actions, num_classes=4).float().to(self.device)
        agent_actions_one_hot = F.one_hot(agent_actions, num_classes=4).float().to(self.device)

        # Compute discriminator loss
        d_expert, _ = self.discriminator_model(expert_obs.to(self.device), expert_actions_one_hot,
                                               torch.zeros(expert_obs.size(0), device=self.device))  # Policy log prob = 0 for experts

        d_agent, _ = self.discriminator_model(agent_obs.to(self.device), agent_actions_one_hot,
                                              agent_log_probs.to(self.device))

        discriminator_loss = -torch.mean(torch.log(d_expert + 1e-8) + torch.log(1 - d_agent + 1e-8))

        self.disc_optimizer.zero_grad()
        discriminator_loss.backward()
        self.disc_optimizer.step()

        return discriminator_loss.item()

    def ppo_update_policy(self, agent_obs, agent_actions, agent_log_probs, agent_value, game_length):
        self.discriminator_model.eval()
        self.policy_network.train()

        agent_actions_one_hot = F.one_hot(agent_actions, num_classes=4).float().to(self.device)

        _, rewards = self.discriminator_model(agent_obs.to(self.device), agent_actions_one_hot,
                                              agent_log_probs.to(self.device))

        # compute advantages and update policy network
        rewards = rewards.detach().cpu()

        # Initialize tensors for returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Compute returns and advantages for each episode
        start_idx = 0

        # we use low gamma to focus on short term reward, because planning is not very necessary
        gamma = 0.9
        # we use high GAE lambda because reward is noisy
        lam = 0.97
        for length in game_length:
            end_idx = start_idx + length

            # Slice data for the current episode
            episode_rewards = rewards[start_idx:end_idx]
            episode_values = agent_value[start_idx:end_idx]

            # Compute returns for this episode
            G = 0
            for t in reversed(range(length)):
                G = episode_rewards[t] + gamma * G
                returns[start_idx + t] = G

            # Compute advantages for this episode
            A = 0
            for t in reversed(range(length)):
                if t < length - 1:
                    next_value = episode_values[t + 1]
                else:
                    next_value = 0
                delta = episode_rewards[t] + gamma * next_value - episode_values[t]
                A = delta + gamma * lam * A
                advantages[start_idx + t] = A

            # Move to the next episode
            start_idx = end_idx

        # Normalize advantages for stability
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(self.device)

        logits_new, values_new = self.policy_network(agent_obs.long().to(self.device))

        log_probs_new = F.log_softmax(logits_new, dim=-1)
        log_probs_selected = log_probs_new[torch.arange(len(agent_actions)), agent_actions]

        policy_loss = -(advantages * log_probs_selected).mean()

        value_loss = F.mse_loss(values_new.squeeze(), returns.to(self.device))

        loss = policy_loss + 0.1 * value_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def train(self, expert_data, writer):
        self.policy_network.to(self.device)
        self.discriminator_model.to(self.device)

        for epoch in tqdm(range(self.args.num_epochs), desc="Epochs", position=0):
            # --------------------  Step 1: Generate agent's trajectories --------------------
            agent_obs, agent_actions, agent_log_probs, agent_value, game_length, best_score = self.collect_agent_traj()

            # Convert lists to tensors
            agent_obs = torch.from_numpy(np.array(agent_obs)).float()
            agent_actions = torch.from_numpy(np.array(agent_actions))
            agent_log_probs = torch.stack(agent_log_probs, dim=0).float()
            agent_value = torch.from_numpy(np.array(agent_value)).float()

            # --------------------  Step 2: Train Discriminator --------------------
            disc_loss = self.train_discriminator(expert_data, agent_obs, agent_actions, agent_log_probs, best_score)

            # --------------------  step 3: Train Policy --------------------
            loss, policy_loss, value_loss = self.ppo_update_policy(agent_obs, agent_actions, agent_log_probs,
                                                                   agent_value, game_length)

            # print(f"Epoch {epoch + 1}/{self.args.num_epochs} - Discriminator Loss: {disc_loss} - "
            #       f"Policy Loss: {loss} - Best score: {best_score}")

            writer.add_scalar("Discriminator Loss", disc_loss, epoch)
            writer.add_scalar("Agent Loss", loss, epoch)
            writer.add_scalar("value Loss", value_loss, epoch)
            writer.add_scalar("policy Loss", policy_loss, epoch)
            writer.add_scalar("Best score", best_score, epoch)

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.policy_network.state_dict(), f'{self.args.checkpoint_dir}/policy_{epoch}.pth')
                torch.save(self.discriminator_model.state_dict(),
                           f'{self.args.checkpoint_dir}/discriminator_{epoch}.pth')


if __name__ == '__main__':
    # get args
    args = Config.get_gail_args()

    # get policy model
    policy = Trainer(args)
    # trainer.load_checkpoint("./checkpoint/35.pth")

    # get game env
    env = Game2048(fill_percent=0.25)

    # get discriminator model
    disc_model = Discriminator(16, 4)

    # get the GAIL trainer
    airl_trainer = AIRL(args, env, policy, disc_model)

    # load the expert data
    expert_dataset = SequenceDataset("./Data/processed_data", 1, "chunk_", ".pt")

    airl_trainer.train(expert_dataset)
