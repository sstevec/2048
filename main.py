import torch
from torch.utils.tensorboard import SummaryWriter

from Trainer.Config import get_gail_args
from Trainer.MyTrainer import Trainer
from Env.game2048 import Game2048
from Model.MyDiscriminator import Discriminator
from Trainer.AIRL import AIRL
from Trainer.Dataset import SequenceDataset

if __name__ == '__main__':
    # get args
    args = get_gail_args()

    # get policy model
    policy = Trainer(args)
    # trainer.load_checkpoint("Trainer/checkpoint/policy_2474.pth")

    # get game env
    env = Game2048(fill_percent=0.25)

    # get discriminator model
    disc_model = Discriminator(16, 4)
    # disc_check_point = torch.load("Trainer/checkpoint/discriminator_2474.pth")
    # disc_model.load_state_dict(disc_check_point)

    # get the GAIL trainer
    airl_trainer = AIRL(args, env, policy, disc_model)

    # load the expert data
    expert_dataset = SequenceDataset(args.data_dir, 10, "chunk_", ".pt")

    writer = SummaryWriter(log_dir="logs/ppo_training")

    airl_trainer.train(expert_dataset, writer)