import torch

from Trainer.Config import get_gail_args
from Trainer.MyTrainer import Trainer
from Env.game2048 import Game2048
from Model.MyDiscriminator import Discriminator
from Trainer.GAIL import GAIL
from Trainer.Dataset import SequenceDataset

if __name__ == '__main__':
    # get args
    args = get_gail_args()

    # get policy model
    trainer = Trainer(args)
    # trainer.load_checkpoint("Trainer/checkpoint/policy_399.pth")

    # get game env
    env = Game2048(fill_percent=0.3)

    # get discriminator model
    disc_model = Discriminator(16, 4)
    # disc_check_point = torch.load("Trainer/checkpoint/discriminator_399.pth")
    # disc_model.load_state_dict(disc_check_point)

    # get the GAIL trainer
    gail_trainer = GAIL(args, env, trainer, disc_model)

    # load the expert data
    expert_dataset = SequenceDataset(args.data_dir, 10, "chunk_", ".pt")

    gail_trainer.train(expert_dataset)