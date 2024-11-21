from Trainer import Config
from Trainer.MyTrainer import Trainer
from Env.game2048 import Game2048
from Model.MyDiscriminator import Discriminator
from Trainer.GAIL import GAIL
from Trainer.Dataset import SequenceDataset

if __name__ == '__main__':
    # get args
    args = Config.get_gail_args()

    # get policy model
    trainer = Trainer(args)

    # get game env
    env = Game2048(fill_percent=0.3)

    # get discriminator model
    disc_model = Discriminator(16, 4)

    # get the GAIL trainer
    gail_trainer = GAIL(args, env, trainer, disc_model)

    # load the expert data
    expert_dataset = SequenceDataset("./Data/processed_data", 1, "chunk_", ".pt")

    gail_trainer.train(expert_dataset)