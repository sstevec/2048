import torch

from Trainer.Config import get_bc_args
from Trainer.MyTrainer import Trainer
from Env.game2048 import Game2048
import numpy as np

class Runner():
    def __init__(self, trainer, env):
        self.trainer = trainer
        self.env = env

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer.model.to(self.device)
        self.trainer.model.eval()

    def generate_next_action(self):
        current_state = self.env.get_current_state()

        output, _ = self.trainer.model(torch.tensor(current_state, device=self.device, dtype=torch.long).unsqueeze(0))
        index = torch.argmax(output).item()

        self.env.step(index)
        self.env.show()



if __name__ == '__main__':
    # start training
    args = get_bc_args()

    trainer = Trainer(args)

    # load model
    # last_epoch_num = trainer.load_latest_checkpoint()
    trainer.load_checkpoint("./checkpoint/policy_99.pth")

    env = Game2048(fill_percent=0.3)

    env.show()

    runner = Runner(trainer, env)

    last_score = 0
    stack_counter = 0
    for i in range(1000):
        runner.generate_next_action()
        if not env.check_alive():
            break
        if env.score == last_score:
            stack_counter += 1
            if stack_counter > 5:
                break
        else:
            last_score = env.score
            stack_counter = 0
