import torch

from Trainer.Config import get_args
from Trainer.MyTrainer  import Trainer
from Env.game2048 import Game2048
import numpy as np

class Runner():
    def __init__(self, trainer, env):
        self.trainer = trainer
        self.env = env
        self.observation = np.zeros((4, 16))
        self.action = np.array([])
        self.next_index = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer.model.to(self.device)

    def generate_next_action(self):
        current_state = self.env.get_current_state().reshape(1, 16)
        self.observation = np.vstack((self.observation, current_state))

        seq = self.observation[self.next_index:self.next_index + 6, :]

        output = self.trainer.model(torch.tensor(seq, device=self.device, dtype=torch.long).unsqueeze(0))
        index = torch.argmax(output).item()

        self.next_index += 1

        self.env.step(index)
        self.env.show()



if __name__ == '__main__':
    # start training
    args = get_args()

    trainer = Trainer(args)

    # load model
    # last_epoch_num = trainer.load_latest_checkpoint()
    trainer.load_checkpoint("./checkpoint/100_no_shuffle.pth")

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
            if stack_counter > 15:
                break
        else:
            last_score = env.score
            stack_counter = 0
