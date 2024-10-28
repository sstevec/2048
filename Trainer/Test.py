import torch

from Config import get_args
from Trainer import Trainer
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
        current_state = self.env.get_current_state()
        previous_state = self.observation[self.next_index:self.next_index + 5, :]
        input_state = np.concatenate((previous_state, np.expand_dims(current_state, axis=0)), axis=0)

        output = self.trainer.model(torch.tensor(input_state, device=self.device, dtype=torch.long).unsqueeze(0))
        index = torch.argmax(output).item()

        self.env.step(index)
        self.env.show()



if __name__ == '__main__':
    # start training
    args = get_args()

    trainer = Trainer(args)

    # load model
    last_epoch_num = trainer.load_latest_checkpoint()

    env = Game2048(fill_percent=0.3)

    env.show()

    runner = Runner(trainer, env)
    for i in range(1000):
        runner.generate_next_action()
        if not env.check_alive():
            break
