import numpy as np
from torch.utils.data import DataLoader

from Trainer.Dataset import SequenceDataset

op_order = ['up', 'right', 'down', 'left']

def show(board, action):
    indent_len = 2
    out_str = ""
    out_str += ("\n" + "-" * (4 * indent_len + 2) + "\n")
    for i in range(4):
        out_str += "|"
        for j in range(4):
            if board[i][j] == 0:
                out_str += (" " * indent_len)
            else:
                val = str(int(board[i][j]))
                out_str += (val + " " * (indent_len - len(val)))
        out_str += "|\n"
    out_str += ("-" * (4 * indent_len + 2))
    print(out_str)
    print(f"last operation: {op_order[action]}")

if __name__ == '__main__':
    train_set = SequenceDataset(directory="./Data", num_chunks=1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    break_counter = 0
    for data, label in train_loader:
        label_last = label[0, -1].long()
        data_last = data[0, -1].long()
        show(data_last.reshape(4, 4).numpy(), label_last)

        if label[0, -2] == -1:
            break_counter += 1
            if break_counter > 1:
                break