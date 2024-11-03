import random
import numpy as np


class Game2048:
    def __init__(self, board_size=4, fill_percent=0.5, seed=None):
        self.board_size = board_size
        self.board = None
        self.init_fill_percent = fill_percent
        self.board = np.zeros((board_size, board_size))
        self.score = 0
        self.merged_count = 0
        self.last_op = -1
        self.op_order = ['up', 'right', 'down', 'left']
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.init_board()

    def init_board(self):
        mask = np.random.randint(1, 100, (self.board_size, self.board_size))
        mask = mask < (self.init_fill_percent * 100)
        self.board = np.ones((self.board_size, self.board_size)) * mask

    def step(self, direction):
        if direction < 0 or direction > 3:
            print("Error: Invalid Direction!")
            return False

        self.last_op = direction
        op = self.op_order[direction]
        first_zero_index = np.ones(self.board_size) * -1
        if op == "up":
            mask = self.board[0] == 0
            first_zero_index = first_zero_index + mask
            self.move_up(first_zero_index)
        elif op == "down":
            mask = self.board[self.board_size - 1] == 0
            first_zero_index = first_zero_index + mask * self.board_size
            self.move_down(first_zero_index)
        elif op == "left":
            mask = self.board[:, 0] == 0
            first_zero_index = first_zero_index + mask
            self.move_left(first_zero_index)
        elif op == "right":
            mask = self.board[:, self.board_size - 1] == 0
            first_zero_index = first_zero_index + mask * self.board_size
            self.move_right(first_zero_index)
        else:
            print("Error: Wrong direction input !")

        self.add_new_entry()
        return self.check_alive()

    def move_up(self, first_zero_index):
        for row in range(1, self.board_size):
            for col in range(0, self.board_size):
                val = self.board[row][col]
                if val == 0:
                    if first_zero_index[col] == -1:
                        first_zero_index[col] = row
                else:
                    pre_val = self.board[row - 1][col]
                    if pre_val == 0:
                        first_zero_row = int(first_zero_index[col])
                        if first_zero_row > 0 and self.board[first_zero_row - 1][col] == val:
                            self.board[first_zero_row - 1][col] += 1
                            self.score += self.board[row][col]
                            self.merged_count += 1
                            self.board[row][col] = 0
                        else:
                            # move up until first zero index
                            self.board[first_zero_row][col] = val
                            self.board[row][col] = 0
                            first_zero_index[col] += 1
                    elif pre_val == val:
                        self.board[row - 1][col] += 1
                        self.score += self.board[row][col]
                        self.merged_count += 1
                        self.board[row][col] = 0
                        first_zero_index[col] = row

    def move_down(self, first_zero_index):
        for row in reversed(range(0, self.board_size - 1)):
            for col in range(0, self.board_size):
                val = self.board[row][col]
                if val == 0:
                    if first_zero_index[col] == -1:
                        first_zero_index[col] = row
                else:
                    pre_val = self.board[row + 1][col]
                    if pre_val == 0:
                        first_zero_row = int(first_zero_index[col])
                        if first_zero_row < self.board_size - 1 and self.board[first_zero_row + 1][col] == val:
                            self.board[first_zero_row + 1][col] += 1
                            self.score += self.board[row][col]
                            self.merged_count += 1
                            self.board[row][col] = 0
                        else:
                            # move down until first zero index
                            self.board[first_zero_row][col] = val
                            self.board[row][col] = 0
                            first_zero_index[col] -= 1
                    elif pre_val == val:
                        self.board[row + 1][col] += 1
                        self.score += self.board[row][col]
                        self.merged_count += 1
                        self.board[row][col] = 0
                        first_zero_index[col] = row

    def move_left(self, first_zero_index):
        for col in range(1, self.board_size):
            for row in range(0, self.board_size):
                val = self.board[row][col]
                if val == 0:
                    if first_zero_index[row] == -1:
                        first_zero_index[row] = col
                else:
                    pre_val = self.board[row][col - 1]
                    if pre_val == 0:
                        first_zero_col = int(first_zero_index[row])
                        if first_zero_col > 0 and self.board[row][first_zero_col - 1] == val:
                            self.board[row][first_zero_col - 1] += 1
                            self.score += self.board[row][col]
                            self.merged_count += 1
                            self.board[row][col] = 0
                        else:
                            # move left until first zero index
                            self.board[row][first_zero_col] = val
                            self.board[row][col] = 0
                            first_zero_index[row] += 1
                    elif pre_val == val:
                        self.board[row][col - 1] += 1
                        self.score += self.board[row][col]
                        self.merged_count += 1
                        self.board[row][col] = 0
                        first_zero_index[row] = col

    def move_right(self, first_zero_index):
        for col in reversed(range(0, self.board_size - 1)):
            for row in range(0, self.board_size):
                val = self.board[row][col]
                if val == 0:
                    if first_zero_index[row] == -1:
                        first_zero_index[row] = col
                else:
                    pre_val = self.board[row][col + 1]
                    if pre_val == 0:
                        first_zero_col = int(first_zero_index[row])
                        if first_zero_col < self.board_size - 1 and self.board[row][first_zero_col + 1] == val:
                            self.board[row][first_zero_col + 1] += 1
                            self.score += self.board[row][col]
                            self.merged_count += 1
                            self.board[row][col] = 0
                        else:
                            # move left until first zero index
                            self.board[row][first_zero_col] = val
                            self.board[row][col] = 0
                            first_zero_index[row] -= 1
                    elif pre_val == val:
                        self.board[row][col + 1] += 1
                        self.score += self.board[row][col]
                        self.merged_count += 1
                        self.board[row][col] = 0
                        first_zero_index[row] = col

    def add_new_entry(self):
        if (self.board != 0).all():
            return False
        filled_percent = (self.board != 0).sum() / (self.board_size ** 2)
        base_num_chance = 0.9
        one_more_chance = 0.3 * (1 - filled_percent)
        while True:
            row = random.randint(0, self.board_size - 1)
            col = random.randint(0, self.board_size - 1)
            if self.board[row][col] == 0:
                if random.random() < base_num_chance:
                    self.board[row][col] = 1
                else:
                    self.board[row][col] = 2

                if random.random() < one_more_chance and (self.board == 0).any():
                    one_more_chance /= 2
                else:
                    return True

    def check_alive(self):
        if (self.board == 0).any():
            return True

        for i in range(self.board_size):
            for j in range(self.board_size):
                val = self.board[i][j]
                if i - 1 >= 0 and val == self.board[i - 1][j]:
                    return True
                if i + 1 < self.board_size and val == self.board[i + 1][j]:
                    return True
                if j - 1 >= 0 and val == self.board[i][j - 1]:
                    return True
                if j + 1 < self.board_size and val == self.board[i][j + 1]:
                    return True
        return False

    def show(self):
        max_num_str = str(np.max(self.board))
        indent_len = len(max_num_str)
        out_str = ""
        out_str += ("\n" + "-" * (self.board_size * indent_len + 2) + "\n")
        for i in range(self.board_size):
            out_str += "|"
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    out_str += (" " * indent_len)
                else:
                    val = str(int(self.board[i][j]))
                    out_str += (val + " " * (indent_len - len(val)))
            out_str += "|\n"
        out_str += ("-" * (self.board_size * indent_len + 2))
        print(out_str)
        print(f"Score: {self.score}, last operation: {self.op_order[self.last_op]}")

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.init_board()
        self.score = 0
        self.last_op = -1
        self.merged_count = 0

    def get_current_state(self):
        return self.board.flatten()
