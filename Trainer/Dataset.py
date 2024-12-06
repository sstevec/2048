import random

import torch
from torch.utils.data import Dataset
import os


class SequenceDataset(Dataset):
    def __init__(self, directory, num_chunks, chunk_prefix="chunk_", extension=".pt"):
        self.data = []
        self.labels = []

        # Load specified number of chunks
        for i in range(1, num_chunks + 1):
            file_path = os.path.join(directory, f"{chunk_prefix}{i}{extension}")
            if os.path.exists(file_path):
                chunk_data = torch.load(file_path)
                self.data += chunk_data['inputs']  # Append sequences
                self.labels += chunk_data['labels']  # Append labels
            else:
                print(f"Warning: {file_path} does not exist and will be skipped.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sequence and its corresponding label
        return self.data[idx], self.labels[idx]

    def random_sample(self, k, max_ep_len):
        max_ep_len = max(max_ep_len, 50)

        # Shuffle the list to ensure random selection
        total_length = 0
        selected_data = []
        selected_labels = []
        selected_indices = set()

        while total_length <= k:
            # Randomly pick an index from the list that hasn't been selected yet
            index = random.randint(0, len(self.data) - 1)
            selected_ep = self.data[index]
            if len(selected_ep) < 1000 or len(selected_ep) < max_ep_len:
                # it is a very small expert episode, likely incorrect
                continue

            if index not in selected_indices:
                _data = self.data[index][:max_ep_len]
                _label = self.labels[index][:max_ep_len]

                total_length += len(_data)

                selected_data.append(_data)
                selected_labels.append(_label)
                selected_indices.add(index)

        tensor_data = torch.cat(selected_data, dim=0)[:k]
        tensor_labels = torch.cat(selected_labels, dim=0).to(dtype=torch.long)[:k]

        return tensor_data, tensor_labels


if __name__ == '__main__':
    dataset = SequenceDataset(directory="./data/processed_data", num_chunks=2, chunk_prefix="chunk_", extension=".pt")
    data, label = dataset.random_sample(2000)
    print(len(data), len(label))
