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
                self.data.append(chunk_data['sequences'])  # Append sequences
                self.labels.append(chunk_data['labels'])  # Append labels
            else:
                print(f"Warning: {file_path} does not exist and will be skipped.")

        # Concatenate all chunks into a single tensor
        self.data = torch.cat(self.data, dim=0) if self.data else torch.tensor([])
        self.labels = torch.cat(self.labels, dim=0) if self.labels else torch.tensor([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sequence and its corresponding label
        return self.data[idx], self.labels[idx]
