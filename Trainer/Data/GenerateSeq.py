import os

import numpy as np
import pandas as pd
import torch


def csv_to_pt(input_folder, output_folder):
    """
    Convert CSV files in the format episodes_chunk_{i}.csv to chunk_{i}.pt files.

    Each CSV should have:
    - First 16 columns as input data.
    - 17th column as labels.

    Args:
        input_folder (str): Path to the folder containing CSV files.
        output_folder (str): Path to the folder where .pt files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.startswith("episodes_chunk_") and file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)

            # Extract the chunk number
            chunk_number = file_name.split("_")[-1].replace(".csv", "")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Identify "END_OF_EPISODE" rows
            episodes = []
            current_episode_inputs = []
            current_episode_labels = []

            for _, row in df.iterrows():
                if row.isnull().any() or "END_OF_EPISODE" in row.values:
                    # Save the current episode if it has data
                    if current_episode_inputs:
                        episodes.append(
                            (torch.from_numpy(np.array(current_episode_inputs, dtype=np.float32)),
                             torch.from_numpy(np.array(current_episode_labels, dtype=np.float32)).unsqueeze(1))
                        )
                        current_episode_inputs = []
                        current_episode_labels = []
                else:
                    # Add data to the current episode
                    current_episode_inputs.append(row.iloc[:16].values)
                    current_episode_labels.append(row.iloc[16])

            # Ensure the last episode is added if the file does not end with "END_OF_EPISODE"
            if current_episode_inputs:
                episodes.append(
                    (torch.from_numpy(np.array(current_episode_inputs, dtype=np.float32)),
                     torch.from_numpy(np.array(current_episode_labels, dtype=np.float32)).unsqueeze(1))
                )

            # Separate inputs and labels into lists of tensors
            inputs = [episode[0] for episode in episodes]
            labels = [episode[1] for episode in episodes]

            # Save tensors as a .pt file
            output_file_name = f"chunk_{chunk_number}.pt"
            output_file_path = os.path.join(output_folder, output_file_name)
            torch.save({"inputs": inputs, "labels": labels}, output_file_path)

            print(f"Converted {file_name} to {output_file_path}")


if __name__ == "__main__":
    input_folder = "./original_data"  # Replace with the actual folder path
    output_folder = "./processed_data"  # Replace with the desired output folder path
    csv_to_pt(input_folder, output_folder)
