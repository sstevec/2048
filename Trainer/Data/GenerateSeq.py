import pandas as pd
import numpy as np
import torch


def process_episode(episode_data):
    """
    Process an episode and generate sequences of shape [5, 16] and labels [5].

    Parameters:
    - episode_data: List of lists where each sublist contains 17 elements (first 16 are input, 17th is label)

    Returns:
    - sequences: A list of 5-step sequences (shape [5, 16] for each sequence)
    - labels: A list of corresponding class indices (shape [5]) for each sequence
    """
    sequences = []
    labels = []

    num_steps = len(episode_data)

    for i in range(num_steps):
        # Check if the label is NaN for the current step
        if pd.isna(episode_data[i][16]):
            # Skip the entire episode if there's a NaN label
            print(f"NaN label encountered at step {i}, skipping episode.")
            return sequences, labels  # Skip the rest of the episode and move to the next one

        # Get the sequence of 5 steps, padding with zeros if needed
        sequence = []
        sequence_labels = []

        # Loop to collect the current step and up to 4 previous steps
        for j in range(i - 4, i + 1):
            if j < 0:
                # Padding for steps before the start of the episode
                sequence.append(np.zeros(16, dtype=np.float32))
                sequence_labels.append(-1)  # Use -1 as a padding label
            else:
                # Use the actual data from the episode
                step_data = np.array(episode_data[j][:16], dtype=np.float32)
                step_label = int(episode_data[j][16])  # Keep label as class index

                sequence.append(step_data)
                sequence_labels.append(step_label)

        # Append the sequence and labels (each of shape [5, 16] and [5], respectively)
        sequences.append(np.array(sequence, dtype=np.float32))
        labels.append(np.array(sequence_labels, dtype=np.int64))

    return sequences, labels


def process_csv_file(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file, header=0)

    all_sequences = []
    all_labels = []
    current_episode = []

    for _, row in df.iterrows():
        if row.iloc[0] == 'END_OF_EPISODE':
            # Process the current episode once we reach the boundary marker
            if len(current_episode) >= 1:  # Only process if there are enough steps for sequences
                sequences, labels = process_episode(current_episode)
                all_sequences.extend(sequences)
                all_labels.extend(labels)
            current_episode = []  # Reset for the next episode
        else:
            # Convert the row to a list and store it as part of the current episode
            current_episode.append(row.tolist())

    # Edge case: Process the last episode if it wasn't followed by a boundary marker
    if len(current_episode) >= 1:
        sequences, labels = process_episode(current_episode)
        all_sequences.extend(sequences)
        all_labels.extend(labels)

    return all_sequences, all_labels


def save_to_pt_file(sequences, labels, output_file):
    data = {
        'sequences': torch.tensor(np.array(sequences), dtype=torch.int8),
        'labels': torch.tensor(np.array(labels), dtype=torch.int8)
    }
    torch.save(data, output_file)
    print(f"Saved processed data to {output_file}")


for i in range(22, 27):
    sequences, labels = process_csv_file("episodes_chunk_" + str(i) + ".csv")
    save_to_pt_file(sequences, labels, f"chunk_{i}.pt")