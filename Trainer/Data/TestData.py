import pandas as pd
import numpy as np

def process_row(row):
    # Ensure the row has exactly 17 elements
    if len(row) != 17:
        raise ValueError("The row must contain exactly 17 elements.")

    # Convert the first 16 elements into a 4x4 NumPy array
    data_4x4 = np.array(row[:16], dtype=np.int8).reshape(4, 4)

    # Extract the last element as the label
    label = row[16]

    return data_4x4, label

# Load the chunked CSV file
chunk_data = pd.read_csv('original_data/episodes_chunk_1.csv')

# Initialize a list to store episodes
loaded_episodes = []
current_episode = []

# Iterate through the rows of the loaded data
for _, row in chunk_data.iterrows():
    if row.iloc[0] == 'END_OF_EPISODE':
        # If we hit the marker, save the current episode and start a new one
        loaded_episodes.append(current_episode)
        current_episode = []
        # to make things faster, we only load the first episode
        break
    else:
        # Otherwise, add the row to the current episode
        current_episode.append(row.tolist())

# Add the last episode if any
if current_episode:
    loaded_episodes.append(current_episode)

# Now, `loaded_episodes` contains all episodes from the CSV
for row in loaded_episodes[0]:
    data_4x4, label = process_row(row)
    print(data_4x4, label)