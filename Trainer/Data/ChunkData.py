import pandas as pd
import numpy as np


def contains_large_numbers(row):
    """Check if the row contains any large numbers (10 or 11)."""
    return any(x in [10, 11] for x in row)


def all_small_numbers(row):
    """Check if all numbers in the row are less than or equal to 2."""
    return all(x <= 2 for x in row)


def process_csv_in_chunks(input_file, output_dir, chunk_size=10000, episodes_per_chunk=1000):
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)  # Read file in chunks
    episodes = []
    current_episode = []
    chunk_counter = 1
    episode_counter = 0

    for chunk in chunk_iter:
        chunk = chunk.apply(pd.to_numeric, downcast='integer')  # Convert all values to int8

        for i in range(len(chunk)):
            current_episode.append(chunk.iloc[i].tolist())  # Add the row to the current episode

            # Check if there's a next row in the chunk or not
            if i < len(chunk) - 1:
                current_row = chunk.iloc[i].tolist()
                next_row = chunk.iloc[i + 1].tolist()

                # Condition: Current row contains large numbers (10, 11) and the next row only has small numbers (≤ 2)
                if contains_large_numbers(current_row) and all_small_numbers(next_row):
                    current_episode.append(['END_OF_EPISODE'])  # Add a marker to indicate the end of an episode
                    episodes.append(current_episode)  # End of the current episode
                    current_episode = []
                    episode_counter += 1

                    # Save episodes every 1000 episodes to avoid memory overload
                    if episode_counter == episodes_per_chunk:
                        output_file = f'{output_dir}/episodes_chunk_{chunk_counter}.csv'

                        # Ensure that numeric data is saved as int8, and the marker is saved correctly
                        chunk_data = [np.array(episode, dtype=object) for episode in episodes]
                        chunk_df = pd.DataFrame([row for episode in chunk_data for row in episode])
                        chunk_df.to_csv(output_file, index=False)
                        print(f'Saved: {output_file}')

                        # Reset for the next chunk
                        episodes = []
                        chunk_counter += 1
                        episode_counter = 0

        # Edge case The last episode in the chunk that hasn't been added yet
        if current_episode:
            current_episode.append(['END_OF_EPISODE'])  # Mark end of the episode
            episodes.append(current_episode)
            current_episode = []

    if episodes:
        output_file = f'{output_dir}/episodes_chunk_{chunk_counter}.csv'

        # Ensure that numeric data is saved as int8, and the marker is saved correctly
        chunk_data = [np.array(episode, dtype=object) for episode in episodes]
        chunk_df = pd.DataFrame([row for episode in chunk_data for row in episode])
        chunk_df.to_csv(output_file, index=False)
        print(f'Saved: {output_file}')

if __name__ == '__main__':
    process_csv_in_chunks(input_file='original_data/HugeDatasetReach4096.csv', output_dir='original_data')