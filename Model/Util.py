import torch
import math

def precompute_2d_positional_encoding(hidden_dim, row, col):
    """
    Precomputes a fixed 2D positional encoding of shape (hidden_dim, row, col).

    Parameters:
    - hidden_dim: Number of dimensions for the positional encoding (typically matches hidden_dim of the input).
    - row: Number of rows in the spatial dimension.
    - col: Number of columns in the spatial dimension.

    Returns:
    - A tensor of shape (hidden_dim, row, col) containing the fixed positional encoding.
    """
    pe = torch.zeros(hidden_dim, row, col)

    # Create row and column positional encodings
    for i in range(row):
        for j in range(col):
            for k in range(0, hidden_dim, 2):
                div_term = 10000 ** (2 * k / hidden_dim)
                pe[k, i, j] = math.sin(i / div_term)  # Sine for even dimensions
                if k + 1 < hidden_dim:
                    pe[k + 1, i, j] = math.cos(j / div_term)  # Cosine for odd dimensions

    return pe