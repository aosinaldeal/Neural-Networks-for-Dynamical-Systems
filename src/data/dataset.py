import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def numpy_to_tensor(time: np.ndarray, position: np.ndarray):

    time_tensor = torch.tensor(
        time,
        dtype=torch.float32
    ).reshape(-1,1)

    position_tensor = torch.tensor(
        position,
        dtype=torch.float32
    )

    if position_tensor.ndim == 1:
        position_tensor = position_tensor.reshape(-1,1)

    return time_tensor, position_tensor


def split_dataset(
    time_tensor,
    position_tensor,
    train_ratio=0.8,
    method="random"
):
    """
    Split tensors into training and testing sets.

    Parameters
    ----------
    method : str
        "random"     -> random split (interpolation)
        "sequential" -> first part train, last part test (extrapolation)
    """

    split_index = int(len(time_tensor) * train_ratio)

    if method == "sequential":

        train_time = time_tensor[:split_index]
        train_position = position_tensor[:split_index]

        test_time = time_tensor[split_index:]
        test_position = position_tensor[split_index:]

    elif method == "random":

        indices = torch.randperm(len(time_tensor))

        train_idx = indices[:split_index]
        test_idx = indices[split_index:]

        train_time = time_tensor[train_idx]
        train_position = position_tensor[train_idx]

        test_time = time_tensor[test_idx]
        test_position = position_tensor[test_idx]

    else:
        raise ValueError(
            "method must be 'random' or 'sequential'"
        )

    return (
        train_time,
        train_position,
        test_time,
        test_position,
    )


def create_dataloader(time_tensor, position_tensor, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader.
    """

    dataset = TensorDataset(time_tensor, position_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader