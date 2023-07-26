import os
import torch
import numpy as np
import random
import warnings


def set_device():
    """
    Returns the device to be used (GPU or CPU)

    Returns:
        device (str): The device to be used ('cuda' if GPU is available, 'cpu' otherwise)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    return device


def set_reproducibility(AE_Params):
    """
    Sets the seed for reproducibility of results.

    Args:
        AE_Params (class): Contains the hyperparameters of the autoencoder
    """

    seed = AE_Params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_path(AE_Params):
    """
    Creates the directory path to store the network results.

    Args:
        AE_Params (class): Contains the hyperparameters of the autoencoder
    """

    path = AE_Params.net_dir
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path, exist_ok=False)

