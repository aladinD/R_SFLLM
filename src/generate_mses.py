import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import random
from typing import Optional
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from models.components.wireless_module import WirelessModule
from omegaconf import DictConfig
import numpy as np

# Seeding
seed = 42
pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

import pickle

def save_to_file_pkl(data, filename):
    """Saves the data to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_file_pkl(filename):
    """Loads data from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_to_file_np(data, filename):
    """Saves the data to a numpy file."""
    np.save(filename, data)

def load_from_file_np(filename):
    """Loads data from a numpy file."""
    return np.load(filename, allow_pickle=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Instantiate wireless module
    has_wireless = cfg.get("wireless", False)
    wireless: Optional[WirelessModule] = hydra.utils.instantiate(cfg.wireless) if has_wireless else None

    # Number of iterations (num)
    num_rounds = 10
    num_epochs = 10
    num_batches = 2100
    # num_batches = 1000

    n = num_rounds * num_epochs * num_batches

    # Generate MSEs
    all_mses = []
    for _ in range(n):
        print(f"{_+1} / {n}")
        mses = wireless()
        all_mses.append(mses)

    # Save all_mses to a file
    # save_to_file_pkl(all_mses, "mnli_mse_w_protection.pkl")
    save_to_file_pkl(all_mses, "barrage_w_protection_long.pkl")

    # Convert list to numpy array and save
    all_mses_array = np.array(all_mses)
    # save_to_file_np(all_mses_array, "mnli_mse_w_protection.npy")
    save_to_file_np(all_mses_array, "barrage_w_protection_long.npy")

if __name__ == "__main__":
    main()

# COMMAND : 
# python src/generate_mses.py -m experiment='FILE'
