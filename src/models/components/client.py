from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from typing import Dict


class Client:
    """
    Client class for SFL.
    """
    def __init__(self, 
                 id: int, 
                 model: LightningModule,
                 trainer: pl.Trainer,
                 train_data: DataLoader,
                 val_data: DataLoader) -> None:
        self.id = id
        self.model = model
        self.trainer = trainer
        self.train_data = train_data
        self.val_data = val_data


    def update_model(self, updates: Dict[str, Parameter]) -> None:
        """
        Updates the client model with the aggregated weight updates.
        """
        for gradient_name in updates.keys():
            for name, param in self.model.named_parameters():
                if name == gradient_name:
                    param.data = updates[gradient_name]