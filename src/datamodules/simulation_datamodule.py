from functools import partial
from typing import Any, Callable, Dict, Literal, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate

from .components.collate_fn import transform_collate
from .components.datasets.simulation_dataset import SimulationDataset


class SimulationDatamodule(LightningDataModule):
    """LightningDataModule for on-the-fly simulation dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: IterableDataset | SimulationDataset = SimulationDataset(),
        batch_size: int = 32,
        pin_memory: bool = False,
        num_workers: int = 0,
        transforms: Optional[Callable] = None,
        max_train_batches: int | float = 1000,
        max_val_batches: int | float = 100,
        max_test_batches: int | float = 100,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # we need to ensure that training, validation and testing will be
        # done on a maximum number of batches.
        # We need this limitation, since the IterableDataset does not have
        # the notion of the epoch, since it generates data according to the
        # simulation indefinetly. these parameters will then need to be passed
        # around to the trainer.
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.max_test_batches = max_test_batches
        # NOTE: lines below do DOES NOT WORK unless self.trainer is initialized!
        # i see three solution:
        # 1. keep state of how many batches have been generated in
        # self.dataset. this can become cumbersome
        # 2. set max_train/val/test_batches in configs/trainer/default
        # self.trainer.limit_train_batches = max_train_batches
        # self.trainer.limit_val_batches = max_val_batches
        # self.trainer.limit_test_batches = max_test_batches

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed.
        Do not use it to assign state (self.x = y).

        We do not need this function for now.
        """
        pass

    def setup(self, stage: Optional[str] = None, **kwargs) -> None:
        """Prepare simulators. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like instantiating the simulators twice!

        TODO: Allow for different simulators for each of the splits!
        TODO: Write method to gather parameters for each of the splits!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset
            self.data_val = self.dataset
            self.data_test = self.dataset

    def get_dataloader(self, stage: Literal["train", "val", "test"] = "train"):
        assert stage in ["train", "val", "test"]
        dataset = getattr(self, f"data_{stage}")
        # if the dataset does not apply transforms
        # but these are provided in the constructor
        # we apply them after constructing the batch
        if self.transforms and (not hasattr(self.dataset, "transforms")):
            collate_fn = partial(transform_collate, transform=self.transforms)
        else:
            collate_fn = default_collate

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,  # no multithreaded dataloading for iterable style dataset
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        """Returns :attr:`torch.utils.data.DataLoader` used for training."""
        return self.get_dataloader(stage="train")

    def val_dataloader(self):
        """Returns :attr:`torch.utils.data.DataLoader` used for validation."""
        return self.get_dataloader(stage="val")

    def test_dataloader(self):
        """Returns :attr:`torch.utils.data.DataLoader` used for testing."""
        return self.get_dataloader(stage="test")

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
