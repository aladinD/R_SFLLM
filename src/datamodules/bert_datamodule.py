from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from typing import List


class BertDataModule(LightningDataModule):
    """
    LightningDataModule Class for BERT models. Loads and preprocesses a specified GLUE dataset.
    """
    def __init__(self, 
                 glue_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 model_type: str = "bert-base-uncased",
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.glue_dataset = glue_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.num_splits = num_splits
        self.truncate = truncate


    def prepare_data(self) -> None:
        """"
        Loads the specified GLUE dataset.
        """
        self.dataset = load_dataset('glue', self.glue_dataset)


    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """

        # Train/val split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)

        
    def train_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the training dataset.
        """
        # Splitting
        if self.num_splits is not None:
            splits = self._split_data(self.train_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the validation dataset.
        """
        # Splitting
        if self.num_splits is not None:
            splits = self._split_data(self.val_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = BertTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["sentence"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        return dataset 


    def _truncate(self, dataset) -> DatasetDict:
        """
        Truncates the dataset to the specified length.
        """
        return dataset.select(range(self.truncate))


    def _split_data(self, 
                    dataset: DatasetDict, 
                    num_splits: int) -> List[DatasetDict]:
        """
        Splits the dataset into equal num_splits splits.
        """
        split_size = len(dataset) // num_splits
        remainder = len(dataset) % num_splits

        split_lengths = [split_size + 1 if i < remainder else split_size for i in range(num_splits)]
        splits = torch.utils.data.random_split(dataset, split_lengths)

        return splits