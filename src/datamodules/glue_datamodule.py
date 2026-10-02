from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List
from abc import ABC, abstractmethod
import os


class GLUEDataModuleBase(LightningDataModule, ABC):
    """
    Abstract Base Class for GLUE Data Modules.
    """
    def __init__(self, 
                 glue_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 model_type: str,
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.glue_dataset = glue_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.num_splits = num_splits
        self.truncate = truncate

        # Disable warnings when using AutoTokenizer instead of explicit tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def prepare_data(self) -> None:
        """"
        Loads the specified GLUE dataset.
        """
        self.dataset = load_dataset('glue', self.glue_dataset)


    @abstractmethod
    def setup(self, stage: str = None) -> None:
        """
        Abstract method for preprocessing the loaded dataset.
        """
        pass


    def train_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the training dataset.
        """
        if self.num_splits is not None:
            splits = self._split_data(self.train_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the validation dataset.
        """
        if self.num_splits is not None:
            splits = self._split_data(self.val_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    @abstractmethod
    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Abstract method for tokenizing the dataset.
        """
        pass


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
    

class SST2DataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the SST2 dataset. 
    """
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


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)  # Add to Config! 
        encodings = tokenizer(dataset["sentence"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    
        return dataset 
    

class MRPCDataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the MRPC dataset.
    """
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


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    
        return dataset 
    


class QNLIDataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the QNLI dataset. 
    """
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


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(text=dataset["question"], 
                              text_pair=dataset["sentence"], 
                              truncation=True, 
                              padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    
        return dataset


class MNLIDataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the MNLI dataset.
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation_matched"]
        
        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["premise"], dataset["hypothesis"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        
        return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    

class COLADataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the COLA dataset. 
    """
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

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["sentence"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset
    

class RTEDataModule(GLUEDataModuleBase):
    """
    LightningDataModule Class for the RTE dataset. 
    """
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

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset