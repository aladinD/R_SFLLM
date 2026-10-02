from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List
from abc import ABC, abstractmethod
import os
from torch.nn.utils.rnn import pad_sequence


class NERDataModuleBase(LightningDataModule, ABC):
    """
    Abstract Base Class for NER Data Modules.
    """
    def __init__(self, 
                 ner_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 model_type: str,
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.ner_dataset = ner_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.num_splits = num_splits
        self.truncate = truncate

        # Disable warnings when using AutoTokenizer instead of explicit tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def prepare_data(self) -> None:
        """"
        Loads the specified NER dataset.
        """
        if self.ner_dataset == "conll2012_ontonotesv5":
            self.dataset = load_dataset(self.ner_dataset, "english_v12")
        else:
            self.dataset = load_dataset(self.ner_dataset)


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


class CoNLL2003DataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the CoNLL-2003 dataset. 
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


    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        """
        Aligns the NER labels with the tokenized words. 
        Takes into account that one word can be split into multiple tokens.
        """
        new_labels = []
        current_word = None

        for word_id in word_ids:
            # Token doesn't correspond to a word
            if word_id is None:
                new_labels.append(-100)
            # Start of a new word
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(labels[word_id])
            # Continuation of a word, adjust 'B-' to 'I-' if needed
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        # Define tokenizer        
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        # Handle RoBERTa specific prefix
        if self.model_type.startswith("roberta-"):
            tokenizer.add_prefix_space = True

        # Tokenize the input tokens with additional parameters
        tokenized_inputs = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True, padding=True, return_offsets_mapping=True)

        # Align the NER labels with the tokenized inputs
        new_labels = []
        for i, labels in enumerate(dataset["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        # Convert tokenized outputs to tensors
        input_ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.long)
        labels = pad_sequence([torch.tensor(label_seq, dtype=torch.long) for label_seq in new_labels], padding_value=-100, batch_first=True)

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset
    

class WNUT17DataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the WNUT-17 dataset. 
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val/test split
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


    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
        """
        Aligns the NER labels with the tokenized words. 
        Takes into account that one word can be split into multiple tokens.
        """
        new_labels = []
        current_word = None

        for word_id in word_ids:

            # Token doesn't correspond to a word
            if word_id is None:
                new_labels.append(-100)

            # Start of a new word
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(labels[word_id])

            # Continuation of a word, adjust 'B-' to 'I-' if needed
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        # Define tokenizer        
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        # Handle RoBERTa specific prefix
        if self.model_type.startswith("roberta-"):
            tokenizer.add_prefix_space = True

        # Tokenize the input tokens with additional parameters
        tokenized_inputs = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True, padding=True, return_offsets_mapping=True)

        # Align the NER labels with the tokenized inputs
        new_labels = []
        for i, labels in enumerate(dataset["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        # Hanlde padding for labels separately 
        # The reason for using pad_sequence for the labels is to ensure all label sequences are of the same length 
        # (padded to the length of the longest label sequence in the batch). It's important to specify -100 as the 
        # padding value because, during the loss computation in models like BERT, -100 is typically used to ignore tokens.
        input_ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.long)
        labels = pad_sequence([torch.tensor(label_seq, dtype=torch.long) for label_seq in new_labels], padding_value=-100, batch_first=True)

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset


class OntoNotesDataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the OntoNotes 5.0 dataset.
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val/test split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        print("Before Tokenization:", len(self.train_dataset))
        self.train_dataset = self._tokenize(self.train_dataset)
        print("After Tokenization:", len(self.train_dataset))
        self.val_dataset = self._tokenize(self.val_dataset)


    def _align_labels_with_tokens(self, labels, word_ids):
        """
        Aligns the NER labels with the tokenized words. 
        Takes into account that one word can be split into multiple tokens.
        """
        new_labels = []
        current_word = None

        for idx, word_id in enumerate(word_ids):

            # Token doesn't correspond to a word
            if word_id is None:
                new_labels.append(-100)

            # Start of a new word
            elif word_id != current_word:
                current_word = word_id
                new_labels.append(labels[word_id])

            # Continuation of a word; adjust 'B-' to 'I-' if needed
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset for OntoNotes.
        """
        # Define tokenizer        
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        # Handle RoBERTa specific prefix
        if self.model_type.startswith("roberta-"):
            tokenizer.add_prefix_space = True

        all_sentences = [sentence['words'] for example in dataset for sentence in example['sentences']]
        all_labels = [sentence['named_entities'] for example in dataset for sentence in example['sentences']]
        
        # Tokenize the sentences
        tokenized_inputs = tokenizer(all_sentences, truncation=True, padding=True, is_split_into_words=True, return_offsets_mapping=True)

        # Align the NER labels with the tokenized inputs
        aligned_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            aligned_labels.append(self._align_labels_with_tokens(labels, word_ids))

        # Convert tokenized outputs to tensors
        input_ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.long)
        labels = pad_sequence([torch.tensor(label_seq, dtype=torch.long) for label_seq in aligned_labels], padding_value=-100, batch_first=True)

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset