from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import os


class QADataModuleBase(LightningDataModule, ABC):
    """
    Abstract Base Class for QA Data Modules.
    """
    def __init__(self, 
                 qa_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 model_type: str,
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.qa_dataset = qa_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.num_splits = num_splits
        self.truncate = truncate

        # Disable warnings when using AutoTokenizer instead of explicit tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def prepare_data(self) -> None:
        """"
        Loads the specified QA dataset.
        """
        self.dataset = load_dataset(self.qa_dataset)


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
            tensor_splits = self._split_data(self.train_dataset, self.num_splits)
            context_splits = self._split_data(self.train_contexts, self.num_splits)
            return [DataLoader(list(zip(tensor_split, context_split)), batch_size=self.batch_size) 
                for tensor_split, context_split in zip(tensor_splits, context_splits)]
        
            # splits = self._split_data(self.train_dataset, self.num_splits)
            # return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(list(zip(self.train_dataset, self.train_contexts)), 
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the validation dataset.
        """
        if self.num_splits is not None:
            tensor_splits = self._split_data(self.val_dataset, self.num_splits)
            context_splits = self._split_data(self.val_contexts, self.num_splits)
            return [DataLoader(list(zip(tensor_split, context_split)), batch_size=self.batch_size) 
                    for tensor_split, context_split in zip(tensor_splits, context_splits)]
        else:
            return DataLoader(list(zip(self.val_dataset, self.val_contexts)), 
                            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

        # if self.num_splits is not None:
        #     splits = self._split_data(self.val_dataset, self.num_splits)
        #     return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        # else:
        #     return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


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


class SQUADDataModule(QADataModuleBase):
    """
    LightningDataModule Class for the SQUAD dataset. 
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Stage stuff
        self.current_stage = stage

        # Train/val split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        # self.train_dataset = self._tokenize(self.train_dataset)
        # self.val_dataset = self._tokenize(self.val_dataset)
        self.train_dataset, self.train_contexts = self._tokenize(self.train_dataset)
        self.val_dataset, self.val_contexts = self._tokenize(self.val_dataset)


    # def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
    #     """
    #     Tokenizes the dataset.
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_type)

    #     # Tokenize questions and contexts
    #     questions = [entry['question'] for entry in dataset]
    #     contexts = [entry['context'] for entry in dataset]
    #     encodings = tokenizer(questions, contexts, truncation=True, padding=True, return_offsets_mapping=True, return_tensors="pt")
        
    #     # Convert answer start and end positions to token positions
    #     start_positions = []
    #     end_positions = []
    #     for i, (answer_start_char, answer_end_char) in enumerate(zip([ans['answer_start'][0] for ans in dataset['answers']], [ans['answer_start'][0] + len(ans['text'][0]) for ans in dataset['answers']])):
    #         # Get the corresponding token position for the char position
    #         start_token = len([offset for offset in encodings.offset_mapping[i] if offset[0] <= answer_start_char])
    #         end_token = len([offset for offset in encodings.offset_mapping[i] if offset[1] <= answer_end_char]) - 1
    #         start_positions.append(start_token)
    #         end_positions.append(end_token)

    #     # Convert to tensors
    #     input_ids = encodings["input_ids"].clone().detach()
    #     attention_mask = encodings["attention_mask"].clone().detach()
    #     start_positions = torch.tensor(start_positions, dtype=torch.long)
    #     end_positions = torch.tensor(end_positions, dtype=torch.long)

    #     dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, start_positions, end_positions)

    #     return dataset, contexts



    # DEFINITELY WORKS BUT SLOW AF
    # def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
    #     """
    #     Tokenizes the dataset.
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        
    #     def process_sample(tokenizer, sample: Dict) -> Dict:
    #         question = sample['question']
    #         context = sample['context']
    #         answer_start_char = sample['answers']['answer_start'][0]
    #         answer_end_char = answer_start_char + len(sample['answers']['text'][0])

    #         encoding = tokenizer(question, context, truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True)
    #         offsets = encoding['offset_mapping']

    #         start_token = None
    #         end_token = None

    #         for idx, (start, end) in enumerate(offsets):
    #             if start <= answer_start_char <= end:
    #                 start_token = idx
    #             if start <= answer_end_char <= end:
    #                 end_token = idx

    #             if start_token is not None and end_token is not None:
    #                 break

    #         if start_token is None or end_token is None:
    #             start_token, end_token = 0, 0

    #         encoding['start_positions'] = start_token
    #         encoding['end_positions'] = end_token

    #         return encoding

    #     def tokenize_and_encode(dataset: List[Dict], tokenizer):
    #         return [process_sample(tokenizer, sample) for sample in dataset]

    #     encoded_dataset = tokenize_and_encode(dataset, tokenizer)

    #     input_ids = torch.tensor([item['input_ids'] for item in encoded_dataset])
    #     attention_mask = torch.tensor([item['attention_mask'] for item in encoded_dataset])
    #     start_positions = torch.tensor([item['start_positions'] for item in encoded_dataset])
    #     end_positions = torch.tensor([item['end_positions'] for item in encoded_dataset])

    #     tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, start_positions, end_positions)
        
    #     contexts = [sample['context'] for sample in dataset]

    #     return tensor_dataset, contexts


        




    # def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
    #     """
    #     Tokenizes the dataset.
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_type)

    #     # Tokenize questions and contexts
    #     questions = [entry['question'] for entry in dataset]
    #     contexts = [entry['context'] for entry in dataset]
    #     encodings = tokenizer(questions, contexts, max_length=384, truncation="only_second", padding="max_length", return_offsets_mapping=True, return_tensors="pt")

    #     # Convert answer start and end positions to token positions
    #     start_positions = []
    #     end_positions = []
    #     for i, (answer_start_char, answer_end_char) in enumerate(zip([ans['answer_start'][0] for ans in dataset['answers']], [ans['answer_start'][0] + len(ans['text'][0]) for ans in dataset['answers']])):
           
    #         # Get the corresponding token position for the char position
    #         sequence_ids = [(i, e) for i, e in enumerate(encodings['token_type_ids'][i]) if e == 1]
            
    #         # Find the start and end of the context
    #         context_start = sequence_ids[0][0]
    #         context_end = sequence_ids[-1][0]

    #         # Print Offset Mappings
    #         for j in range(context_start, context_end+1):
    #             print(j, encodings.offset_mapping[i][j])
            
    #         # If the answer is not fully inside the context, label it (0, 0)
    #         if encodings.offset_mapping[i][context_start][0] > answer_end_char or encodings.offset_mapping[i][context_end][1] < answer_start_char:
    #             start_positions.append(0)
    #             end_positions.append(0)
    #         else:
    #             # Otherwise it's the start and end token positions
    #             idx = context_start
    #             while idx <= context_end and encodings.offset_mapping[i][idx][0] <= answer_start_char:
    #                 # Print Conversion Steps
    #                 idx += 1
    #             start_positions.append(idx - 1)

    #             idx = context_end
    #             while idx >= context_start and encodings.offset_mapping[i][idx][1] >= answer_end_char:
    #                 idx -= 1
    #             end_positions.append(idx + 1)

    #     # Convert to tensors        
    #     input_ids = encodings['input_ids'].clone().detach()
    #     attention_mask = encodings['attention_mask'].clone().detach()
    #     start_positions = torch.tensor(start_positions, dtype=torch.long)
    #     end_positions = torch.tensor(end_positions, dtype=torch.long)

    #     dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, start_positions, end_positions)

    #     return dataset, contexts



    def _tokenize(self, dataset) -> Tuple[torch.utils.data.TensorDataset, List[str]]:
        """
        Tokenizes the SQuAD dataset.

        Args:
        - dataset: The SQuAD dataset.

        Returns:
        - A tuple containing the TensorDataset and the list of contexts.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        input_ids, attention_masks, token_type_ids, start_positions, end_positions = [], [], [], [], []
        contexts = []

        # Iterate over the data to convert it into the format BERT requires
        for example in dataset:
            context = example['context']
            question = example['question']
            answer = example['answers']['text'][0]
            answer_start = example['answers']['answer_start'][0]
            answer_end = answer_start + len(answer)

            # Tokenize context and question
            inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt', return_offsets_mapping=True, truncation=True, max_length=512)
            input_id = inputs['input_ids'].squeeze()
            attention_mask = inputs['attention_mask'].squeeze()
            token_type_id = inputs['token_type_ids'].squeeze()
            offsets = inputs['offset_mapping'].squeeze()

            max_length = 512  # or whatever maximum length you've determined
            padding_length = max_length - len(input_id)
            input_id = torch.cat([input_id, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            token_type_id = torch.cat([token_type_id, torch.zeros(padding_length, dtype=torch.long)])

            # Find the tokenized start and end positions of the answer
            answer_start_token = 0
            while answer_start > offsets[answer_start_token][1]:
                answer_start_token += 1

            answer_end_token = answer_start_token
            while answer_end > offsets[answer_end_token][1]:
                answer_end_token += 1

            # Append the tokenized data
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            start_positions.append(answer_start_token)
            end_positions.append(answer_end_token)
            contexts.append(context)

        # Convert lists to tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        start_positions = torch.tensor(start_positions)
        end_positions = torch.tensor(end_positions)

        tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, start_positions, end_positions)
        return tensor_dataset, contexts