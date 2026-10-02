import math
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.text.squad import SQuAD

from pytorch_lightning import LightningModule
from transformers.modeling_outputs import (BaseModelOutputWithPoolingAndCrossAttentions,
                                           SequenceClassifierOutput, QuestionAnsweringModelOutput)
from transformers.models.bert.modeling_bert import (BertEmbeddings, BertEncoder,
                                                   BertPooler, BertPreTrainedModel, TokenClassifierOutput)
from transformers import get_linear_schedule_with_warmup

import numpy as np



class BERTForSequenceClassificationModule(BertPreTrainedModel, LightningModule):
    """
    LitModule BertForSequenceClassification Model with the option to add noise to
    the word_embeddings.
    """
    def __init__(self, config):
        super().__init__(config)

        # BertForSequenceClassification init
        self.num_labels = config.num_labels
        self.config = config
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # BertModel init
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        add_pooling_layer = True
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Optimizer params
        self.lr_val = None
        self.eps_val = None
        self.warmup = None
        self.scheduler_training_steps = None

        # Classes params
        self.num_classes = None

        # Adversarial training
        self.adversarial_training = False

        # Assign noise per round
        self.skip_noise = False
        self.add_noise: Optional[float] = None
        self.noise_mode = None 

        # Identifies the correct MSE value to be added to the current user
        self.user_id = None

        # Identifies the current batch, epoch and number of batches
        self.batch_index = 0
        self.current_train_epoch = 0
        self.num_batches = 0

        # Identifies the current global round
        self.current_round = 0

        # Initialize weights and apply final processing
        self.post_init()


    def load_mses(self, mse_filepath: str):
        """Loads the MSE file."""
        self.all_mses = np.load(mse_filepath, allow_pickle=True)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:


        ### Bert Model class implementation
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # Conditionally add noise
        if self.adversarial_training:

            if self.training and not self.skip_noise:

                if self.noise_mode == 'per_batch': 
                    
                    index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    current_mse = self.all_mses[index][self.user_id]

                    # DEBUG
                    # print("PER BATCH ADVERSARIAL")
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # print("BATCH INDEX: ", index)
                    # print(f"CURRENT BATCH {self.batch_index} AND CLIENT {self.user_id} with MSE {current_mse}")

                    self.add_noise = current_mse
                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise


                elif self.noise_mode == 'per_round' and self.add_noise is not None:

                    # DEBUG
                    # print("PER ROUND ADVERSARIAL")
                    # print("NOISE : ", self.add_noise)
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    # print("BATCH INDEX: ", index)

                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise

        else:

            if not self.skip_noise:

                if self.noise_mode == 'per_batch': 
                    
                    index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    current_mse = self.all_mses[index][self.user_id]

                    # DEBUG
                    # print("PER BATCH NON ADVERSARIAL")
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # print("BATCH INDEX: ", index)
                    # print(f"CURRENT BATCH {self.batch_index} AND CLIENT {self.user_id} with MSE {current_mse}")

                    self.add_noise = current_mse
                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise


                elif self.noise_mode == 'per_round' and self.add_noise is not None:

                    # DEBUG
                    # print("PER ROUND NON ADVERSARIAL")
                    # print("NOISE : ", self.add_noise)
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    # print("BATCH INDEX: ", index)

                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        ### Post encoder
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def init_metrics(self) -> None:
        """
        Initializes the metrics for the model.
        """
        if self.num_classes == 2:
            self.accuracy = Accuracy(task="binary", num_classes=self.num_classes)
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        

    def training_step(self, batch: Any, batch_idx) -> torch.Tensor:
        # Metrics tracking for noise
        self.batch_index = batch_idx
        self.current_train_epoch = self.trainer.current_epoch
        self.num_batches = len(self.trainer.train_dataloader)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }

        outputs = self.forward(**inputs)
        loss = outputs.loss
        accuracy = self.accuracy(torch.argmax(outputs.logits, dim=1), inputs["labels"])
        
        # self.log_dict({'train_loss': loss, 'train_acc': accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy, on_epoch=True, on_step=False)

        return loss
    

    def validation_step(self, batch: Any, batch_idx) -> torch.Tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        loss = outputs.loss
        accuracy = self.accuracy(torch.argmax(outputs.logits, dim=1), inputs["labels"])

        # self.log_dict({'val_loss': loss, 'val_acc': accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", accuracy, on_epoch=True, on_step=False)

        return loss
    

    def test_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.Tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        loss = outputs.loss
        accuracy = self.accuracy(torch.argmax(outputs.logits, dim=1), inputs["labels"])

        self.log_dict({'test_loss': loss, 'test_acc': accuracy}, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def predict_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        return preds
    

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_val, eps=self.eps_val)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=self.warmup * self.scheduler_training_steps if self.warmup is not None else 1256, 
                                                       num_training_steps=self.scheduler_training_steps) 
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


    def lr_scheduler_step(self, scheduler, optimizer_idx, *args, **kwargs):
        """
        Needs to be overwritten due to non LambdaLR scheduler in configure_optimizers.
        Transformer scheduler get_linear_schedule_with_warmup is not compatible with the PL workflow.
        """
        scheduler.step()



class BERTForTokenClassificationModule(BertPreTrainedModel, LightningModule):
    """
    LitModule BertForTokenClassification Model with the option to add noise to
    the word_embeddings.
    """
    def __init__(self, config):
        super().__init__(config)

        # BertForTokenClassification init
        self.num_labels = config.num_labels
        self.config = config
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # BertModel init
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        add_pooling_layer = True
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Optimizer params
        self.lr_val = None
        self.eps_val = None
        self.warmup = None
        self.scheduler_training_steps = None

        # Classes params
        self.num_classes = None

        # Adversarial training
        self.adversarial_training = False

        # Assign noise per round
        self.skip_noise = False
        self.add_noise: Optional[float] = None
        self.noise_mode = None 

        # Identifies the correct MSE value to be added to the current user
        self.user_id = None

        # Identifies the current batch, epoch and number of batches
        self.batch_index = 0
        self.current_train_epoch = 0
        self.num_batches = 0

        # Identifies the current global round
        self.current_round = 0

        # Initialize weights and apply final processing
        self.post_init()


    def load_mses(self, mse_filepath: str):
        """Loads the MSE file."""
        self.all_mses = np.load(mse_filepath, allow_pickle=True)


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        ### Bert Model class implementation
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # Conditionally add noise
        if self.adversarial_training:

            if self.training and not self.skip_noise:

                if self.noise_mode == 'per_batch': 
                    
                    index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    current_mse = self.all_mses[index][self.user_id]

                    # DEBUG
                    # print("PER BATCH ADVERSARIAL")
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # print("BATCH INDEX: ", index)
                    # print(f"CURRENT BATCH {self.batch_index} AND CLIENT {self.user_id} with MSE {current_mse}")

                    self.add_noise = current_mse
                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise


                elif self.noise_mode == 'per_round' and self.add_noise is not None:

                    # DEBUG
                    # print("PER ROUND ADVERSARIAL")
                    # print("NOISE : ", self.add_noise)
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    # print("BATCH INDEX: ", index)

                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise

        else:

            if not self.skip_noise:

                if self.noise_mode == 'per_batch': 
                    
                    index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    current_mse = self.all_mses[index][self.user_id]

                    # DEBUG
                    # print("PER BATCH NON ADVERSARIAL")
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # print("BATCH INDEX: ", index)
                    # print(f"CURRENT BATCH {self.batch_index} AND CLIENT {self.user_id} with MSE {current_mse}")

                    self.add_noise = current_mse
                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise


                elif self.noise_mode == 'per_round' and self.add_noise is not None:

                    # DEBUG
                    # print("PER ROUND NON ADVERSARIAL")
                    # print("NOISE : ", self.add_noise)
                    # print("BATCH N0: ", self.batch_index)
                    # print("CURRENT EPOCH: ", self.current_train_epoch)
                    # print("CURRENT ROUND: ", self.current_round)
                    # print("NUM BATCHES: ", self.num_batches)
                    # index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
                    # print("BATCH INDEX: ", index)

                    noise = torch.normal(mean=0, std=math.sqrt(self.add_noise), size=embedding_output.shape).to(embedding_output.device)
                    embedding_output += noise

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        ### Post encoder
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def init_metrics(self) -> None:
        """
        Initializes the metrics for the model.
        """
        if self.num_classes == 2:
            self.f1 = F1Score(task="binary", num_classes=self.num_classes)
            self.precision = Precision(task="binary", num_classes=self.num_classes)
            self.recall = Recall(task="binary", num_classes=self.num_classes)
        else:
            # Use macro averaging for WNUT_17 and CONLL2003 datasets due to class imbalance
            self.f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
            self.precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
            self.recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")


    def on_epoch_start(self):
        """
        Called at the start of each epoch to reset metrics and avoid accumulation.
        """
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        

    def training_step(self, batch: Any, batch_idx) -> torch.Tensor:
        # Metrics tracking for noise
        self.batch_index = batch_idx
        self.current_train_epoch = self.trainer.current_epoch
        self.num_batches = len(self.trainer.train_dataloader)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }

        outputs = self.forward(**inputs)
        loss = outputs.loss

        # Metrics
        preds = torch.argmax(outputs.logits, dim=2).view(-1)
        target = inputs["labels"].view(-1)

        # Masking out the PAD token
        mask = inputs["attention_mask"].view(-1).bool()
        final_mask = (mask & (target != -100))
        valid_preds = preds[final_mask]
        valid_target = target[final_mask]

        preds = valid_preds
        target = valid_target

        f1 = self.f1(preds, target)
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)
        
        # Logging
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_f1", f1, on_epoch=True, on_step=False)
        self.log("train_precision", precision, on_epoch=True, on_step=False)
        self.log("train_recall", recall, on_epoch=True, on_step=False)

        return loss
    

    def validation_step(self, batch: Any, batch_idx) -> torch.Tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        loss = outputs.loss

        # Metrics
        preds = torch.argmax(outputs.logits, dim=2).view(-1)
        target = inputs["labels"].view(-1)

        # Masking out the PAD token
        mask = inputs["attention_mask"].view(-1).bool()
        final_mask = (mask & (target != -100))
        valid_preds = preds[final_mask]
        valid_target = target[final_mask]

        preds = valid_preds
        target = valid_target

        f1 = self.f1(preds, target)
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)
        
        # Logging
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_f1", f1, on_epoch=True, on_step=False)
        self.log("val_precision", precision, on_epoch=True, on_step=False)
        self.log("val_recall", recall, on_epoch=True, on_step=False)

        return loss
    

    def test_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.Tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)
        loss = outputs.loss
        
        # Metrics
        preds = torch.argmax(outputs.logits, dim=2).view(-1)
        target = inputs["labels"].view(-1)

        # Masking out the PAD token
        mask = inputs["attention_mask"].view(-1).bool()
        final_mask = (mask & (target != -100))
        valid_preds = preds[final_mask]
        valid_target = target[final_mask]

        preds = valid_preds
        target = valid_target

        f1 = self.f1(preds, target)
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)
        
        # Logging
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_f1", f1, on_epoch=True, on_step=False)
        self.log("test_precision", precision, on_epoch=True, on_step=False)
        self.log("test_recall", recall, on_epoch=True, on_step=False)

        return loss
    

    def predict_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.tensor:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        with torch.no_grad():
            outputs = self.forward(**inputs)

        preds = torch.argmax(outputs.logits, dim=2)

        return preds
    

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_val, eps=self.eps_val)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=self.warmup * self.scheduler_training_steps if self.warmup is not None else 1256, 
                                                       num_training_steps=self.scheduler_training_steps) 
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


    def lr_scheduler_step(self, scheduler, optimizer_idx, *args, **kwargs):
        """
        Needs to be overwritten due to non LambdaLR scheduler in configure_optimizers.
        Transformer scheduler get_linear_schedule_with_warmup is not compatible with the PL workflow.
        """
        scheduler.step()


# class BERTForQuestionAnsweringModule(BertPreTrainedModel, LightningModule):
#     """
#     LitModule BertForQuestionAnswering Model with the option to add noise to
#     the word_embeddings.
#     """
#     def __init__(self, config):
#         super().__init__(config)

#         # BertForQuestionAnswering init
#         self.num_labels = config.num_labels
#         self.config = config
#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

#         # BertModel init
#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         add_pooling_layer = True
#         self.pooler = BertPooler(config) if add_pooling_layer else None

#         # Optimizer params
#         self.lr_val = None
#         self.eps_val = None
#         self.warmup = None
#         self.scheduler_training_steps = None

#         # Classes params
#         self.num_classes = None

#         # Assign noise per round
#         self.add_noise: Optional[float] = None
#         self.noise_mode = None 

#         # Identifies the correct MSE value to be added to the current user
#         self.user_id = None

#         # Identifies the current batch, epoch and number of batches
#         self.batch_index = 0
#         self.current_train_epoch = 0
#         self.num_batches = 0

#         # Identifies the current global round
#         self.current_round = 0

#         # Initialize weights and apply final processing
#         self.post_init()


#     def load_mses(self, mse_filepath: str):
#         """Loads the MSE file."""
#         self.all_mses = np.load(mse_filepath, allow_pickle=True)


#     # def init_mse_logger(self, log_filepath: str):
#     #     """Initializes the MSE logger."""
#     #     self.mse_logger = CSVLogger(save_dir=log_filepath, name="batch_mse")


#     def forward(
#             self,
#             input_ids: Optional[torch.Tensor] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             inputs_embeds: Optional[torch.Tensor] = None,
#             encoder_hidden_states: Optional[torch.Tensor] = None,
#             encoder_attention_mask: Optional[torch.Tensor] = None,
#             past_key_values: Optional[List[torch.FloatTensor]] = None,
#             use_cache: Optional[bool] = None,
#             start_positions: Optional[torch.Tensor] = None,
#             end_positions: Optional[torch.Tensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#         ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:


#         ### Bert Model class implementation
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.config.is_decoder:
#             use_cache = use_cache if use_cache is not None else self.config.use_cache
#         else:
#             use_cache = False

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         batch_size, seq_length = input_shape
#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         # past_key_values_length
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

#         if attention_mask is None:
#             attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

#         if token_type_ids is None:
#             if hasattr(self.embeddings, "token_type_ids"):
#                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         import pdb
#         pdb.set_trace()

#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#             past_key_values_length=past_key_values_length,
#         )

#         # Conditionally add noise
#         if self.training:
#             if self.noise_mode == 'per_batch': 
                
#                 # current_mse = self.all_mses[self.batch_index][self.user_id]
#                 index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
#                 current_mse = self.all_mses[index][self.user_id]

#                 # self.log("batch_mse", current_mse, on_epoch=False, on_step=True, logger=self.mse_logger)

#                 # DEBUG
#                 # print("PER BATCH")
#                 # print("BATCH N0: ", self.batch_index)
#                 # print("CURRENT EPOCH: ", self.current_train_epoch)
#                 # print("CURRENT ROUND: ", self.current_round)
#                 # print("NUM BATCHES: ", self.num_batches)
#                 # print("BATCH INDEX: ", index)
#                 # print(f"CURRENT BATCH {self.batch_index} AND CLIENT {self.user_id} with MSE {current_mse}")

#                 self.add_noise = current_mse
#                 noise = torch.normal(mean=0.0, std=self.add_noise, size=embedding_output.shape, device=self.device)
#                 embedding_output += noise


#             elif self.noise_mode == 'per_round' and self.add_noise is not None:

#                 # DEBUG
#                 # print("PER ROUND")
#                 # print("BATCH N0: ", self.batch_index)
#                 # print("CURRENT EPOCH: ", self.current_train_epoch)
#                 # print("CURRENT ROUND: ", self.current_round)
#                 # print("NUM BATCHES: ", self.num_batches)
#                 # index = self.batch_index + (self.current_train_epoch * self.num_batches) + (self.current_train_epoch * self.current_round * self.num_batches)
#                 # print("BATCH INDEX: ", index)

#                 noise = torch.normal(mean=0.0, std=self.add_noise, size=embedding_output.shape, device=self.device)
#                 embedding_output += noise

#         encoder_outputs = self.encoder(
#             embedding_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

#         outputs = BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             past_key_values=encoder_outputs.past_key_values,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             cross_attentions=encoder_outputs.cross_attentions,
#         )

#         ### Post encoder
#         sequence_output = outputs[0]

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
    

#     def init_metrics(self) -> None:
#         """
#         Initializes the metrics for the model.
#         """
#         self.squad = SQuAD()
        

#     def training_step(self, batch: Any, batch_idx) -> torch.Tensor:
#         # Metrics tracking for noise
#         self.batch_index = batch_idx
#         self.current_train_epoch = self.trainer.current_epoch
#         self.num_batches = len(self.trainer.train_dataloader)

#         tensor_data, contexts = batch
#         input_ids, attention_mask, start_positions, end_positions = tensor_data
#         contexts = list(contexts)

#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "start_positions": start_positions,
#             "end_positions": end_positions
#         }
        
#         outputs = self.forward(**inputs)
#         loss = outputs.loss

#         # Start and end position predictions
#         start_preds = torch.argmax(outputs.start_logits, dim=1)
#         end_preds = torch.argmax(outputs.end_logits, dim=1)

#         # DEBUG
#         print("start_preds: ", start_preds)
#         print("end_preds: ", end_preds)
        
#         # Convert predictions to the required format
#         preds = [{"prediction_text": context[start.item():end.item()+1], 
#                 "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_preds, 
#                                                                             end_preds, 
#                                                                             contexts))]

#         targets = [{"answers": {"answer_start": [start.item()], 
#                         "text": [context[start.item():end.item()+1]]}, 
#             "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_positions, 
#                                                                         end_positions, 
#                                                                         contexts))]

#         # Metrics
#         self.squad(preds, targets)
#         metrics_results = self.squad.compute()
#         f1 = metrics_results["f1"]
#         em = metrics_results["exact_match"]
        
#         # Logging
#         self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=128)
#         self.log("train_f1", f1, on_epoch=True, on_step=False, batch_size=128)
#         self.log("train_em", em, on_epoch=True, on_step=False, batch_size=128)

#         return loss
    

#     def validation_step(self, batch: Any, batch_idx) -> torch.Tensor:
#         tensor_data, contexts = batch
#         input_ids, attention_mask, start_positions, end_positions = tensor_data
#         contexts = list(contexts)

#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "start_positions": start_positions,
#             "end_positions": end_positions
#         }

#         with torch.no_grad():
#             outputs = self.forward(**inputs)
#         loss = outputs.loss

#         # Start and end position predictions
#         start_preds = torch.argmax(outputs.start_logits, dim=1)
#         end_preds = torch.argmax(outputs.end_logits, dim=1)

#         # Convert predictions to the required format
#         preds = [{"prediction_text": context[start.item():end.item()+1], 
#                 "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_preds, 
#                                                                             end_preds, 
#                                                                             contexts))]

#         targets = [{"answers": {"answer_start": [start.item()], 
#                         "text": [context[start.item():end.item()+1]]}, 
#             "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_positions, 
#                                                                         end_positions, 
#                                                                         contexts))]

#         # Metrics
#         self.squad(preds, targets)
#         metrics_results = self.squad.compute()
#         f1 = metrics_results["f1"]
#         em = metrics_results["exact_match"]
        
#         # Logging
#         self.log("val_loss", loss, on_epoch=True, on_step=False, batch_size=128)
#         self.log("val_f1", f1, on_epoch=True, on_step=False, batch_size=128)
#         self.log("val_em", em, on_epoch=True, on_step=False, batch_size=128)

#         return loss
    

#     def test_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.Tensor:
#         tensor_data, contexts = batch
#         input_ids, attention_mask, start_positions, end_positions = tensor_data
#         contexts = list(contexts)

#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "start_positions": start_positions,
#             "end_positions": end_positions
#         }

#         with torch.no_grad():
#             outputs = self.forward(**inputs)
#         loss = outputs.loss
        
#         # Start and end position predictions
#         start_preds = torch.argmax(outputs.start_logits, dim=1)
#         end_preds = torch.argmax(outputs.end_logits, dim=1)

#         # Convert predictions to the required format
#         preds = [{"prediction_text": context[start.item():end.item()+1], 
#                 "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_preds, 
#                                                                             end_preds, 
#                                                                             contexts))]

#         targets = [{"answers": {"answer_start": [start.item()], 
#                         "text": [context[start.item():end.item()+1]]}, 
#             "id": str(idx)} for idx, (start, end, context) in enumerate(zip(start_positions, 
#                                                                         end_positions, 
#                                                                         contexts))]
        
#         # Metrics
#         self.squad(preds, targets)
#         metrics_results = self.squad.compute()
#         f1 = metrics_results["f1"]
#         em = metrics_results["exact_match"]
        
#         # Logging
#         self.log("test_loss", loss, on_epoch=True, on_step=False, batch_size=128)
#         self.log("test_f1", f1, on_epoch=True, on_step=False, batch_size=128)
#         self.log("test_em", em, on_epoch=True, on_step=False, batch_size=128)

#         return loss
    

#     def predict_step(self, batch: Any, batch_idx, dataloader_idx=None) -> torch.tensor:
#         tensor_data, contexts = batch
#         input_ids, attention_mask, start_positions, end_positions = tensor_data

#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#         }
#         with torch.no_grad():
#             outputs = self.forward(**inputs)

#         # These are the predicted start and end positions
#         start_preds = torch.argmax(outputs.start_logits, dim=1)
#         end_preds = torch.argmax(outputs.end_logits, dim=1)

#         return start_preds, end_preds
    

#     def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_val, eps=self.eps_val)
#         lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                                        num_warmup_steps=self.warmup * self.scheduler_training_steps if self.warmup is not None else 1256, 
#                                                        num_training_steps=self.scheduler_training_steps) 
#         return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


#     def lr_scheduler_step(self, scheduler, optimizer_idx, *args, **kwargs):
#         """
#         Needs to be overwritten due to non LambdaLR scheduler in configure_optimizers.
#         Transformer scheduler get_linear_schedule_with_warmup is not compatible with the PL workflow.
#         """
#         scheduler.step()