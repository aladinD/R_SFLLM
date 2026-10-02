import torch
from torch.nn.parameter import Parameter
from typing import List, Dict
from pytorch_lightning import LightningModule


class Aggregator:
   """
   Aggregator class for SFL. Handles the aggregation of client gradients.
   """
   def __init__(self, name: str) -> None:
        self.name = name

      
   def accumulate_attentions(self, client_models: List[LightningModule]) -> List[Dict]:
    """
    Accumulates/Collects the attentions of all clients and returns a List of 
    Dicts including the parameter names and values as a key-value pair.
    """
    attentions = []

    for model in client_models:
        attention_gradients = {}
        for name, param in model.named_parameters():
            if name.startswith("bert.encoder"):
                attention_gradients[name] = param

        attentions.append(attention_gradients)

    return attentions
   
   
   def accumulate_heads(self, client_models: List[LightningModule]) -> List[Dict]:
    """
    Accumulates/Collects the heads (classifier and pooler) of all clients and 
    returns a List of Dicts including the parameter names and values as a 
    key-value pair.
    """
    heads = []

    for model in client_models:
        head_gradients = {}
        for name, param in model.named_parameters():
            if name.startswith("classifier") or name.startswith("bert.pooler"):
                head_gradients[name] = param

        heads.append(head_gradients)

    return heads


   def accumulate_embeddings(self, client_models: List[LightningModule]) -> List[Dict]:
    """
    Accumulates/Collects the embeddings of all clients and returns a List of 
    Dicts including the parameter names and values as a key-value pair.
    """
    embeddings = []

    for model in client_models:
        embedding_gradients = {}
        for name, param in model.named_parameters():
            if name.startswith("bert.embeddings"):
                embedding_gradients[name] = param

        embeddings.append(embedding_gradients)

    return embeddings

   
   def aggregate(self, client_gradients: List[Dict[str, Parameter]]) -> Dict[str, Parameter]:
      """
      Aggregates the parameters of the clients by simple averaging.
      """
      aggregated_gradients = {}
      num_clients = len(client_gradients)

      for client_gradient in client_gradients:
         # Iterate over each parameter gradient in the client's gradients
         for param_name, gradient in client_gradient.items():
               if param_name not in aggregated_gradients:
                  # Initialize the tensor to store the average
                  aggregated_gradients[param_name] = torch.zeros_like(gradient.data)
               
               # Accumulate the gradient values across all clients
               aggregated_gradients[param_name] += gradient.data

      # Compute the average for each parameter gradient
      for param_name in aggregated_gradients:
         aggregated_gradients[param_name] /= num_clients

      return aggregated_gradients