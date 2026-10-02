import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader
from src.models.bert_module import BERTForSequenceClassificationModule
from src.models.components.client import Client


@pytest.fixture
def client() -> Client:
    """
    Fixture for creating a Client instance.
    """
    # Create a sample model and trainer for testing
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })
    model = BERTForSequenceClassificationModule.from_pretrained(**model_config)
    trainer = pl.Trainer(max_epochs=3, fast_dev_run=True)

    # Create dummy data for training and validation
    train_data = DataLoader(torch.randn(100, 3), batch_size=1)
    val_data = DataLoader(torch.randn(20, 3), batch_size=1)

    # Initialize the Client instance for testing
    return Client(id=1, model=model, trainer=trainer, train_data=train_data, val_data=val_data)


def test_client_initialization(client: Client):
    """
    Test for the initialization of the Client class.
    """
    assert client.id == 1
    assert isinstance(client.model, BERTForSequenceClassificationModule)
    assert isinstance(client.trainer, pl.Trainer)
    assert isinstance(client.train_data, DataLoader)
    assert isinstance(client.val_data, DataLoader)

        
def test_update_model(client: Client):
    """
    Test for the update_model method of the Client class.
    """
    # Create dummy weight updates as a dictionary
    updates = {
        "layer1.weight": torch.randn(10, 20),
        "layer2.weight": torch.randn(5, 10),
    }

    # Update the model with the dummy weight updates
    client.update_model(updates)

    # Check if the model parameters have been updated correctly
    for gradient_name, updated_param in updates.items():
        for name, param in client.model.named_parameters():
            if name == gradient_name:
                assert torch.allclose(param.data, updated_param)
