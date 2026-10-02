import pytest
import torch
from copy import deepcopy
from src.models.roberta_module import RoBERTaForSequenceClassificationModule, RoBERTaForTokenClassificationModule
from src.datamodules.glue_datamodule import SST2DataModule
from src.datamodules.ner_datamodule import CoNLL2003DataModule
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning import Trainer


MODEL_DATAMODULE_MAPPING = {
    "bert_sequence_classification": {
        "model": RoBERTaForSequenceClassificationModule,
        "datamodule": SST2DataModule,
        "num_classes": 2,
        "label_shape": (32,)  # for a batch size of 32
    },
    "bert_token_classification": {
        "model": RoBERTaForTokenClassificationModule,
        "datamodule": CoNLL2003DataModule,
        "num_classes": 10,
        "label_shape": (32, 128)  # for a batch size of 32 and sequence length of 128
    }
}


@pytest.fixture(params=MODEL_DATAMODULE_MAPPING.values(), ids=MODEL_DATAMODULE_MAPPING.keys())
def model_datamodule_config(request):
    """ 
    Fixture for creating the configuration for model and datamodule.
    """
    return request.param


@pytest.fixture
def model(model_datamodule_config):
    """ 
    Fixture for creating an instance of the model.
    """
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": model_datamodule_config["num_classes"]
    })
    
    return model_datamodule_config["model"].from_pretrained(**model_config)


@pytest.fixture
def test_batch(model_datamodule_config):
    """ 
    Fixture for creating a sample batch of input data for testing.
    """
    input_ids = torch.randint(1000, (32, 128))
    attention_mask = torch.ones(32, 128)
    labels_shape = model_datamodule_config["label_shape"]
    labels = torch.randint(model_datamodule_config["num_classes"], labels_shape)

    return input_ids, attention_mask, labels


@pytest.fixture
def datamodule(model_datamodule_config):
    """ 
    Fixture for creating an instance of the appropriate datamodule.
    """
    data_config = AttributeDict({
        "batch_size": 32,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "truncate": 10
    })

    if model_datamodule_config["datamodule"] == SST2DataModule:
        data_config["glue_dataset"] = "sst2"
    elif model_datamodule_config["datamodule"] == CoNLL2003DataModule:
        data_config["ner_dataset"] = "conll2003"

    return model_datamodule_config["datamodule"](**data_config)


def test_forward(model, test_batch):
    """
    Test for the forward method of the model class.
    """
    input_ids, attention_mask, labels = test_batch
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels=labels)

    # Check for necessary keys
    # The output should contain a 'loss' and 'logits' key
    assert "loss" in outputs    
    assert "logits" in outputs 

    # Check for correct loss type and shape
    assert isinstance(outputs["loss"], torch.Tensor) 
    assert outputs["loss"].shape == ()  


def test_forward_w_emb_noise(model, test_batch):
    """
    Test forward pass with embedding noise.
    """
    input_ids, attention_mask, labels = test_batch
    
    # Forward pass
    _ = model(input_ids, attention_mask, labels=labels)
    emb_clean = deepcopy(model.embeddings.word_embeddings.weight.data)

    # Forward pass with added noise for embeddings
    model.add_noise = 100.
    _ = model(input_ids, attention_mask, labels=labels)
    emb_noisy = model.embeddings.word_embeddings.weight.data

    assert not torch.allclose(emb_clean, emb_noisy)


def test_training_with_trainer(model, datamodule):
    """
    Test the training process using PyTorch Lightning's Trainer.
    """
    # Prepare data
    datamodule.prepare_data()
    datamodule.setup()
    train_data = datamodule.train_dataloader()
    val_data = datamodule.val_dataloader()

    # Adjust model parameters
    model.lr_val = 1e-5
    model.eps_val = 1e-6
    model.warmup = 0.1
    model.scheduler_training_steps = 3 * len(datamodule.train_dataloader())

    # Set number of classes based on the model type
    if isinstance(model, RoBERTaForSequenceClassificationModule):
        model.num_classes = 2
    elif isinstance(model, RoBERTaForTokenClassificationModule):
        model.num_classes = 10
    else:
        raise ValueError("Unexpected model type!")
    
    model.init_metrics()

    # Create a trainer instance with some basic configurations
    trainer = Trainer(fast_dev_run=True, devices=[0])

    # # Fit the model using the trainer and data module
    trainer.fit(model, train_data, val_data)
    
    # Assert that the process completed without errors
    assert trainer.state.status == "finished"