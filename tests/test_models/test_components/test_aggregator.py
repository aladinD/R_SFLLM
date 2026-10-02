import pytest
import torch
from src.models.components.aggregator import Aggregator
from src.models.bert_module import BERTForSequenceClassificationModule
from torch.nn.parameter import Parameter
from pytorch_lightning.utilities.parsing import AttributeDict


@pytest.fixture
def aggregator():
    """
    Fixture for creating an Aggregator instance.
    """
    # Initialize the Aggregator instance for testing
    return Aggregator(name="test_aggregator")


@pytest.fixture
def client_models():
    """
    Fixture for creating a list of sample BERTForSequenceClassification model instances.
    """
    num_clients = 3
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })
    return [BERTForSequenceClassificationModule.from_pretrained(**model_config) for _ in range(num_clients)]


def test_accumulate_attentions(aggregator, client_models):
    """
    Test for the accumulate_attentions method of the Aggregator class.
    """
    # Call the accumulate_attentions method and get the results
    attentions = aggregator.accumulate_attentions(client_models)
    assert isinstance(attentions, list)

    # Check if the attentions list contains the expected number of dictionaries
    assert len(attentions) == len(client_models)

    # Check if each dictionary in the attentions list contains only "bert.encoder" parameters
    for attention in attentions:
        for name, param in attention.items():
            assert name.startswith("bert.encoder")
            assert isinstance(param, Parameter)


def test_accumulate_heads(aggregator, client_models):
    """
    Test for the accumulate_heads method of the Aggregator class.
    """
    # Call the accumulate_heads method and get the results
    heads = aggregator.accumulate_heads(client_models)
    assert isinstance(heads, list)

    # Check if the heads list contains the expected number of dictionaries
    assert len(heads) == len(client_models)

    # Check if each dictionary in the heads list contains only "classifier" and "bert.pooler" parameters
    for head in heads:
        for name, param in head.items():
            assert name.startswith("classifier") or name.startswith("bert.pooler")
            assert isinstance(param, Parameter)


def test_accumulate_embeddings(aggregator, client_models):
    """
    Test for the accumulate_embeddings method of the Aggregator class.
    """
    # Call the accumulate_embeddings method and get the results
    embeddings = aggregator.accumulate_embeddings(client_models)
    assert isinstance(embeddings, list)

    # Check if the embeddings list contains the expected number of dictionaries
    assert len(embeddings) == len(client_models)

    # Check if each dictionary in the embeddings list contains only "bert.embeddings" parameters
    for embedding in embeddings:
        for name, param in embedding.items():
            assert name.startswith("bert.embeddings")
            assert isinstance(param, Parameter)


def test_aggregate(aggregator):
    """
    Test for the aggregate method of the Aggregator class.
    """
    # Create a list of dummy gradients for testing
    client_gradients = [
        {
            "param1": Parameter(torch.randn(10, 5)),
            "param2": Parameter(torch.randn(5, 3)),
        },
        {
            "param1": Parameter(torch.randn(10, 5)),
            "param2": Parameter(torch.randn(5, 3)),
        },
    ]

    # Call the aggregate method and get the aggregated gradients
    aggregated_gradients = aggregator.aggregate(client_gradients)

    # Check if the aggregated_gradients dictionary contains the expected keys
    assert set(aggregated_gradients.keys()) == set(client_gradients[0].keys())

    # Check if the aggregated_gradients values are correct (averaged gradients)
    num_clients = len(client_gradients)
    for param_name in aggregated_gradients:
        expected_average = torch.zeros_like(client_gradients[0][param_name].data)
        for grad_dict in client_gradients:
            expected_average += grad_dict[param_name].data
        expected_average /= num_clients
        assert torch.allclose(aggregated_gradients[param_name], expected_average)
