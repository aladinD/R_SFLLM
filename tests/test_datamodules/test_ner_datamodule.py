import pytest
from src.datamodules.ner_datamodule import NERDataModuleBase, CoNLL2003DataModule, WNUT17DataModule, OntoNotesDataModule


DATA_MODULES = {
    "conll2003": CoNLL2003DataModule,
    "wnut_17": WNUT17DataModule,
    "conll2012_ontonotesv5": OntoNotesDataModule
}


@pytest.fixture(params=DATA_MODULES.keys()) 
def ner_dataset(request):
    """
    Fixture for creating an instance of NERDataModuleBase.
    """
    return request.param


@pytest.fixture
def ner_config(ner_dataset):
    """ 
    Fixture for creating a sample configuration dictionary for the selected NER dataset.    
    """
    return {
        "ner_dataset": ner_dataset,
        "batch_size": 8,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "num_splits": 3,
        "truncate": 100
    }

    
@pytest.fixture
def datamodule(ner_dataset, ner_config):
    """ 
    Fixture for creating an instance of the selected NERDataModuleBase class.
    """
    return DATA_MODULES[ner_dataset](**ner_config)

    
def test_datamodule_initialization(datamodule, ner_config):
    """ 
    Test for the initialization of the NERDataModuleBase class.
    """
    assert datamodule.ner_dataset == ner_config["ner_dataset"]
    assert datamodule.batch_size == ner_config["batch_size"]
    assert datamodule.num_workers == ner_config["num_workers"]
    assert datamodule.model_type == ner_config["model_type"]
    assert datamodule.num_splits == ner_config["num_splits"]
    assert datamodule.truncate == ner_config["truncate"]

    
def test_datamodule_prepare_data(ner_dataset, datamodule):
    """ 
    Test for the prepare_data method of the NERDataModuleBase class.
    """
    datamodule.prepare_data()

    # Check if the train and validation splits of the dataset are loaded. 
    assert "train" in datamodule.dataset
    assert "validation" in datamodule.dataset


def test_datamodule_setup(datamodule, ner_config):
    """ 
    Test for the setup method of the NERDataModuleBase class. Includes truncation check.
    """
    datamodule.prepare_data()
    datamodule.setup()

    # Check the length of the dataset after truncation
    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0

    if ner_config["truncate"] is not None:
        assert len(datamodule.train_dataset) == ner_config["truncate"]
        assert len(datamodule.val_dataset) == ner_config["truncate"]


def test_datamodule_tokenization(ner_dataset, datamodule):
    """ 
    Test for the tokenization method of the NERDataModuleBase class.
    """
    datamodule.prepare_data()
    datamodule.setup()

    train_dl = datamodule.train_dataloader()[0]

    # Check for the presence of tokenized attributes in the dataset
    batch = next(iter(train_dl))
    input_ids, attention_mask, labels = batch

    # Check labels for the GLUE datasets. 
    # Specific to dataset : {"conll2003": 10, "wnut_17": 14, "conll2012_ontonotesv5" : 38}
    for label in labels[0]:
        if ner_dataset == "conll2003":
           assert label.item() in range(10)
        elif ner_dataset == "wnut_17":
            assert label.item() in range(14)
        elif ner_dataset == "conll2012_ontonotesv5":
           assert label.item() in range(38) 


def test_datamodule_splitting(datamodule, ner_config):
    """ 
    Test for the split method of the NERDataModuleBase class.
    """
    datamodule.prepare_data()
    datamodule.setup()

    train_dls = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()

    # Check if the number of splits is correct
    assert len(train_dls) == ner_config["num_splits"]
    assert len(val_dls) == ner_config["num_splits"]

    # Check if the length of each split is correct
    for dl in train_dls:
        assert len(dl) > 0
        assert len(dl) <= ner_config["truncate"]