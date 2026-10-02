import pytest
from src.datamodules.glue_datamodule import GLUEDataModuleBase, SST2DataModule, MRPCDataModule, QNLIDataModule, MNLIDataModule


DATA_MODULES = {
    "sst2": SST2DataModule,
    "mrpc": MRPCDataModule,
    "qnli": QNLIDataModule,
    "mnli": MNLIDataModule
}


@pytest.fixture(params=DATA_MODULES.keys()) 
def glue_dataset(request):
    """
    Fixture for creating an instance of GLUEDataModuleBase.
    """
    return request.param


@pytest.fixture
def glue_config(glue_dataset):
    """ 
    Fixture for creating a sample configuration dictionary for the selected GLUE dataset.    
    """
    return {
        "glue_dataset": glue_dataset,
        "batch_size": 8,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "num_splits": 3,
        "truncate": 100
    }

    
@pytest.fixture
def datamodule(glue_dataset, glue_config):
    """ 
    Fixture for creating an instance of the selected GLUEDataModuleBase class.
    """
    return DATA_MODULES[glue_dataset](**glue_config)

    
def test_datamodule_initialization(datamodule, glue_config):
    """ 
    Test for the initialization of the GLUEDataModuleBase class.
    """
    assert datamodule.glue_dataset == glue_config["glue_dataset"]
    assert datamodule.batch_size == glue_config["batch_size"]
    assert datamodule.num_workers == glue_config["num_workers"]
    assert datamodule.model_type == glue_config["model_type"]
    assert datamodule.num_splits == glue_config["num_splits"]
    assert datamodule.truncate == glue_config["truncate"]

    
def test_datamodule_prepare_data(glue_dataset, datamodule):
    """ 
    Test for the prepare_data method of the GLUEDataModuleBase class.
    """
    datamodule.prepare_data()

    # Check if the train and validation splits of the dataset are loaded. 
    # For MNLI, validation = validation_matched.
    if glue_dataset == "mnli":
        assert "train" in datamodule.dataset
        assert "validation_matched" in datamodule.dataset
    else:
        assert "train" in datamodule.dataset
        assert "validation" in datamodule.dataset


def test_datamodule_setup(datamodule, glue_config):
    """ 
    Test for the setup method of the GLUEDataModuleBase class. Includes truncation check.
    """
    datamodule.prepare_data()
    datamodule.setup()

    # Check the length of the dataset after truncation
    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0

    if glue_config["truncate"] is not None:
        assert len(datamodule.train_dataset) == glue_config["truncate"]
        assert len(datamodule.val_dataset) == glue_config["truncate"]


def test_datamodule_tokenization(glue_dataset, datamodule):
    """ 
    Test for the tokenization method of the GLUEDataModuleBase class.
    """
    datamodule.prepare_data()
    datamodule.setup()

    train_dl = datamodule.train_dataloader()[0]

    # Check for the presence of tokenized attributes in the dataset
    batch = next(iter(train_dl))
    input_ids, attention_mask, labels = batch

    # Check labels for the GLUE datasets. 
    # Except for MNLI, all datasets are binary.
    if glue_dataset == "mnli":
        assert labels[0].item() in [0, 1, 2]
    else:
        assert labels[0].item() in [0, 1] 


def test_datamodule_splitting(datamodule, glue_config):
    """ 
    Test for the split method of the GLUEDataModuleBase class.
    """
    datamodule.prepare_data()
    datamodule.setup()

    train_dls = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()

    # Check if the number of splits is correct
    assert len(train_dls) == glue_config["num_splits"]
    assert len(val_dls) == glue_config["num_splits"]

    # Check if the length of each split is correct
    for dl in train_dls:
        assert len(dl) > 0
        assert len(dl) <= glue_config["truncate"]

