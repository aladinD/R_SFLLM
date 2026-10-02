import torch

from src.datamodules.components.collate_fn import default_collate
from src.datamodules.components.transforms.core import ComplexToReal
from src.datamodules.simulation_datamodule import SimulationDatamodule


def test_simulation_datamodule(batch_size=32):
    dm = SimulationDatamodule(batch_size=batch_size)

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    x, y = next(iter(train_loader))

    assert len(x) == batch_size
    assert all(len(v) == batch_size for v in y.values())
    assert torch.is_complex(x)

    xv, _ = next(iter(val_loader))
    assert not torch.allclose(x, xv)
    assert train_loader.collate_fn == default_collate

    dm_trans = SimulationDatamodule(
        batch_size=batch_size, transforms=ComplexToReal(rearrange=False)
    )
    dm_trans.setup()

    train_loader = dm_trans.train_dataloader()
    x, y = next(iter(train_loader))
    print(y.keys())
    assert len(x) == batch_size
    assert all(len(v) == batch_size for v in y.values())

    # check if correct collate fn was applied
    assert x.shape == torch.Size(
        [
            batch_size,
            dm.dataset.rx_array.num_antennas,
            dm.dataset.ofdm_config.Nfft,
            dm.dataset.num_symbols,
            2,
        ]
    )
    assert torch.isreal(x).all()
    assert all(torch.isreal(v).all() for v in y.values())
