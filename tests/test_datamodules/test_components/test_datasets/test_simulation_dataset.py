import pytest
import torch
from torch.utils.data import DataLoader

from src.datamodules.components.datasets.simulation_dataset import SimulationDataset


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_symbols", [14, 64])
@pytest.mark.parametrize("num_carriers", [32, 64])
def test_simulation_dataset(batch_size, num_symbols, num_carriers):
    dset = SimulationDataset(num_symbols=num_symbols, num_carriers=num_carriers)

    dloader = DataLoader(dataset=dset, batch_size=batch_size)

    # check correct shape and dtype
    x1, gt = next(iter(dloader))
    assert x1.shape == torch.Size(
        [batch_size, dset.rx_array.num_antennas, dset.ofdm_config.Nfft, dset.num_symbols]
    )
    assert torch.is_complex(x1)
    assert all((x.shape[0] == batch_size) for x in gt.values())

    # check if the generated data is different for latter invocations
    x2, _ = next(iter(dloader))
    assert not torch.allclose(x1, x2)
