import pytest
import torch

from src.datamodules.components.transforms.core import ComplexToReal


def test_complex_to_real(batch_size=32, n_ant=16, n_fft=64, n_sym=14, n_paths=5):
    shape = [batch_size, n_ant, n_fft, n_sym]
    x = torch.randn(*shape) + 1j * torch.randn(*shape)
    gt = {
        "real": torch.ones(batch_size, n_paths, 2),
        "complex": torch.ones(batch_size, n_paths) + 1j * torch.ones(batch_size, n_paths),
    }
    batch = (x, gt)

    complex2real = ComplexToReal(rearrange=False)
    x_real, gt_real_nr = complex2real(batch)

    assert x_real.shape == torch.Size(shape + [2])
    assert torch.isreal(x_real).all()
    assert isinstance(gt_real_nr, dict)
    assert gt_real_nr.keys() == gt.keys()
    assert all(torch.isreal(v).all() for v in gt_real_nr.values())

    # test with default rearrangement
    complex2real.rearrange = True
    x_real, gt_real_arr = complex2real(batch)

    assert x_real.shape == torch.Size([batch_size, n_ant * 2, n_fft, n_sym])
    # check if ground truth is the same as before
    assert all(
        torch.allclose(v_nr, v_arr)
        for (v_nr, v_arr) in zip(gt_real_nr.values(), gt_real_arr.values())
    )

    # test with another pattern, suitable for nn.Conv3D
    complex2real.pattern = "b c w h d -> b d c w h"
    x_real, gt_real_arr = complex2real(batch)

    assert x_real.shape == torch.Size([batch_size, 2, n_ant, n_fft, n_sym])

    # test with wrong input
    with pytest.raises(AssertionError):
        x_wrong = torch.randn(batch_size, n_fft) + 1j * torch.randn(batch_size, n_fft)
        complex2real((x_wrong, gt))
