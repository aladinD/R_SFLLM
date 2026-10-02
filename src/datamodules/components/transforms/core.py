from typing import Any, Callable, List

import torch
from einops import rearrange

from .base import BaseTransform, Input


class ComplexToReal(BaseTransform):
    def __init__(self, rearrange: bool = True, pattern: str = "b c w h d -> b (c d) w h") -> None:
        """Converts complex samples and labels to real tensors. Optionally performs rearrangement
        of samples. We assume the samples have already been batched, i.e. have following shape:
        (batch_size, num_rx_ant, num_carriers, num_symbols) and :attr:`complex dtype`.

        Thus this transform is best used
        when passed to :attr:`datamodules.components.transform_collate`.

        The result of this transform will map the complex input tensor, to
        a real tensor with shape (batch_size, num_rx_ant, num_carriers, num_symbols, 2).
        If :attr:`rearrange` is true, then the input tensor will have shape
        (batch_size, 2 * num_rx_ant, num_carriers, num_symbols).
        You can control the way this is rearranged by modifying the
        :attr:`pattern` parameter.

        An in-depth explanation can be found here:
        https://einops.rocks/api/rearrange/

        :param rearrange: Whether to rearrange the complex input sample, defaults to True.
        :type rearrange: bool, optional
        :param pattern: How the complex input sample is rearranged, defaults to "b c w h d -> b (c d) w h".
        :type pattern: str, optional
        """
        self.rearrange = rearrange
        self.pattern = pattern

    def validate_input(self, inp: Input):
        assert isinstance(
            inp, tuple
        ), "Input must be a tuple \
            containing a sample and the ground truth!"
        x, gt = inp
        assert isinstance(x, torch.Tensor)
        assert isinstance(gt, dict), "Currently only dicts supported as ground truth!"
        assert torch.is_complex(x)
        assert (
            x.ndim == 4
        ), f"Shape must contain 4 elements, (batch_size, num_rx_ant, num_carriers, num_symbols) but contains {x.ndim} elements!"

    def apply(self, inp: Input):
        x, gt = inp
        x_real = torch.view_as_real(x).to(dtype=torch.float32)
        gt_real = {}
        for k, v in gt.items():
            gt_real[k] = (
                torch.view_as_real(v).to(dtype=torch.float32) if torch.is_complex(v) else v
            )

        if self.rearrange:
            x_real = rearrange(tensor=x_real, pattern=self.pattern)

        return x_real, gt_real


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        """Applies multiple transforms sequentially.

        :param transforms: List of transforms
        :type transforms: list[Callable]
        """
        self.transforms = transforms

    def __call__(self, inp: Any) -> Any:
        for t in self.transforms:
            inp = t(inp)
        return inp
