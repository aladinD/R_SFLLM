"""Module for collate functions, i.e. functions to create batches of data from datas tructures
other than :attr:`torch.Tensor`.

See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
"""

from typing import Any, Callable, List, Optional

from torch.utils.data import default_collate


def transform_collate(batch: List[Any], transform: Optional[Callable] = None) -> Any:
    """Applies batch-wise transform. This is done by first using default collate to construct the
    batches and then applying the transform.

    :param batch: Batch data, usually a tuple :attr:`(tensor, ground_truth)`
    :type batch: List[Any]
    :param transform: Transform to apply, defaults to None.
    If none given, then default_collate is applied.
    :type transform: Optional[Callable], optional
    :return: Transformed batch
    :rtype: Any
    """
    batch_default = default_collate(batch)
    return (
        transform((batch_default[0], batch_default[1])) if transform is not None else batch_default
    )
