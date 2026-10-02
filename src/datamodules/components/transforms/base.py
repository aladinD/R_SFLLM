from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

# TODO: replace Any with torch.Tensor or isac.ISACTensor
# can be thought of the input tensor
Sample = Any
# can be thought of as a label,
# we prefer dictionaries, since they are more readable
GroundTruth = Dict[str, Any]
# input to the network consists of samples and ground truth
Input = Tuple[Sample, GroundTruth]
# batch containing samples and ground truth
Batch = List[Input]


class BaseTransform(ABC):
    """Base class for all transforms. Note that unlike torchvision, we do not make the distinction
    between transforms on samples (i.e. tensors) and transforms on targets (labels). We assume the
    child class implements both, for following reasons:

    1. It is arguably faster.
    2. In certain applications (object detection) transformation of the tensor,
    also changes the label.
    Think of how rotating an image affects the bounding boxes in the annotation
    3. Sometimes the transformation applied on one input depends on information
    about other inputs in the batch (think of SSL).
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def validate_input(self, inp):
        """Checks if the input has correct attributes."""
        ...

    @abstractmethod
    def apply(self, inp):
        """Actual transformation of the input."""
        ...

    def __call__(self, inp):
        """Validates the input and applies transformation."""
        self.validate_input(inp=inp)
        return self.apply(inp=inp)
