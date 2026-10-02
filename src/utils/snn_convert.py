import torch
from spikingjelly.activation_based import ann2snn
from torch import fx
from torch.utils.data import IterableDataset
from tqdm import tqdm


class SNNConverter(ann2snn.Converter):
    """Custom Converter class to handle IterableDataset."""

    def __init__(self, dataloader, device=None, mode="Max", momentum=0.1, fuse_flag=True):

        super().__init__(dataloader, device, mode, momentum, fuse_flag)

    def forward(self, ann: torch.nn.Module):
        if self.device is None:
            self.device = next(ann.parameters()).device
        ann = fx.symbolic_trace(ann).to(self.device)
        ann.eval()
        ann_fused = self.fuse(ann, fuse_flag=self.fuse_flag).to(self.device)
        ann_with_hook = self.set_voltagehook(ann_fused, momentum=self.momentum, mode=self.mode).to(
            self.device
        )
        # check if iterable dataset and if yes, then only sample 10 batches for conversion
        assert self.dataloader.dataset
        is_iterable_dataset = isinstance(self.dataloader.dataset, IterableDataset)
        if is_iterable_dataset:
            for _ in tqdm(range(10)):
                imgs, _ = next(iter(self.dataloader))
                ann_with_hook(imgs.to(self.device))
        else:
            for _, (imgs, _) in enumerate(tqdm(self.dataloader)):
                ann_with_hook(imgs.to(self.device))
        snn = self.replace_by_ifnode(ann_with_hook).to(self.device)
        return snn  # return type: GraphModule
