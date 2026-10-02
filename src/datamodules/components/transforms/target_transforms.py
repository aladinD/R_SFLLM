"""Transforms which only operate on the labels, e.g. converting the angles from degrees to radians,
normalizing them between -1 and 1, returning only azimuth and elevation."""

import torch

from .base import BaseTransform, Input


class NormalizeAngles(BaseTransform):
    def __init__(
        self, normalize: bool = False, azimuth_only: bool = False, elevation_only: bool = False
    ) -> None:
        """Processes the angles in the ground truth dict, in following ways:

            - Extraction of azimuth or elevation
            - Normalization between -1 and 1 for both azimuth and elevation

        Azimuth angles are normalized by simple division by pi, while elevation angles
        are divided by 0.5*pi.

        We assume the angles are given in radians.
        While we assume the input has already been batched, i.e. this transform is best used
        when passed to :attr:`datamodules.components.transform_collate`, you can also use it
        as a standalone.

        This target transform is done in-place, i.e. the ground truth is directly modified,
        rather than returning a modified copy

        :param normalize: Whether to normalize the input angles, defaults to False
        :type normalize: bool, optional
        :param azimuth_only: Whether to only return the azimuth of departure/arrival, defaults to False
        :type azimuth_only: bool, optional
        :param elevation_only: Whether to only return the azimuth of departure/arrival, defaults to False
        :type elevation_only: bool, optional
        """
        super().__init__()
        assert not (
            (azimuth_only is True) and (elevation_only is True)
        ), "Only one of azimuth_only or elevation_only can be set True!"

        self.normalize = normalize
        self.azimuth_only = azimuth_only
        self.elevation_only = elevation_only

        # names of the angles as they appear in the ground truth dict
        self.names = ("aoas", "aods")

    def validate_input(self, inp: Input):
        _, gt = inp
        assert isinstance(gt, dict), "Ground truth must be a dict!"
        assert all(name in gt.keys() for name in self.names)
        assert all(
            gt[name].shape[-1] == 2 for name in self.names
        ), "Angles must be specified as a (D, 2) tensor a where a[0] denotes the azimuth dimension and a[1] the elevation"

    def apply(self, inp):
        x, gt = inp
        for name in self.names:
            if self.normalize:
                gt[name][..., 0] /= torch.pi
                gt[name][..., 1] /= 0.5 * torch.pi

            angles = gt[name]
            if self.azimuth_only:
                gt[name] = angles[..., 0]

            if self.elevation_only:
                gt[name] = angles[..., 1]

        return x, gt


class NormalizeDelays(BaseTransform):
    def __init__(self, t_sym) -> None:
        """Normalize the path delays in the ground truth between 0 and t_sym/2.

        :param t_sym: Symbol time.
        :type t_sym: float
        :param normalize: _description_, defaults to False
        :type normalize: bool, optional
        """
        self.t_sym = t_sym

        # names of the timedelay as they appear in the ground truth dict
        self.names = ("path_delays",)

    def validate_input(self, inp: Input):
        _, gt = inp
        assert isinstance(gt, dict), "Ground truth must be a dict!"
        assert "path_delays" in gt.keys(), "Ground truth must contain path delays!"
        assert gt["path_delays"].shape[-1] > 0, "Path delays tensor is empty!"

    # t_sym as symbol duration
    def apply(self, inp):
        x, gt = inp
        for name in self.names:
            gt[name] /= 0.5 * self.t_sym
        return x, gt
