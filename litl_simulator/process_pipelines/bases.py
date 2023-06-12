"""Base classes for processor module."""
import abc

import numpy as np

from .utils import get_subray_data
from ..bases import BaseUtility
from ..utils import spherical2cartesian


class BaseProcessor(BaseUtility, abc.ABC):
    """Base class for all processors."""

    def __init__(self, framedir=None, **kwargs):
        """Base processor init method."""
        self.framedir = framedir
        BaseUtility.__init__(self, **kwargs)

    @abc.abstractmethod
    def process(self):
        """Process data."""
        pass


class BaseLidarProcessor(BaseProcessor):
    """Base class for lidar processors."""

    def __init__(self, **kwargs):
        """Base lidar processor init method."""
        super().__init__(**kwargs)
        # common data properties
        self._raw_pc = None
        self._raw_intensities = None
        self._raw_tags = None
        self._hitmask = None
        self._processed_pc = None
        self._processed_intensities = None
        self._processed_colors = None

    @property
    def hitmask(self):
        """The hitmask."""
        if self._hitmask is not None:
            return self._hitmask
        raise RuntimeError("Need to set 'hitmask'.")

    @hitmask.setter
    def hitmask(self, hitmask):
        self._hitmask = hitmask

    @property
    def processed_colors(self):
        """The colors array."""
        if self._processed_colors is not None:
            return self._processed_colors
        raise ValueError("Need to call 'process()' in order to get the colors.")

    @processed_colors.setter
    def processed_colors(self, colors):
        self._processed_colors = colors

    @property
    def processed_intensities(self):
        """The final intensities."""
        if self._processed_intensities is None:
            raise RuntimeError("Need to call 'process()' to get processed_intensities.")
        return self._processed_intensities

    @processed_intensities.setter
    def processed_intensities(self, intensities):
        self._processed_intensities = intensities

    @property
    def processed_pc(self):
        """The final point cloud."""
        if self._processed_pc is None:
            raise RuntimeError("Need to call 'process()' to get processed_pc.")
        return self._processed_pc

    @processed_pc.setter
    def processed_pc(self, pc):
        self._processed_pc = pc

    @property
    def raw_intensities(self):
        """The raw intensities for this processor."""
        if self._raw_intensities is None:
            raise RuntimeError("Need to set 'raw_intensities'.")
        return self._raw_intensities

    @raw_intensities.setter
    def raw_intensities(self, intensities):
        self._raw_intensities = intensities

    @property
    def raw_pc(self):
        """The raw pc for this processor."""
        if self._raw_pc is None:
            raise RuntimeError("Need to set 'raw_pc'.")
        return self._raw_pc

    @raw_pc.setter
    def raw_pc(self, pc):
        self._raw_pc = pc

    @property
    def raw_tags(self):
        """The raw tags for this processor."""
        if self._raw_tags is None:
            raise RuntimeError("Need to set 'raw_tags'.")
        return self._raw_tags

    @raw_tags.setter
    def raw_tags(self, tags):
        self._raw_tags = tags

    def convert_to_cartesian(self, pc, hitmask):
        """Converts spherical coordinates of point cloud (only hit values) to cartesian."""
        rhos = pc[hitmask, 2]
        phis = pc[hitmask, 1]
        thetas = pc[hitmask, 0]
        pc[hitmask, :3] = spherical2cartesian(rhos, thetas, phis)
        return pc

    def rearrange_data_in_subrays(
            self, *args, upsampling_ratio=3,  # for now no functions uses this feature but TODO in the future
            ):
        """Rearrange given point cloud in arrays of subrays."""
        if not args:
            return
        # need to reshape by number of lasers
        subrays = [[] if arg is not None else None for arg in args]
        for arg, subray in zip(args, subrays):
            if arg is None:
                continue
            for i in range(upsampling_ratio):
                for j in range(upsampling_ratio):
                    subray.append(get_subray_data(arg, i, j, upsampling_ratio))
        if len(args) == 1:
            return np.asarray(subrays[0])
        return (np.asarray(subray) if subray is not None else None for subray in subrays)

    def set_raw_data(self, raw_tags=None, raw_pc=None,  hitmask=None, raw_intensities=None):
        """Sets some raw data properties."""
        if raw_intensities is not None:
            self.raw_intensities = raw_intensities
        if raw_pc is not None:
            self.raw_pc = raw_pc
        if hitmask is not None:
            self.hitmask = hitmask
        if raw_tags is not None:
            self.raw_tags = raw_tags
