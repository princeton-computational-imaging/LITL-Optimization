"""Noise processing module."""
import numpy as np

from .bases import BaseLidarProcessor


class BaseNoiseModelProcessor(BaseLidarProcessor):
    """Base noise processor."""
    model_name = None

    def __init__(self, model=None, std=None, **kwargs):
        """Base noise model init method."""
        super().__init__(**kwargs)
        if self.model_name is None:
            raise ValueError("Need to set cls attribute 'model_name'.")
        if model != self.model_name:
            raise RuntimeError("Noise model mismatch!")
        self.std = float(std)  # in mm


class NoiseLessModelProcessor(BaseNoiseModelProcessor):
    """No noise at all."""
    model_name = "Noiseless"

    def process(self):
        """Applies no noise."""
        self.processed_pc = self.raw_pc
        self.processed_intensities = self.raw_intensities


class GaussianNoiseModelProcessor(BaseNoiseModelProcessor):
    """Adds gaussian noise on XYZ data only."""
    model_name = "Gaussian noise"

    def process(self):
        """Adds gaussian noise to point cloud."""
        shape = self.raw_pc.shape[:-1]
        shape = list(shape) + [3]
        # translate mm to m as point cloud positions are in m
        noise = np.random.normal(loc=0.0, scale=self.std / 1000, size=shape)
        self.processed_pc = self.raw_pc[..., :3] + noise
        self.processed_intensities = self.raw_intensities


AVAILABLE_NOISE_PROCESSOR_CLS = [
        NoiseLessModelProcessor,  # 1st one
        GaussianNoiseModelProcessor,
        ]
NOISE_MODEL_2_NOISE_PROCESSOR_CLS = {x.model_name: x for x in AVAILABLE_NOISE_PROCESSOR_CLS}
