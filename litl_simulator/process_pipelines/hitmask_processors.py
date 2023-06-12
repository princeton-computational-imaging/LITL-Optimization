"""Hitmask processors module.

Their responsability is to compute the hitmask of the oversampled point cloud (raw).
This dictates which points are dropped and which ones will be present in the final
point cloud.
"""
from .bases import BaseLidarProcessor


class BasicHitMaskProcessor(BaseLidarProcessor):
    """Basic hit mask processor only drops the 'no hit' pts.

    These pts are tagged with cos(hit angle) = 2 which is impossible!
    """

    def process(self):
        """Computes the hitmask."""
        # where cos(theta) == 2 means no hit were registered => I = 0
        self.hitmask = self.raw_pc[..., 3] < 2.0
        if not self.hitmask.any():
            raise ValueError("No hits...")
