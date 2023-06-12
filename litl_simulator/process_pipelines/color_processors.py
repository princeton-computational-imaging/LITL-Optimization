"""The point cloud coloring processor module."""
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from .bases import BaseLidarProcessor


class ColorProcessor(BaseLidarProcessor):
    """Processed the pc intensities and returns resulting colors."""

    def map_colors_from_intensities(self, intensities, cmap="jet"):
        """Map gray intensities to a color map."""
        # convert to colors
        min_, max_ = 0, 255
        norm = mpl.colors.Normalize(vmin=min_, vmax=max_)
        # go here for a list of colormaps
        intensities = np.copy(intensities)
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = cm.ScalarMappable(norm=norm, cmap=getattr(cm, cmap))
        max_ = np.max(intensities)
        if max_ != 0.0:
            intensities /= max_
        if intensities.ndim > 2:
            # this pc was not downsampled
            colors = []
            for subray in intensities:
                colors.append(cmap.to_rgba(subray, norm=False))
            return np.asarray(colors)
        return cmap.to_rgba(intensities, norm=False)

    def get_colors_from_intensities(self, intensities, **kwargs):
        """Compute color array from intensities."""
        if intensities.shape[-1] != 4:
            # intensities is an array of floats representing intensities
            opacity = 0.5 * np.ones_like(intensities)
            opacity[intensities == 0.0] = 0
            # raw intensity (gray tone)
            colors = self.map_colors_from_intensities(intensities, **kwargs)
            colors[..., -1] = opacity
            return colors
        else:
            # else: intensities are already colors
            if intensities.dtype == np.uint8:
                # convert to floats
                intensities /= 255
            max_ = np.max(intensities)
            if max_ != 1.0:
                intensities /= max_
            if intensities.shape[-1] == 3:
                # concatenate with opacity
                opacity = np.ones_like(intensities[..., 0], dtype=np.float64)
                return np.concatenate((intensities, opacity), axis=-1)
            elif intensities.shape[-1] == 4:
                # already has opacity
                return intensities
            elif intensities.shape[-1] == 2:
                # gray scale + opacity
                colors = self.map_colors_from_intensities(intensities[..., 0])
                colors[..., -1] = intensities[..., -1]
                return colors
        raise LookupError(f"weird intensities shape: {intensities.shape}")

    def process(self, **kwargs):
        """Process colors."""
        self.processed_colors = self.get_colors_from_intensities(self.raw_intensities, **kwargs)
