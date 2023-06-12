"""Camera image processor."""
import os

import numpy as np
from PIL import Image

from .bases import BaseProcessor
from ..settings import (
        CAM_FILENAME,
        PROCESSED_CAM_FILENAME,
        PROCESSED_DATA_FOLDER,
        RAW_DATA_FOLDER,
        )


class CameraProcessor(BaseProcessor):
    """Actually to the processing for the RGB images in a frame."""

    def __init__(self, camtype="all", **kwargs):
        """Image processor init method."""
        super().__init__(**kwargs)
        self.avail_camtypes, self.yaws = self._get_camtypes_and_yaws()
        if camtype == "all":
            self.requested_camtypes = self.avail_camtypes
        else:
            if camtype not in self.avail_camtypes:
                raise ValueError(f"Camtype not available: {camtype}.")
            self.requested_camtypes = [camtype]

    def process(self, overwrite):
        """Process image."""
        for camtype in self.requested_camtypes:
            for yaw in self.yaws:
                filepath = os.path.join(self.framedir, RAW_DATA_FOLDER, camtype, yaw, CAM_FILENAME)
                dirname = os.path.join(self.framedir, PROCESSED_DATA_FOLDER, camtype, yaw)
                os.makedirs(dirname, exist_ok=True)
                out_path = os.path.join(dirname, PROCESSED_CAM_FILENAME)
                if os.path.isfile(out_path):
                    if overwrite:
                        os.remove(out_path)
                    else:
                        continue
                arr = np.load(filepath)["arr_0"]
                image = Image.fromarray(arr)
                image.save(out_path)

    def _get_camtypes_and_yaws(self):
        # check framedir for camtypes
        camtypes = []
        for name in os.listdir(os.path.join(self.framedir, RAW_DATA_FOLDER)):
            if not os.path.isdir(os.path.join(self.framedir, RAW_DATA_FOLDER, name)):
                continue
            camtypes.append(name)
        yaws = []
        for filename in os.listdir(os.path.join(self.framedir, RAW_DATA_FOLDER, camtypes[0])):
            fp = os.path.join(self.framedir, RAW_DATA_FOLDER, camtypes[0], filename)
            if not os.path.isdir(fp):
                continue
            yaws.append(filename)
        return camtypes, yaws
