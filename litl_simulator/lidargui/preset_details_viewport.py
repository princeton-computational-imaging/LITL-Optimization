"""Preset details viewport module."""
import json

from PyQt5.QtWidgets import (
        QMessageBox,
        )

from ..bases import BaseUtility
from ..utils import get_metadata


class PresetDetailsViewport(QMessageBox, BaseUtility):
    """Preset details viewport object."""

    def __init__(self, mainhub, **kwargs):
        """Preset details viewport init method."""
        BaseUtility.__init__(self, loglevel=kwargs.pop("loglevel", None))
        QMessageBox.__init__(self, **kwargs)
        self.mainhub = mainhub
        self.preset = self.mainhub.current_preset
        self.framedir = self.mainhub.frame_dirs[self.mainhub.current_frame]
        self.setWindowTitle(f"Preset '{self.preset}' details")
        self.setIcon(QMessageBox.Information)
        self.setText(f"Preset '{self.preset}' details:")
        self.setInformativeText(json.dumps(get_metadata(self.framedir, self.preset), indent=4))
        self.setStandardButtons(QMessageBox.Ok)
