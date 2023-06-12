"""Rename preset viewport module."""
import json
import os
import shutil

from PyQt5.QtWidgets import (
        QDialog, QGridLayout, QLabel, QLineEdit, QPushButton,
        )

from ..bases import BaseUtility
from ..settings import METADATA_FILENAME, PROCESSED_DATA_FOLDER
from ..utils import sanitize_preset_name


class RenamePresetViewport(QDialog, BaseUtility):
    """Dialog window to rename presets."""

    def __init__(self, mainhub, **kwargs):
        """Rename preset viewport init method."""
        QDialog.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        self._preset_name_to_change = None
        self.mainhub = mainhub
        self.build_gui()

    @property
    def new_name(self):
        """The new name entered."""
        new = self.new_name_lineedit.text().replace(" ", "_")
        return sanitize_preset_name(new)

    @property
    def preset_name_to_change(self):
        """The preset name to change."""
        return self._preset_name_to_change

    @preset_name_to_change.setter
    def preset_name_to_change(self, name):
        self._preset_name_to_change = sanitize_preset_name(name)  # actual name
        name = sanitize_preset_name(name, reverse=True)
        self.setWindowTitle(f"Renaming preset: '{name}'")
        self.oldname_label.setText(f"{name} ->")

    def build_gui(self):
        """Builds the window interface."""
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.new_name_label = QLabel("New name:")
        self.layout.addWidget(self.new_name_label, 0, 0, 1, 1)

        self.oldname_label = QLabel()
        self.layout.addWidget(self.oldname_label, 1, 0, 1, 1)
        self.new_name_lineedit = QLineEdit()
        self.new_name_lineedit.setText("new_name")
        self.layout.addWidget(self.new_name_lineedit, 1, 1, 1, 1)

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self.rename)
        self.layout.addWidget(self.rename_btn, 2, 0, 1, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        self.layout.addWidget(self.cancel_btn, 2, 1, 1, 1)

    def rename(self):
        """Actually rename the folders."""
        self._logger.info(
                f"Renaming preset: {self.preset_name_to_change} -> {self.new_name}")
        for framedir in self.mainhub.frame_dirs:
            processed_data_dir = os.path.join(
                    framedir, PROCESSED_DATA_FOLDER)
            if self.preset_name_to_change not in os.listdir(processed_data_dir):
                continue
            newdir = os.path.join(processed_data_dir, self.new_name)
            self.move_directory(os.path.join(processed_data_dir, self.preset_name_to_change), newdir)
            # also change the preset name in metadatafile
            metadatafile = os.path.join(newdir, METADATA_FILENAME)
            with open(metadatafile, "r") as f:
                metadata = json.load(f)
            metadata["preset_name"] = self.new_name
            with open(os.path.join(newdir, METADATA_FILENAME), "w") as f:
                json.dump(metadata, f, indent=4)
        # refresh pc viewport
        self.mainhub.pc_viewport.refresh_viewport()
        self.close()

    def move_directory(self, oldpath, newpath):
        """Move a single directory."""
        if os.path.exists(newpath):
            # overwrite
            shutil.rmtree(newpath)
        os.makedirs(newpath)
        for filename in os.listdir(oldpath):
            shutil.move(os.path.join(oldpath, filename), newpath)
        shutil.rmtree(oldpath)
