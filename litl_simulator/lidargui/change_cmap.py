"""The change cmap submodule."""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
        QCheckBox, QComboBox, QDialog, QGridLayout, QLabel, QPushButton,
        )
import matplotlib as mpl
import numpy as np
import tqdm

from .custom_widgets import LineWidget
from ..bases import BaseUtility
from ..process_pipelines.color_processors import ColorProcessor
from ..settings import (
        PRESET_SUBFOLDER_BASENAME,
        PROCESSED_COLORS_FILENAME,
        PROCESSED_DATA_FOLDER,
        PROCESSED_INTENSITIES_FILENAME,
        )
from ..utils import sanitize_preset_name


VALID_CMAPS = mpl.colormaps  # ["jet", ]


class ChangeCMAPDialog(QDialog, BaseUtility):
    """The change cmap dialog."""

    def __init__(self, mainhub, **kwargs):
        """Change cmap dialog init method."""
        QDialog.__init__(self, mainhub)
        self.mainhub = mainhub
        BaseUtility.__init__(self, **kwargs)
        self.build_gui()

    def build_gui(self):
        """Builds the dialog window."""
        self.setWindowTitle("Change presets cmap")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.build_cmap_selection(0, 0, 1, 1)
        self.build_presets_section(1, 0, 1, 1)
        self.build_dialog_btns(2, 0, 1, 1)

    def build_cmap_selection(self, *args):
        """Builds the cmap selection layout."""
        self.cmap_selection_layout = QGridLayout()
        self.layout.addLayout(self.cmap_selection_layout, *args)
        self.desc_label = QLabel(
                "<a href=\"https://matplotlib.org/stable/tutorials/colors/colormaps.html\">Click here for cmap descriptions!</a>")  # noqa: E501
        self.desc_label.setTextFormat(Qt.RichText)
        self.desc_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.desc_label.setOpenExternalLinks(True)
        self.cmap_selection_layout.addWidget(self.desc_label, 0, 0, 1, -1)
        self.cmap_selection_label = QLabel("New cmap:")
        self.cmap_selection_layout.addWidget(self.cmap_selection_label, 1, 0, 1, 1)
        self.cmap_dropdown_menu = QComboBox()
        self.cmap_dropdown_menu.addItems(VALID_CMAPS)
        self.cmap_selection_layout.addWidget(self.cmap_dropdown_menu, 1, 1, 1, -1)

    def build_dialog_btns(self, *args):
        """Builds the buttons section."""
        self.btns_layout = QGridLayout()
        self.layout.addLayout(self.btns_layout, *args)
        self.change_cmap_btn = QPushButton("Change cmap")
        self.change_cmap_btn.clicked.connect(self.change_cmap)
        self.change_cmap_btn.setEnabled(False)
        self.btns_layout.addWidget(self.change_cmap_btn, 0, 0, 1, 1)

        self.quit_btn = QPushButton("Skra pop pop")
        self.quit_btn.clicked.connect(self.close)
        self.btns_layout.addWidget(self.quit_btn, 0, 1, 1, 1)

    def build_presets_section(self, *args):
        """Builds the presets selection section."""
        self.presets_selection_layout = QGridLayout()
        self.layout.addLayout(self.presets_selection_layout, *args)
        # start with header
        self.preset_header = QLabel("Preset")
        self.presets_selection_layout.addWidget(self.preset_header, 0, 0, 1, 1)
        self.preset_selection_vline = LineWidget(alignment="vertical")
        self.presets_selection_layout.addWidget(self.preset_selection_vline, 0, 1, -1, 1)
        self.nframes_header_label = QLabel("# frames")
        self.presets_selection_layout.addWidget(self.nframes_header_label, 0, 2, 1, 1)
        self.preset_selection_vline2 = LineWidget(alignment="vertical")
        self.presets_selection_layout.addWidget(self.preset_selection_vline2, 0, 3, -1, 1)
        self.select_all_chkbox = QCheckBox("Select all")
        self.presets_selection_layout.addWidget(self.select_all_chkbox, 0, 4, 1, 1)
        self.select_all_chkbox.clicked.connect(self.on_select_all_chkbox_clicked)
        self.presets_selection_hline = LineWidget(alignment="horizontal")
        self.presets_selection_layout.addWidget(self.presets_selection_hline, 1, 0, 1, -1)
        # now list all presets
        self.presets_checkboxes = {}
        self.presets_nframes_labels = {}
        self.presets_names_labels = {}
        for i, (preset, data) in enumerate(sorted(self.get_presets_data().items())):
            label = QLabel(preset)
            self.presets_names_labels[preset] = label
            self.presets_selection_layout.addWidget(label, 2 + i, 0, 1, 1)
            framelabel = QLabel(str(data["nframes"]))
            self.presets_nframes_labels[preset] = framelabel
            self.presets_selection_layout.addWidget(framelabel, 2 + i, 2, 1, 1)
            checkbox = QCheckBox()
            checkbox.clicked.connect(self.on_any_chkbox_clicked)
            self.presets_checkboxes[preset] = checkbox
            self.presets_selection_layout.addWidget(checkbox, 2 + i, 4, 1, 1)

    @property
    def cmap(self):
        """The selected 'new' cmap."""
        return self.cmap_dropdown_menu.currentText()

    def change_cmap(self):
        """Do the change of cmaps."""
        for preset, checkbox in self.presets_checkboxes.items():
            if not checkbox.isChecked():
                continue
            self._logger.info(f"Changing cmap of preset '{preset}' to '{self.cmap}'.")
            color_processor = ColorProcessor(loglevel=self._loglevel)
            # read intensities
            preset = sanitize_preset_name(preset)
            for framedir in tqdm.tqdm(self.mainhub.frame_dirs):
                if preset not in os.listdir(os.path.join(framedir, PROCESSED_DATA_FOLDER)):
                    continue
                presetdir = os.path.join(framedir, PROCESSED_DATA_FOLDER, preset)
                out_path = os.path.join(presetdir, PROCESSED_COLORS_FILENAME)
                intensities_path = os.path.join(presetdir, PROCESSED_INTENSITIES_FILENAME)
                intensities = np.load(intensities_path)
                color_processor.raw_intensities = intensities
                color_processor.process(cmap=self.cmap)
                newcolors = color_processor.processed_colors
                # save new colors
                np.save(out_path, newcolors)
        self.mainhub.pc_viewport.refresh_only_pc_viewer(force=True)
        self.close()

    def get_presets_data(self):
        """Get presets data."""
        data = {}
        for framedir in self.mainhub.frame_dirs:
            for name in os.listdir(os.path.join(framedir, PROCESSED_DATA_FOLDER)):
                if not name.startswith(PRESET_SUBFOLDER_BASENAME):
                    continue
                name = sanitize_preset_name(name, reverse=True)
                if name not in data:
                    data[name] = {"nframes": 1}
                else:
                    data[name]["nframes"] += 1
        return data

    def on_any_chkbox_clicked(self, state):
        """Method called if any checkbox is clicked (except the select all one)."""
        for chkbox in self.presets_checkboxes.values():
            if chkbox.isChecked():
                if self.change_cmap_btn.isEnabled():
                    return
                self.change_cmap_btn.setEnabled(True)
                return
        self.change_cmap_btn.setEnabled(False)

    def on_select_all_chkbox_clicked(self, state):
        """Method called when the select_all chkbox is clicked."""
        for checkbox in self.presets_checkboxes.values():
            checkbox.setChecked(self.select_all_chkbox.isChecked())
        self.on_any_chkbox_clicked(checkbox.isChecked())
