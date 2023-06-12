"""Delete presets module."""
import os
import shutil
import time

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
        QCheckBox, QDialog, QGridLayout, QLabel, QProgressBar, QPushButton, QWidget,
        )

from .custom_widgets import LineWidget
from .bases import BaseTaskManager
from ..bases import BaseUtility
from ..settings import (
        PRESET_SUBFOLDER_BASENAME,
        PROCESSED_DATA_FOLDER,
        )
from ..utils import sanitize_preset_name


SI_PREFACTORS = ["", "k", "M", "G", "T"]


def complexify_scalar(value, si_prefactor, order_factor=1000):
    """Returns the base value with stipped SI factor."""
    order_of_magnitude = SI_PREFACTORS.index(si_prefactor)
    return value * order_factor ** order_of_magnitude


def simplify_scalar(value, order_factor=1000):
    """Returns simplified size and units from initial size in bytes."""
    units = "b"
    if value == 0:
        return value, units
    order_of_magnitude = 0
    lower_bound = order_factor ** order_of_magnitude
    higher_bound = order_factor ** (order_of_magnitude + 1)
    while not (value > lower_bound and value <= higher_bound):
        order_of_magnitude += 1
        lower_bound = higher_bound
        higher_bound = order_factor ** (order_of_magnitude + 1)
    return round(value / lower_bound), SI_PREFACTORS[order_of_magnitude] + units


class DeletePresetsProgressDialog(QDialog, BaseUtility):
    """Progress dialog for deleting presets."""

    def __init__(self, parent, presets_data, **kwargs):
        """Delete presets progress dialog init method."""
        QDialog.__init__(self, parent)
        self.presets_data = presets_data
        BaseUtility.__init__(self, **kwargs)
        self.build_gui()

    def build_gui(self):
        """Build the UI."""
        self.setWindowTitle("Deleting presets in progress...")
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        # create progress bars
        total = 0
        self.progress_bars = {
                "total": {
                    "bar": QProgressBar(),
                    "label": QLabel("Total progress:"),
                    },
                }
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.progress_bars["total"]["label"], 0, 0, 1, 1)
        self.layout.addWidget(self.progress_bars["total"]["bar"], 0, 1, 1, 1)
        ipreset = 0
        for ipreset, (preset, data) in enumerate(self.presets_data.items()):
            total += data["nframes"]
            self.progress_bars[preset] = {
                    "bar": QProgressBar(),
                    "label": QLabel(f"{preset} progress:"),
                    }
            self.progress_bars[preset]["bar"].setRange(0, data["nframes"] - 1)
            self.layout.addWidget(self.progress_bars[preset]["label"], ipreset + 1, 0, 1, 1)
            self.layout.addWidget(self.progress_bars[preset]["bar"], ipreset + 1, 1, 1, 1)
        self.progress_bars["total"]["bar"].setRange(0, total - 1)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.layout.addWidget(self.close_btn, ipreset + 2, 0, 1, -1)

    def increment(self, preset):
        """Increment progress bars."""
        self.increment_bar(preset)
        self.increment_bar("total")

    def increment_bar(self, bar):
        """Increment a bar."""
        value = self.progress_bars[bar]["bar"].value()
        self.progress_bars[bar]["bar"].setValue(value + 1)


class DeletePresetTaskManager(BaseTaskManager):
    """Delete preset task manager."""
    all_jobs_finished = pyqtSignal(object)
    finished_deleting_a_directory = pyqtSignal(object)

    def __init__(self, presets_to_delete, mainhub, **kwargs):
        """Task manager init method."""
        super().__init__(**kwargs)
        self.presets_to_delete = presets_to_delete
        self.all_paths = {}
        self.njobs = 0
        for preset_name in presets_to_delete:
            self.futures[preset_name] = []
            self.all_paths[preset_name] = []
            true_preset_name = sanitize_preset_name(preset_name)
            for framedir in mainhub.frame_dirs:
                processed_dir = os.path.join(framedir, PROCESSED_DATA_FOLDER)
                if not os.path.isdir(processed_dir):
                    continue
                if true_preset_name not in os.listdir(processed_dir):
                    continue
                self.all_paths[preset_name].append(os.path.join(processed_dir, true_preset_name))
                self.njobs += 1
        self.jobs_done = 0

    def start_jobs(self):
        """Submits all jobs."""
        for preset_name, paths in self.all_paths.items():
            for path in paths:
                self.submit(
                    self.delete_preset, path,  done_callback=self.emit_signal,
                    futures_name=preset_name,
                    )

    def delete_preset(self, path):
        """Delete a preset dir."""
        shutil.rmtree(path)
        time.sleep(0.01)

    def emit_signal(self, future):
        """Emit signal that a dir have been deleted."""
        # check for which preset this future was part of
        for preset_name, futures in self.futures.items():
            if future in futures:
                self.finished_deleting_a_directory.emit(preset_name)
                break
        else:
            raise ValueError("Could not find future...")
        self.jobs_done += 1
        if self.jobs_done == self.njobs:
            self.all_jobs_finished.emit(1)


class DeletePresetsDialog(QDialog, BaseUtility):
    """Dialog for deleting presets."""
    def __init__(self, mainhub):
        """Main dialog init method."""
        QDialog.__init__(self, mainhub)
        self.mainhub = mainhub
        BaseUtility.__init__(self, loglevel=mainhub._loglevel)
        self.build_gui()

    def build_gui(self):
        """Build the GUI."""
        self.setWindowTitle("Delete presets data")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.presets_data_widget = PresetsDataLayout(
                self.mainhub.frame_dirs, loglevel=self._loglevel)
        self.layout.addWidget(self.presets_data_widget, 0, 0, 1, -1)
        self.delete_btn = QPushButton("Delete presets")
        self.delete_btn.clicked.connect(self.delete_presets)
        self.delete_btn.setEnabled(False)  # disabled by default
        self.layout.addWidget(self.delete_btn, 1, 0, 1, 1)
        self.cancel_btn = QPushButton("Skra pop pop")
        self.cancel_btn.clicked.connect(self.close)
        self.layout.addWidget(self.cancel_btn, 1, 1, 1, 1)
        self.presets_data_widget.any_presets_checked.connect(self.update_delete_btn_text)

    def delete_presets(self):
        """Delete presets data."""
        to_delete = self.presets_data_widget.get_data_to_delete()
        self.progress_dialog = DeletePresetsProgressDialog(
                self,
                to_delete,
                loglevel=self._loglevel,
                )
        self.progress_dialog.show()
        self.task_manager = DeletePresetTaskManager(
                to_delete, self.mainhub, loglevel=self._loglevel)
        self.task_manager.finished_deleting_a_directory.connect(self.progress_dialog.increment)
        self.task_manager.start_jobs()
        self.task_manager.all_jobs_finished.connect(self.close)
        self.task_manager.all_jobs_finished.connect(self.mainhub.pc_viewport.refresh_viewport)
        self.task_manager.all_jobs_finished.connect(self.progress_dialog.close)

    def update_delete_btn_text(self):
        """Update delete btn text with size to be deleted."""
        to_delete = self.presets_data_widget.get_data_to_delete()
        if not to_delete:
            self.delete_btn.setText("Delete presets")
            self.delete_btn.setEnabled(False)
            return
        self.delete_btn.setEnabled(True)
        tot_size = 0
        for preset_data in to_delete.values():
            size = preset_data["total_size"]
            units = preset_data["total_size_units"]
            tot_size += complexify_scalar(size, units.strip("b"), order_factor=1024)
        tot_size, units = simplify_scalar(tot_size, order_factor=1024)
        self.delete_btn.setText(f"Delete presets ({tot_size} {units})")


class PresetsDataLayout(QWidget, BaseUtility):
    """Presets list data layout."""
    any_presets_checked = pyqtSignal(object)

    def __init__(self, framedirs, **kwargs):
        """Data layout init method."""
        QWidget.__init__(self)
        self.framedirs = framedirs
        BaseUtility.__init__(self, **kwargs)
        self.presets_data = self.get_presets_data()
        self.build_gui()

    def build_gui(self):
        """Build the data layout."""
        # main layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # start with headers
        preset_headers = ["Preset", "N frames", "Total Size"]
        self.sep_lines = []
        for iheader, header in enumerate(preset_headers):
            label = QLabel(header)
            if header != preset_headers[0]:
                label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(label, 0, iheader * 2, 1, 1)
            # separator
            sep_line = LineWidget(alignment="vertical")
            self.sep_lines.append(sep_line)
            self.layout.addWidget(sep_line, 0, iheader * 2 + 1, -1, 1)

        self.erase_all_checkbox = QCheckBox("Erase all")
        self.erase_all_checkbox.clicked.connect(self.erase_all_checkbox_clicked)
        self.layout.addWidget(self.erase_all_checkbox, 0, len(preset_headers) * 2, 1, 1)
        sep_line = LineWidget(alignment="horizontal")
        self.layout.addWidget(sep_line, 1, 0, 1, -1)
        self.sep_lines.append(sep_line)

        # main data layout
        # self.scroll_area = QScrollArea()
        self.data_layout = QGridLayout()
        # self.data_layout_widget_container = QWidget()
        # self.scroll_area.setWidget(self.data_layout_widget_container)
        # self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setFixedWidth(400)
        # self.data_layout_widget_container.setLayout(self.data_layout)
        # self.layout.addWidget(self.scroll_area, 1, 0, 1, 1)
        # self.layout.addLayout(self.data_layout, 1, 0, 1, 1)

        self.presets_widgets = {}
        for ipreset, (preset_name, preset_data) in enumerate(self.presets_data.items()):
            checkbox = QCheckBox("erase")
            checkbox.clicked.connect(self.erase_checkbox_clicked)
            self.presets_widgets[preset_name] = {
                    "name_label": QLabel(preset_name),
                    "frames_label": QLabel(str(preset_data["nframes"])),
                    "size_label": QLabel(f"{preset_data['total_size']} ({preset_data['total_size_units']})"),
                    "checkbox": checkbox,
                    }
            ipreset += 2
            self.presets_widgets[preset_name]["name_label"].setAlignment(Qt.AlignLeft)
            self.presets_widgets[preset_name]["frames_label"].setAlignment(Qt.AlignCenter)
            self.presets_widgets[preset_name]["size_label"].setAlignment(Qt.AlignCenter)
            self.layout.addWidget(
                    self.presets_widgets[preset_name]["name_label"], ipreset, 0, 1, 1)
            self.layout.addWidget(
                    self.presets_widgets[preset_name]["frames_label"], ipreset, 2, 1, 1)
            self.layout.addWidget(
                    self.presets_widgets[preset_name]["size_label"], ipreset, 4, 1, 1)
            self.layout.addWidget(checkbox, ipreset, 6, 1, 1)

    def get_data_to_delete(self):
        """Return the dict of the data to delete."""
        data = {}
        for preset_name, preset_widget in self.presets_widgets.items():
            if preset_widget["checkbox"].isChecked():
                data[preset_name] = self.presets_data[preset_name].copy()
        return data

    def erase_all_checkbox_clicked(self):
        """The erase all checkbox have been clicked."""
        for preset_widgets in self.presets_widgets.values():
            checkbox = preset_widgets["checkbox"]
            checkbox.setChecked(self.erase_all_checkbox.isChecked())
        self.any_presets_checked.emit(None)

    def erase_checkbox_clicked(self, state):
        """An erase checkbox was clicked."""
        self.any_presets_checked.emit(None)
        if state is False and self.erase_all_checkbox.isChecked():
            self.erase_all_checkbox.setChecked(state)

    def get_presets_data(self):
        """Get the presets data."""
        data = {}
        for framedir in self.framedirs:
            # list all presets in this dir
            prc_dir = os.path.join(framedir, PROCESSED_DATA_FOLDER)
            if not os.path.isdir(prc_dir):
                continue
            for name in os.listdir(prc_dir):
                if not name.startswith(PRESET_SUBFOLDER_BASENAME):
                    continue
                path = os.path.join(framedir, PROCESSED_DATA_FOLDER, name)
                name = sanitize_preset_name(name, reverse=True)
                if name not in data:
                    size = self.get_dirsize(path)
                    data[name] = {
                            "nframes": 1,
                            "total_size": self.get_dirsize(path),
                            }
                else:
                    data[name]["nframes"] += 1
                    data[name]["total_size"] += self.get_dirsize(path)
        for preset_name, preset_data in data.items():
            size = preset_data["total_size"]
            units = "b"
            if size >= 1024:
                size /= 1024
                units = "kb"
            if size >= 1024:
                size /= 1024
                units = "Mb"
            if size >= 1024:
                size /= 1024
                units = "Gb"
            preset_data["total_size"] = round(size, 2)
            preset_data["total_size_units"] = units
        return data

    def get_dirsize(self, path):
        """Return the size in bytes of this directory."""
        if not os.path.isdir(path):
            raise NotADirectoryError(path)
        size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    size += os.path.getsize(fp)
        return size
