"""GUI for point clout viewer."""
import logging
import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
        QGridLayout, QMessageBox, QPushButton,
        QWidget,
        )

from .cam_viewport import CamViewPort
from .change_cmap import ChangeCMAPDialog
from .delete_presets import DeletePresetsDialog
from .data_loader import DataLoader
from .movie_maker import MovieMaker
from .pc_viewport import PointCloudViewPort
from .preset_details_viewport import PresetDetailsViewport
from .process_viewport import ProcessViewPort
from .rename_preset_viewport import RenamePresetViewport
from .waveforms_viewport import WaveformsViewport
from ..bases import BaseUtility
from ..utils import get_metadata


class MainWidget(QWidget, BaseUtility):
    """Main Widget for the point cloud viewer and accessories."""

    current_frame_changed = pyqtSignal(object)
    current_preset_changed = pyqtSignal(object)

    def __init__(
            self,
            app,
            root,
            loglevel=logging.INFO,
            ):
        """Window init method."""
        QWidget.__init__(self)
        BaseUtility.__init__(self, loglevel=loglevel)
        self.app = app
        self.root_dir = root

        self.frame_dirs = [
                os.path.join(self.root_dir, x)
                for x in os.listdir(self.root_dir) if x.startswith("frame")]
        frame_indices = sorted(
                [int(frame.replace("frame", ""))
                 for frame in os.listdir(self.root_dir) if frame.startswith("frame")])
        self.nframes = len(self.frame_dirs)
        self.frame_dirs = [
                os.path.join(self.root_dir, f"frame{i}")
                for i in frame_indices]
        self.data = DataLoader(self, loglevel=self._loglevel)
        self.build_gui()
        # once everything is there refresh viewports
        self.refresh_all_viewports()

    def build_gui(self):
        """Build the GUI."""
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # main features
        # total nframes
        # reorder framedirs in chronological order
        self.process_viewport = ProcessViewPort(self)
        self.pc_viewport = PointCloudViewPort(self)
        self.cam_viewport = CamViewPort(self)
        self.rename_preset_viewport = RenamePresetViewport(
                self, loglevel=self._loglevel)

        # SETUP GUI
        # pc viewer
        self.layout.addWidget(self.pc_viewport, 0, 0, 1, -1)

        # show cam viewer btn
        self.show_cam_viewport_btn = QPushButton("Cameras")
        self.show_cam_viewport_btn.clicked.connect(self.show_cam_viewport)
        self.layout.addWidget(self.show_cam_viewport_btn, 1, 1, 1, 1)

        self.waveform_btn = QPushButton("Waveforms")
        self.waveform_btn.clicked.connect(self.show_waveforms)
        self.layout.addWidget(self.waveform_btn, 1, 2, 1, 1)

        self.movie_btn = QPushButton("Make Movie")
        self.movie_btn.clicked.connect(self.create_movie)
        self.layout.addWidget(self.movie_btn, 1, 3, 1, 1)

        self.close_btn = QPushButton("Skra Pop Pop")
        self.close_btn.clicked.connect(self.close)
        self.layout.addWidget(self.close_btn, 1, 4, 1, 1)

        ##################
        # other widgets
        ##################
        self.waveforms = WaveformsViewport(self, loglevel=self._loglevel)
        self.movie_maker = None
        # connect current frame change to screenshot
        # self.current_frame_changed.connect(self.screenshot)
        # adjust widget sizes
        self.layout.setRowMinimumHeight(
                0, self.app.monitor.height() - 2 * self.close_btn.height())

    @property
    def current_frame(self):
        """Return the current frame index."""
        return self.pc_viewport.current_frame

    @property
    def current_cam_images(self):
        """Return th current camera images dictionary."""
        return self.data.cam_images

    @property
    def current_frame_dir(self):
        """Return the current frame directory."""
        return self.frame_dirs[self.current_frame]

    @property
    def current_pc(self):
        """Return the complete point cloud data for the current frame."""
        return self.data.pc

    @property
    def current_pc_colors(self):
        """Return the colors array for the current point cloud."""
        return self.data.pc_colors

    @property
    def current_preset(self):
        """Return the current preset selected."""
        return self.pc_viewport.current_preset

    def delete_presets(self):
        """Open delete presets dialog."""
        delete_presets_dialog = DeletePresetsDialog(self)
        delete_presets_dialog.exec()

    def change_cmap(self):
        """Open change cmap dialog."""
        change_cmap_dialog = ChangeCMAPDialog(self, loglevel=self._loglevel)
        change_cmap_dialog.exec()

    def close(self):
        """Close window(s)."""
        self.waveforms.close()
        if self.movie_maker is not None:
            self.movie_maker.close()
        self.cam_viewport.close()
        self.process_viewport.close()
        super().close()
        self.app.close()

    def create_movie(self):
        """Create a movie of the grabbed images."""
        self._logger.info(
                "Creating movie out of grabbed frames + lidar point cloud.")
        if self.movie_maker is None:
            self.movie_maker = MovieMaker(self, loglevel=self._loglevel)
        self.movie_maker.show()

    def go_to_beginning(self):
        """Rewind to beginning of frames."""
        self.go_to_frame(0)

    def go_to_frame(self, frame):
        """Go to a specific frame."""
        self.pc_viewport.current_frame = frame

    def open_rename_preset_viewport(self):
        """Open the rename preset window."""
        self.rename_preset_viewport.preset_name_to_change = self.current_preset
        self.rename_preset_viewport.show()

    def play(self, *args, **kwargs):
        """Play movie and call callbacks if needed after each frames."""
        self.pc_viewport.play(*args, **kwargs)

    def recompute_preset(self, *args, **kwargs):
        """Recompute the preset."""
        # ARE YOU SURE?
        confirm_dialog = QMessageBox()
        confirm_dialog.setWindowTitle("Are you sure?")
        confirm_dialog.setIcon(QMessageBox.Question)
        confirm_dialog.setText(
                "Are you sure you want to recompute preset "
                f"'{self.current_preset}'?")
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        clicked_btn = confirm_dialog.exec_()
        if clicked_btn != QMessageBox.Yes:
            return
        # get current preset details
        processing_parameters = get_metadata(
                self.frame_dirs[self.current_frame], self.current_preset)
        self.process_viewport.process(
                processing_parameters=processing_parameters,
                force_overwrite_lidar=True,
                )

    def set_ready(self):
        """Set the app ready for viewing."""
        # adjust maximum number of frames if needed
        self.pc_viewport.set_ready(
                preset=self.process_viewport.preset,
                min_frame=self.process_viewport.begin_frame,
                max_frame=self.process_viewport.end_frame,
                )
        self.cam_viewport.refresh_available_options()

    def show_cam_viewport(self):
        """Show and refresh the cam viewport."""
        self._logger.info("Showing cam viewport.")
        self.cam_viewport.show()
        self.cam_viewport.refresh_viewport()

    def show_preset_details(self):
        """Show preset details."""
        self.show_preset_details_viewport = PresetDetailsViewport(
                self, loglevel=self._loglevel)
        self.show_preset_details_viewport.exec_()

    def show_waveforms(self):
        """Show the waveforms for the current frame."""
        self._logger.info("Opening waveform window.")
        self.waveforms.show()
        self.waveforms.refresh_viewport()

    def stop(self):
        """Stop playing movie."""
        self.pc_viewport.stop()

    def refresh_all_viewports(self):
        """Refresh all viewports."""
        self.pc_viewport.refresh_viewport()

    def reset(self):
        """Reset the cam/pc viewer."""
        self.waveforms.current_pt_mesh = None
        self.pc_viewport.reset()
