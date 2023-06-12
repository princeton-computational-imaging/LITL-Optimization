"""Point cloud view port module."""
# import json
# import os
from itertools import product
import time
import os

import numpy as np
from pyquaternion import Quaternion
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QVector3D
from PyQt5.QtWidgets import (
        QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton, QSlider,
        QSpinBox, QWidget,
        )
import pyqtgraph.opengl as gl

from ..bases import BaseUtility
from ..settings import (
        KITTI_TYPE_2_BBOX_COLOR,
        WHITE_BACKGROUND,
        SHOW_ONLY_CAR_PED_CYC,
        SCREENSHOT_SAVE_PATH
        )


class PointCloudViewPort(QWidget, BaseUtility):
    """Viewport for point cloud.

    Contains all viewport controls.
    """
    current_frame_changed = pyqtSignal(object)

    def __init__(self, lidargui, **kwargs):
        """View port init method."""
        QWidget.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        self.mainhub = lidargui
        self._current_frame = 0
        self.avail_presets = None
        self._current_preset = None
        self._current_max_framerate = None
        self.build_gui()
        self._playing = False
        self._stop = False
        self._play_callbacks = None  # function(s) called each time a frame is played
        self.current_max_framerate = 10  # default
        self.refresh_viewport()

    def build_gui(self):
        """Builds the GUI."""
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        # Point cloud viewer (opengl)
        self.grid_dimensions = 1000
        self.pc_viewer = gl.GLViewWidget()
        self.pc_viewer.setWindowTitle("Point Cloud")
        cam_position = QVector3D(-30.752685546875, 0.32226863503456116, 19.175369262695312)
        self.pc_viewer.setCameraPosition(
                distance=1,
                pos=cam_position,
                elevation=28, azimuth=180)
        if WHITE_BACKGROUND:
            self.pc_viewer.setBackgroundColor('w')

        self.grid = gl.GLGridItem()
        if WHITE_BACKGROUND:
            self.grid.setColor('k')
            self.grid.setGLOptions('translucent')
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(5, 5)
        self.grid.translate(0, 0, -2)
        self.pc_viewer.addItem(self.grid)
        self.layout.addWidget(self.pc_viewer, 0, 0, 1, 1)

        # add presets controls
        self.build_presets_controls(1, 0, 1, 1)

        # frames controls
        self.build_frames_controls(2, 0, 1, 1)

        # annotations controls
        self.build_annotations_controls(3, 0, 1, 1)

    def build_annotations_controls(self, *args):
        """Builds the annotations controls."""
        self.annotations_controls_layout = QGridLayout()
        self.layout.addLayout(self.annotations_controls_layout, *args)
        self.show_annotations_chkbox = QCheckBox("Show annotations")
        self.show_annotations_chkbox.stateChanged.connect(self.on_show_annotations_changed)
        self.annotations_controls_layout.addWidget(self.show_annotations_chkbox, 0, 0, 1, 1)

        # show hidden (occluded)
        self.show_occluded_annotations_chkbox = QCheckBox("Show occluded bboxes")
        self.show_occluded_annotations_chkbox.stateChanged.connect(self.refresh_only_pc_viewer)
        self.annotations_controls_layout.addWidget(self.show_occluded_annotations_chkbox, 0, 1, 1, 1)
        self.show_occluded_annotations_chkbox.setEnabled(False)

        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.annotations_controls_layout.addWidget(self.screenshot_btn, 0, 2, 1, 1)

        # # use kitti dtset file instead
        # self.use_kitti_fmt_chkbox = QCheckBox("Kitti fmt")
        # self.use_kitti_fmt_chkbox.stateChanged.connect(self.refresh_only_pc_viewer)
        # self.annotations_controls_layout.addWidget(self.use_kitti_fmt_chkbox, 0, 1, 1, 1)

    def build_frames_controls(self, *args):
        """Builds the frames controls layout."""
        self.frames_controls_layout = QGridLayout()
        self.layout.addLayout(self.frames_controls_layout, *args)
        self.to_beginning_btn = QPushButton("<<")
        self.to_beginning_btn.clicked.connect(self.go_to_beginning)
        self.frames_controls_layout.addWidget(self.to_beginning_btn, 0, 0, 1, 1)

        self.previous_btn = QPushButton("<")
        self.previous_btn.clicked.connect(self.previous)
        self.frames_controls_layout.addWidget(self.previous_btn, 0, 1, 1, 1)

        self.next_btn = QPushButton(">")
        self.next_btn.clicked.connect(self.next)
        self.frames_controls_layout.addWidget(self.next_btn, 0, 2, 1, 1)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        self.frames_controls_layout.addWidget(self.play_btn, 0, 3, 1, 1)
        # pt size controls
        self.pt_size_label = QLabel("Pt size:")
        self.frames_controls_layout.addWidget(self.pt_size_label, 0, 4, 1, 1)
        self.pt_size_control = QSpinBox()
        self.pt_size_control.setMinimum(1)
        self.pt_size_control.setValue(3)
        self.pt_size_control.valueChanged.connect(self.refresh_only_pc_viewer)
        self.frames_controls_layout.addWidget(self.pt_size_control, 0, 5, 1, 1)

        # movie slider to grab current frame
        self.movie_time_label = QLabel()
        self.movie_time_label.setText(f"1/{self.nframes}")
        self.frames_controls_layout.addWidget(self.movie_time_label, 1, 0, 1, 1)
        self.movie_slider = QSlider(Qt.Horizontal)
        self.movie_slider.setMinimum(0)
        self.movie_slider.setMaximum(self.nframes - 1)
        self.movie_slider.sliderReleased.connect(self.change_current_frame_from_slider)
        self.movie_slider.valueChanged.connect(self.change_current_frame_from_slider)
        self.frames_controls_layout.addWidget(self.movie_slider, 1, 1, 1, 2)
        self.loop_chkbox = QCheckBox("Loop")
        self.frames_controls_layout.addWidget(self.loop_chkbox, 1, 3, 1, 1)

        # max framerate controls
        self.decrement_max_framerate_btn = QPushButton("-")
        self.decrement_max_framerate_btn.clicked.connect(self.decrement_max_framerate)
        self.frames_controls_layout.addWidget(self.decrement_max_framerate_btn, 1, 4, 1, 1)
        self.max_framerate_label = QLabel()  # f"Max FPS = {self.current_max_framerate}")
        self.frames_controls_layout.addWidget(self.max_framerate_label, 1, 5, 1, 1)
        self.increment_max_framerate_btn = QPushButton("+")
        self.frames_controls_layout.addWidget(self.increment_max_framerate_btn, 1, 6, 1, 1)
        self.increment_max_framerate_btn.clicked.connect(self.increment_max_framerate)

    def build_presets_controls(self, *args):
        """Builds the presets controls layout."""
        self.preset_controls_layout = QGridLayout()
        self.layout.addLayout(self.preset_controls_layout, *args)
        # preset selection
        self.preset_selection_label = QLabel("Presets:")
        self.preset_controls_layout.addWidget(self.preset_selection_label, 0, 0, 1, 1)
        # small inner grid layout here for controls
        self.preset_selection_grid_layout = QGridLayout()
        self.preset_controls_layout.addLayout(self.preset_selection_grid_layout, 0, 1, 1, 2)

        self.previous_preset_btn = QPushButton("<")
        self.previous_preset_btn.clicked.connect(self.previous_preset)
        self.preset_selection_grid_layout.addWidget(self.previous_preset_btn, 0, 0, 1, 1)
        self.preset_selection_menu = QComboBox()
        self.preset_selection_menu.textActivated.connect(self.change_current_preset)
        self.preset_selection_grid_layout.addWidget(self.preset_selection_menu, 0, 1, 1, 4)
        self.next_preset_btn = QPushButton(">")
        self.next_preset_btn.clicked.connect(self.next_preset)
        self.preset_selection_grid_layout.addWidget(self.next_preset_btn, 0, 5, 1, 1)

        self.add_preset_btn = QPushButton("Add/Edit")
        self.preset_controls_layout.addWidget(self.add_preset_btn, 0, 3, 1, 1)
        self.add_preset_btn.clicked.connect(self.mainhub.process_viewport.show)

        self.rename_preset_btn = QPushButton("Rename")
        self.preset_controls_layout.addWidget(self.rename_preset_btn, 0, 4, 1, 1)
        self.rename_preset_btn.clicked.connect(self.mainhub.open_rename_preset_viewport)

        self.delete_presets_btn = QPushButton("Delete")
        self.delete_presets_btn.clicked.connect(self.mainhub.delete_presets)
        self.preset_controls_layout.addWidget(self.delete_presets_btn, 0, 5, 1, 1)

        self.show_preset_details_btn = QPushButton("Details")
        self.show_preset_details_btn.clicked.connect(self.mainhub.show_preset_details)
        self.preset_controls_layout.addWidget(self.show_preset_details_btn, 0, 6, 1, 1)

        self.recompute_preset_btn = QPushButton("Recompute")
        self.recompute_preset_btn.clicked.connect(self.mainhub.recompute_preset)
        self.preset_controls_layout.addWidget(self.recompute_preset_btn, 0, 7, 1, 1)

        self.change_cmap_btn = QPushButton("Change cmap")
        self.change_cmap_btn.clicked.connect(self.mainhub.change_cmap)
        self.preset_controls_layout.addWidget(self.change_cmap_btn, 0, 8, 1, 1)

    @property
    def current_frame(self):
        """The current frame number."""
        return self._current_frame

    @current_frame.setter
    def current_frame(self, frame):
        if frame < 0 or frame >= self.nframes:
            raise ValueError(
                    f"frame should be 0 < frame < {self.nframes}")
        if frame == self.current_frame:
            # nothing to do
            return
        self._current_frame = frame
        # propagate signal
        self.mainhub.current_frame_changed.emit(self.current_frame)
        self.refresh_viewport()
        # update the movie label
        self.movie_time_label.setText(f"{frame + 1}/{self.nframes}")
        if not self.movie_slider.isSliderDown():
            self.movie_slider.setSliderPosition(frame)

    @property
    def current_max_framerate(self):
        """The current maximum framerate."""
        return self._current_max_framerate

    @current_max_framerate.setter
    def current_max_framerate(self, fps):
        self._current_max_framerate = fps
        self.max_framerate_label.setText(f"Max FPS = {fps}")

    @property
    def current_pc(self):
        """The current point cloud."""
        return self.mainhub.current_pc

    @property
    def current_pc_colors(self):
        """The current point cloud color array."""
        return self.mainhub.data.pc_colors

    @property
    def current_preset(self):
        """Current preset selection."""
        return self._current_preset

    @current_preset.setter
    def current_preset(self, preset):
        if self.avail_presets is None:
            self.avail_presets = self.mainhub.data.get_available_presets(self.current_frame)
        if preset not in self.avail_presets and preset:
            raise ValueError(preset)
        if preset is None:
            # set to first one
            preset = self.avail_presets[0]
        self._current_preset = preset
        self.mainhub.current_preset_changed.emit(self.current_preset)
        self.refresh_only_pc_viewer()

    @property
    def frame_dirs(self):
        """The list of frame directories."""
        return self.mainhub.frame_dirs

    @property
    def point_size(self):
        """The size of points in the point cloud."""
        return self.pt_size_control.value()

    @property
    def loop(self):
        """Return True if the loop chkbox is checked."""
        return self.loop_chkbox.isChecked()

    @property
    def nframes(self):
        """The number of frames total."""
        return self.mainhub.nframes

    @property
    def show_annotations(self):
        """Return true if we want to show annotations."""
        return self.show_annotations_chkbox.isChecked()

    def add_annotations(self):
        """Adds annotations to the point cloud view."""
        kitti_coll = self.mainhub.data.lidar_bboxes
        vertex_graph = {
                0: [1, 2, 4],
                1: [3, 5],
                2: [3, 6],
                3: [7],
                4: [5, 6],
                5: [7],
                6: [7],
                }
        colors_present = {}
        for idx, kitti_data in enumerate(kitti_coll):
            if kitti_data.type not in ['Car', 'Van', 'Truck', 'Cyclist', 'Pedestrian'] and SHOW_ONLY_CAR_PED_CYC:
                continue

            if kitti_data.occluded == 4 and not self.show_occluded_annotations_chkbox.isChecked():
                continue
            # loc = np.array(kitti_data.location.split(" "), dtype=float).copy()
            loc = kitti_data.location.copy()
            loc[1] *= -1  # mirror to right handed system
            extent = kitti_data.extent
            # copy quaternion here because changing elements stays fixed afterwards
            quaternion = Quaternion(
                    kitti_data.quaternion.w, kitti_data.quaternion.x,
                    kitti_data.quaternion.y, kitti_data.quaternion.z)
            quaternion.elements[3] *= -1  # since we mirror one axis
            # get vertez array
            vertices = []
            color = KITTI_TYPE_2_BBOX_COLOR[kitti_data.type]
            colors_present[kitti_data.type] = color
            for s1, s2, s3 in product((1, -1), repeat=3):
                delta = np.array([s1 * extent[0], s2 * extent[1], s3 * extent[2]])
                vertex = loc + quaternion.rotate(delta)
                # vertex = quaternion.rotate(vertex)
                vertices.append(vertex)
            vertices = np.array(vertices)
            # draw edges
            for init_vertex_idx, end_vertex_indices in vertex_graph.items():
                begin_vertex = vertices[init_vertex_idx]
                for end_vertex_idx in end_vertex_indices:
                    end_vertex = vertices[end_vertex_idx]
                    line = gl.GLLinePlotItem(
                            pos=np.array([begin_vertex, end_vertex]),
                            color=color,
                            width=2,
                            )
                    if WHITE_BACKGROUND:
                        line.setGLOptions('translucent')
                    self.pc_viewer.addItem(line)
        self.add_annotations_legend(colors_present)

    def add_annotations_legend(self, colors):
        """Add annotations legend."""
        # TODO: maybe we'll have to use vispy for this?
        pass

    def change_current_preset(self, preset):
        """Change the current preset."""
        if not preset:
            return
        self.current_preset = preset

    def change_current_frame_from_slider(self):
        """Change the current frame from the slider."""
        if self.movie_slider.isSliderDown():
            # don't change if slider still pressed
            return
        self.current_frame = self.movie_slider.value()

    def decrement_max_framerate(self):
        """Decrement maximum framerate by 1."""
        if self.current_max_framerate == 1:
            return
        self.current_max_framerate -= 1

    def get_framedir(self, frame):
        """Return the framedir for the given frame number."""
        return self.mainhub.frame_dirs[frame]

    def go_to_beginning(self):
        """Go to 1st frame and show it."""
        self.current_frame = 0

    def increment_max_framerate(self):
        """Increment maximal framerate by 1."""
        self.current_max_framerate += 1

    def next(self):
        """Show the next lidar point cloud + camera frame."""
        if self.current_frame >= self.nframes - 1:
            if not self.loop:
                return
            self.go_to_beginning()
            return
        self.current_frame += 1

    def next_preset(self):
        """Move to next preset down the list."""
        if len(self.avail_presets) == 1:
            return
        idx = (self.avail_presets.index(self.current_preset) + 1)
        if idx == len(self.avail_presets):
            idx = 0  # wrap around
        new_preset = self.avail_presets[idx]
        self.current_preset = new_preset
        self.preset_selection_menu.setCurrentText(new_preset)

    def on_show_annotations_changed(self, state):
        """Method called when the show annotations chkbox has changed."""
        self.refresh_only_pc_viewer()
        if state == Qt.Checked:
            enable = True
        else:
            enable = False
        self.show_occluded_annotations_chkbox.setEnabled(enable)

    def play(self, *args, **kwargs):
        """Play the lidar point cloud video up to the end from the current frame."""
        if self._playing:
            # already playing => stop
            self.stop()
            return
        self.play_btn.setText("Stop")
        self._do_play(**kwargs)

    def previous(self):
        """Show the previous lidar point cloud + camera frame."""
        if self.current_frame <= 0:
            return
        self.current_frame -= 1

    def previous_preset(self):
        """Move to previous preset down the list."""
        if len(self.avail_presets) == 1:
            return
        idx = (self.avail_presets.index(self.current_preset) - 1) % len(self.avail_presets)
        new_preset = self.avail_presets[idx]
        self.current_preset = new_preset
        self.preset_selection_menu.setCurrentText(new_preset)

    # TODO: fix this function when the available presets changes, there is a problem...
    def refresh_viewport(self, **kwargs):
        """Load and view point cloud data for given frame path."""
        # refresh presets available
        avail_presets = self.mainhub.data.get_available_presets(self.current_frame)
        if not avail_presets:
            # ask lidar gui to process data
            self.mainhub.process_viewport.ready = False
            self.reset()
            self.mainhub.process_viewport.show()
            return
        if self.current_preset is None:
            self._current_preset = avail_presets[0]
            self.refresh_viewport(**kwargs)
            return
        # only change if it is different
        if avail_presets != self.avail_presets:
            self.avail_presets = avail_presets
            # both the following calls will recall the refresh function
            # (since we change the current text)
            self.preset_selection_menu.clear()
            self.preset_selection_menu.addItems(avail_presets)
            # try to keep same preset if it exists
            if self.current_preset in self.avail_presets:
                self.preset_selection_menu.setCurrentText(self.current_preset)
                return
            self.current_preset = self.avail_presets[0]
            return
        if self.current_preset is None or self.current_preset not in self.avail_presets:
            # this should avoid infinite recursion
            self._current_preset = self.avail_presets[0]
        self.refresh_only_pc_viewer(**kwargs)

    def refresh_only_pc_viewer(self, force=False):
        """Only refreshes the pc viewer part of the viewport."""
        self.reset()
        # now display point cloud
        if force:
            # force reloading data
            self.mainhub.data.delete_lidar_cached_data()
        pc = self.current_pc[..., :3]
        colors = self.current_pc_colors
        if self.current_pc.ndim > 2:
            # concatenate
            colors = np.concatenate(colors, axis=0)
            pc = np.concatenate(pc, axis=0)
        mesh = gl.GLScatterPlotItem(
                pos=pc,
                size=self.point_size,
                color=colors)
        mesh.setGLOptions('translucent')
        self.pc_viewer.addItem(mesh)
        if self.show_annotations:
            self.add_annotations()

    def take_screenshot(self):
        """Take a screenshot.

        WARNING: the path is hardcoded so this might make the app crash!
        """
        assert os.path.exists(SCREENSHOT_SAVE_PATH), f'{SCREENSHOT_SAVE_PATH} does not exist.'
        screenshot = self.pc_viewer.readQImage()
        save_path = f'{SCREENSHOT_SAVE_PATH}/pc_{self.current_preset}.png'
        screenshot.save(save_path, 'png')
        print(f'Saved point cloud screenshot in {save_path}')

    def reset(self):
        """Resets viewport."""
        self.pc_viewer.items = []
        self.pc_viewer.addItem(self.grid)

    def set_ready(self, preset=None, min_frame=None, max_frame=None):
        """Sets the viewer ready."""
        if self.current_frame is None:
            # this will not refresh
            self._current_frame = 0
        if self.current_preset is None and preset is not None:
            # this will not refresh
            self._current_preset = preset
        # set movie slider accordingly
        if min_frame is None:
            min_frame = 0
        if max_frame is None:
            max_frame = self.nframes - 1
        if self.current_frame >= max_frame:
            self._current_frame = max_frame
        if self.current_frame < min_frame:
            self._current_frame = min_frame
        self.refresh_viewport(force=True)

    def stop(self):
        """Stop video from playing."""
        self._stop = True
        self.play_btn.setText("Play")

    def _do_play(self, callback=None, callback_args=None, callback_kwargs=None, loop=None):
        """Actually do the playing.

        Calls callback if needed.
        """
        if loop is None:
            loop = self.loop
        if self._stop or (self.current_frame >= self.nframes - 1 and not loop):
            self.stop()
            self._play_callbacks = None
            self._stop = False
            self._playing = False
            return
        if self._play_callbacks is None:
            self._play_callbacks = []
        if callback is not None:
            if not callable(callback):
                raise ValueError(
                        f"callback should be callable but it is '{callback}'.")
            if callback_args is None:
                callback_args = []
            if callback_kwargs is None:
                callback_kwargs = {}
            self._play_callbacks.append(
                    {"callback": callback, "args": callback_args, "kwargs": callback_kwargs}
                    )
        self._playing = True
        fps = 1 / self.current_max_framerate  # in seconds
        start = time.time()
        self.next()
        end = time.time()
        remaining = (fps + start - end) * 1000  # in ms
        if remaining < 0:
            remaining = 0
        else:
            remaining = int(round(remaining, 0))
        QTimer.singleShot(remaining, self._do_play)
        # call the callbacks
        for tocall in self._play_callbacks:
            tocall["callback"](*tocall["args"], **tocall["kwargs"])
