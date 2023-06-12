"""Cam viewer module."""
import numpy as np
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
        QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton, QSlider, QWidget,
        )
import qimage2ndarray

from .error_dialog_wrapper import error_dialog
from .main_app import kill_app
from ..bases import BaseUtility
from ..process_pipelines.pc_projector import PointCloudProjector
from ..settings import (
        KITTI_TYPE_2_BBOX_COLOR,
        SCREENSHOT_SAVE_PATH,
        SHOW_ONLY_CAR_PED_CYC,
        )
from ..utils import pixel_in_image


PROJECTION_GREEN = "Green"
PROJECTION_INTENSITY = "Intensity"
PROJECTION_COLORS = [
        PROJECTION_GREEN,
        PROJECTION_INTENSITY,  # "RGB", "Diffuse",
        ]

SAVE = False  # TODO make button


class CamViewPort(QWidget, BaseUtility):
    """The Cam Viewer widget."""

    def __init__(self, lidargui, **kwargs):
        """Cam viewer init method."""
        QWidget.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        self.mainhub = lidargui
        self.mainhub.current_frame_changed.connect(self.on_current_frame_changed)
        self.mainhub.current_preset_changed.connect(self.refresh_pixmap)
        self.build_gui()

        self.avail_yaws, self.avail_camtypes = [], []
        self.avail_projection_colors = PROJECTION_COLORS
        self.current_yaw = None
        self.current_camtype = None
        self.current_projection_color = PROJECTION_COLORS[0]
        self.current_projection_color_idx = 0
        self.current_yaw_idx = 0
        self.current_camtype_idx = 0
        self.refresh_viewport()

    def build_annotations_controls(self, *args):
        """Builds the annotations controls."""
        self.annotations_controls_layout = QGridLayout()
        self.layout.addLayout(self.annotations_controls_layout, *args)
        self.show_annotations_chkbox = QCheckBox("Show annotations")
        self.annotations_controls_layout.addWidget(self.show_annotations_chkbox, 0, 0, 1, 1)
        # unchecked by default
        self.show_annotations_chkbox.stateChanged.connect(
                self.on_show_annotations_state_changed)

        self.show_hidden_annotations_chkbox = QCheckBox("Show hidden annotations")
        self.annotations_controls_layout.addWidget(
                self.show_hidden_annotations_chkbox, 0, 1, 1, 1)
        self.show_hidden_annotations_chkbox.stateChanged.connect(
                self.refresh_pixmap)
        # 3D bboxes / 2D bboxes
        self.bboxes_3D_chkbox = QCheckBox("3D bboxes")
        self.annotations_controls_layout.addWidget(self.bboxes_3D_chkbox, 0, 2, 1, 1)
        self.bboxes_3D_chkbox.stateChanged.connect(self.refresh_pixmap)
        # since the show chkbox is unchecked by default, disable these
        self.show_hidden_annotations_chkbox.setEnabled(False)
        self.bboxes_3D_chkbox.setEnabled(False)

    def build_gui(self):
        """Builds the GUI."""
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Camera Viewer")

        self.cam_viewport = QLabel()
        # spans all columns
        self.layout.addWidget(self.cam_viewport, 0, 0, 1, -1)

        # projection controls
        self.build_projection_controls(1, 0, 1, 1)

        # BBoxes controls
        self.build_annotations_controls(1, 1, 1, 1)

        # image selection controls
        self.build_image_selection_controls(2, 0, 1, 1)

        # IQ options
        self.build_image_iq_controls(2, 1, 1, 1)

        # close btn
        self.build_window_controls(3, 0, 1, -1)

    def build_image_iq_controls(self, *args):
        """Build the image iq controls."""
        self.image_iq_controls_layout = QGridLayout()
        self.layout.addLayout(self.image_iq_controls_layout, *args)
        # image filter options
        self.image_filter_label = QLabel("Image Filter: ")
        self.image_iq_controls_layout.addWidget(self.image_filter_label, 0, 0, 1, 1)
        self.image_filter_dropdown_menu = QComboBox()
        self.image_filter_dropdown_menu.addItems(["None", "R", "G", "B"])
        self.image_filter_dropdown_menu.currentTextChanged.connect(self.refresh_pixmap)
        self.image_iq_controls_layout.addWidget(self.image_filter_dropdown_menu, 0, 1, 1, 1)
        # gamma correction
        self.gamma_correction_label = QLabel("Gamma:")
        self.image_iq_controls_layout.addWidget(self.gamma_correction_label, 1, 0, 1, 1)
        self.decrease_gamma_btn = QPushButton("-")
        self.decrease_gamma_btn.clicked.connect(self.decrease_gamma)
        self.image_iq_controls_layout.addWidget(self.decrease_gamma_btn, 1, 1, 1, 1)

        self.gamma_correction_slider = QSlider(Qt.Horizontal)
        self.gamma_lut = np.linspace(0, 2, 21)[::-1]  # make sure 1 is in it
        self.gamma_correction_for_camtype = {}
        self.gamma_correction_slider.setMinimum(0)
        self.gamma_correction_slider.setMaximum(len(self.gamma_lut) - 1)
        self.reset_gamma()
        self.gamma_correction_slider.valueChanged.connect(self.refresh_pixmap)
        self.image_iq_controls_layout.addWidget(self.gamma_correction_slider, 1, 2, 1, 1)

        self.increase_gamma_btn = QPushButton("+")
        self.increase_gamma_btn.clicked.connect(self.increase_gamma)
        self.image_iq_controls_layout.addWidget(self.increase_gamma_btn, 1, 3, 1, 1)
        self.gamma_stick_chkbox = QCheckBox("stick")
        self.image_iq_controls_layout.addWidget(self.gamma_stick_chkbox, 1, 4, 1, 1)

    def build_image_selection_controls(self, *args):
        """Build the image selection controls."""
        self.image_selection_controls_layout = QGridLayout()
        self.layout.addLayout(self.image_selection_controls_layout, *args)
        self.previous_projection_color_btn = QPushButton("<")
        self.previous_projection_color_btn.clicked.connect(self.previous_projection_color)
        self.image_selection_controls_layout.addWidget(self.previous_projection_color_btn, 0, 0, 1, 1)
        self.projection_color_label = QLabel("Projection color")
        self.image_selection_controls_layout.addWidget(self.projection_color_label, 0, 1, 1, 1)
        self.next_projection_color_btn = QPushButton(">")
        self.next_projection_color_btn.clicked.connect(self.next_projection_color)
        self.image_selection_controls_layout.addWidget(self.next_projection_color_btn, 0, 2, 1, 1)

        # push buttons to alter camera view
        self.previous_camtype_btn = QPushButton("<")
        self.previous_camtype_btn.clicked.connect(self.previous_camtype)
        self.image_selection_controls_layout.addWidget(self.previous_camtype_btn, 1, 0, 1, 1)
        self.camtype_label = QLabel("Camera type")
        self.image_selection_controls_layout.addWidget(self.camtype_label, 1, 1, 1, 1)
        self.next_camtype_btn = QPushButton(">")
        self.next_camtype_btn.clicked.connect(self.next_camtype)
        self.image_selection_controls_layout.addWidget(self.next_camtype_btn, 1, 2, 1, 1)

        self.previous_yaw_btn = QPushButton("<")
        self.previous_yaw_btn.clicked.connect(self.previous_yaw)
        self.image_selection_controls_layout.addWidget(self.previous_yaw_btn, 2, 0, 1, 1)
        self.yaw_label = QLabel("Yaw angle")
        self.image_selection_controls_layout.addWidget(self.yaw_label, 2, 1, 1, 1)
        self.next_yaw_btn = QPushButton(">")
        self.next_yaw_btn.clicked.connect(self.next_yaw)
        self.image_selection_controls_layout.addWidget(self.next_yaw_btn, 2, 2, 1, 1)

    def build_projection_controls(self, *args):
        """Builds the projection controls section."""
        self.projection_controls_layout = QGridLayout()
        self.layout.addLayout(self.projection_controls_layout, *args)
        self.show_point_cloud_chkbox = QCheckBox("Project point cloud.")
        self.show_point_cloud_chkbox.setChecked(False)
        self.show_point_cloud_chkbox.stateChanged.connect(self.refresh_pixmap)
        self.projection_controls_layout.addWidget(self.show_point_cloud_chkbox, 1, 0, 1, 1)

    def build_window_controls(self, *args):
        """Build window controls layout."""
        self.window_controls_layout = QGridLayout()
        self.layout.addLayout(self.window_controls_layout, *args)
        self.close_btn = QPushButton("Skra pop pop")
        self.close_btn.clicked.connect(self.close)
        self.window_controls_layout.addWidget(self.close_btn, 0, 0, 1, 1)

    @property
    def all_labels_and_pixmaps(self):
        """The list of tuples of all (pixmap labels, pixmaps)."""
        if self.cam_labels is None:
            return None
        pixmaps = []
        for cam_label_row in self.cam_labels:
            for cam_label in cam_label_row:
                if cam_label.pixmap() is not None:
                    pixmaps.append((cam_label, cam_label.pixmap()))
        return pixmaps

    @property
    def current_frame(self):
        """The current frame index."""
        return self.mainhub.current_frame

    @property
    def current_pc(self):
        """Current point cloud array."""
        return self.mainhub.current_pc

    @property
    def current_pc_colors(self):
        """The current point cloud color array."""
        return self.mainhub.current_pc_colors

    @property
    def gamma(self):
        """The gamma correction factor."""
        return self.gamma_lut[self.gamma_correction_slider.value()]

    @gamma.setter
    def gamma(self, ga):
        if ga not in self.gamma_lut:
            raise ValueError("Gamma not in LUT.")
        self.gamma_correction_slider.setValue(np.where(self.gamma_lut == ga)[0][0])

    @property
    def gamma_stick(self):
        """Return True if we keep gamma settings accross frames / preset / yaw."""
        return self.gamma_stick_chkbox.isChecked()

    @property
    def show_3D_bboxes(self):
        """Return True if we want to show 3D annotations instead of 2D."""
        return self.bboxes_3D_chkbox.isChecked()

    @property
    def show_annotations(self):
        """True if we want to show annotations."""
        return self.show_annotations_chkbox.isChecked()

    @property
    def show_hidden_annotations(self):
        """True if we want to show the hidden edges."""
        return self.show_hidden_annotations_chkbox.isChecked()

    @property
    def show_point_cloud_projection(self):
        """True if we show pc projection on image."""
        return self.show_point_cloud_chkbox.isChecked()

    @show_point_cloud_projection.setter
    def show_point_cloud_projection(self, show):
        self.show_point_cloud_chkbox.setChecked(show)

    def add_annotations(self, image):
        """Add annotated bboxes if we want."""
        if not self.show_annotations:
            return image
        if self.show_3D_bboxes:
            vertex_graph = {
                    0: [1, 2, 4],
                    1: [3, 5],
                    2: [3, 6],
                    3: [7],
                    4: [5, 6],
                    5: [7],
                    6: [7],
                    }
        else:
            vertex_graph = {0: [1, 2], 3: [1, 2]}
        annotations = self.mainhub.data.get_cam_bboxes(self.current_yaw)
        for idata, kitti_data in enumerate(annotations["kitti"]):
            if kitti_data.type not in ['Car', 'Van', 'Truck', 'Cyclist', 'Pedestrian'] and SHOW_ONLY_CAR_PED_CYC:
                continue
            if kitti_data.occluded == 4 and not self.show_hidden_annotations:
                continue
            if self.show_3D_bboxes:
                corners_coordinates = annotations["vertices"][idata]
                # list of vertice coordintes [n_vertex x 2]
            else:
                vertices = kitti_data.bbox  # list of 4 elements [x1, y1, x2, y2]
                corners_coordinates = np.array([
                    [vertices[0], vertices[1]],  # bottom left
                    [vertices[0], vertices[3]],  # top left
                    [vertices[2], vertices[1]],  # bottom right
                    [vertices[2], vertices[3]],  # top right
                    ])
            color = KITTI_TYPE_2_BBOX_COLOR[kitti_data.type]
            color = np.round(np.array(color) * 255).astype(np.uint8)
            # draw edges from vertices
            for begin_vertex_id, connected_vertices_id in vertex_graph.items():
                begin_vertex = corners_coordinates[begin_vertex_id]
                for connected_vertex_id in connected_vertices_id:
                    end_vertex = corners_coordinates[connected_vertex_id]
                    # draw line
                    image = self.add_line(begin_vertex, end_vertex, color, image)
        return image

    def add_line(self, begin_px, end_px, color, image):
        """Draws a line on the cam viewport."""
        line_points = self.get_line_points(begin_px, end_px)
        for px, py in line_points:
            if pixel_in_image(px, py, image):
                image[px, py] = color
        return image

    def get_line_points(self, begin_px, end_px):
        """Return the list of pixel points for a line."""
        # taken from
        # https://github.com/Ozzyz/carla-data-export/blob/26c0bec203a2f3d370ff8373ca6371b7eef35300/camera_utils.py#L91
        x1, y1, x2, y2 = begin_px[1], begin_px[0], end_px[1], end_px[0]
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points

    def apply_gamma_correction(self, image):
        """Apply gamma correction to image array."""
        if self.gamma == 1:
            return image
        # save gamma in case sticky
        if self.current_camtype not in self.gamma_correction_for_camtype:
            self.gamma_correction_for_camtype[self.current_camtype] = self.gamma
        self.gamma_correction_for_camtype[self.current_camtype] = self.gamma
        # image are np.uint8
        return np.clip(np.round(np.power(image / 255, self.gamma) * 255, 0), 0, 255).astype(np.uint8)

    def apply_image_filter(self, imarray):
        """Apply an image filter."""
        filt = self.image_filter_dropdown_menu.currentText()
        if filt == "None":
            return imarray
        # mask out channels as necessary
        tokeep = ["R", "G", "B"].index(filt)
        for channel in range(3):
            if channel == tokeep:
                continue
            imarray[:, :, channel] = 0
        return imarray

    def close(self):
        """Close the window."""
        # also uncheck the point cloud projection
        self.show_point_cloud_projection = False
        super().close()

    def decrease_gamma(self):
        """Decrement gamma slider by 1."""
        current = self.gamma_correction_slider.value()
        if current == 0:
            return
        self.gamma_correction_slider.setValue(current - 1)

    @error_dialog(
            text="An error occured while displaying PC projection.",
            title="CamViewport Error",
            callback=kill_app)
    def display_point_cloud_projection(self, imarray):
        """Display point cloud projection on image."""
        # image here is a QImage
        # project current PC first we need to load lidar + cam matrices
        lidar_matrix = self.mainhub.data.lidar_matrix
        camera_matrix = self.mainhub.data.get_cam_matrix(self.current_camtype, self.current_yaw)
        projector = PointCloudProjector(
                self.current_pc, lidar_matrix, camera_matrix,
                imarray.shape[1], imarray.shape[0])
        if self.current_projection_color == PROJECTION_GREEN:
            colors = np.tile(
                    np.array([0, 255, 0, 123], dtype=np.uint8),
                    list(self.current_pc.shape[:-1]) + [1])
        elif self.current_projection_color == PROJECTION_INTENSITY:
            colors = self.current_pc_colors
        else:
            raise NotImplementedError(self.current_projection_color)
        return projector.compute_image_projection(
                image=imarray,
                colors=colors,
                radius=0,
                )

    def increase_gamma(self):
        """Increment gamma slider by 1."""
        current = self.gamma_correction_slider.value()
        if current == self.gamma_correction_slider.maximum():
            return
        self.gamma_correction_slider.setValue(current + 1)

    def next_camtype(self):
        """Move to next camtype."""
        self._change_camtype(1)

    def next_projection_color(self):
        """Move to next projection color."""
        self._change_projection_color(1)

    def next_yaw(self):
        """Move to next yaw."""
        # inverse here in order for arrows switch towards where the heading
        self._change_yaw(-1)

    def on_current_frame_changed(self):
        """Method called when current frame is changed."""
        if not self.gamma_stick:
            self.reset_gamma()
        self.refresh_viewport()

    def on_show_annotations_state_changed(self, state):
        """Method called when the show annotations check box is changed."""
        if state == Qt.Checked:
            enable = True
        else:
            enable = False
        self.refresh_pixmap()
        # enable/disable other controls
        self.show_hidden_annotations_chkbox.setEnabled(enable)
        self.bboxes_3D_chkbox.setEnabled(enable)

    def previous_camtype(self):
        """Move to previous camtype."""
        self._change_camtype(-1)

    def previous_projection_color(self):
        """Move to previous projection color."""
        self._change_projection_color(-1)

    def previous_yaw(self):
        """Move to previous yaw."""
        self._change_yaw(1)

    def refresh_viewport(self):
        """Load and view the cameras for given frame."""
        if self.current_frame is None:
            return
        self.refresh_available_options()
        self.refresh_pixmap()

    def refresh_available_options(self):
        """Refreshes the controls / available options for this frame."""
        self.avail_camtypes = self.mainhub.data.get_available_camtypes(self.current_frame)
        if not self.avail_camtypes:
            # clear pixmap if needed
            if self.cam_viewport.pixmap() is not None:
                self.cam_viewport.clear()
            return
        if self.current_camtype not in self.avail_camtypes:
            # reset
            self.current_camtype = self.avail_camtypes[0]
            self.current_camtype_idx = 0
        else:
            self.current_camtype_idx = self.avail_camtypes.index(self.current_camtype)
        self.avail_yaws = self.mainhub.data.get_available_yaws(self.current_frame, self.current_camtype)
        if not self.avail_yaws:
            # clear pixmap if needed
            if self.cam_viewport.pixmap() is not None:
                self.cam_viewport.clear()
            return
        if self.current_yaw not in self.avail_yaws:
            self.current_yaw = self.avail_yaws[0]
            self.current_yaw_idx = 0
        else:
            self.current_yaw_idx = self.avail_yaws.index(self.current_yaw)

    @error_dialog(
            text="An error occured while refreshing pixmap.",
            title="CamViewport Error",
            callback=kill_app)
    def refresh_pixmap(self, *args):
        """Refresh pixmap and labels for this currentframe/camtype/yaw."""
        # adjust labels
        self.camtype_label.setText(f"Camera type: {self.current_camtype}")
        if self.current_yaw is None:
            return
        self.yaw_label.setText(f"Yaw angle: {self.current_yaw.strip('yaw')}")
        if not self.isVisible():
            # don't load data if window is not visible! (big performance gain).
            return
        # load image
        try:
            imarray = self.mainhub.data.get_cam_array(
                self.current_camtype, self.current_yaw).copy()
        except FileNotFoundError:
            # at startup this might happen if no data processed yet
            return
        # apply gamma corrections first
        imarray = self.apply_gamma_correction(imarray)
        if self.show_point_cloud_projection:
            imarray = self.display_point_cloud_projection(imarray)
            self.projection_color_label.setText(
                f"Projection color: {self.current_projection_color}")
            # enable buttons
            self.next_projection_color_btn.setEnabled(True)
            self.previous_projection_color_btn.setEnabled(True)
        else:
            # disable buttons
            self.projection_color_label.setText(
                    "Projection color: disabled")
            self.next_projection_color_btn.setEnabled(False)
            self.previous_projection_color_btn.setEnabled(False)
            # self.next_projection_color_btn.setUpdatesEnabled(False)
        # finally add annotated bboxes
        imarray = self.add_annotations(imarray)
        imarray = self.apply_image_filter(imarray)
        image = qimage2ndarray.array2qimage(imarray)
        pixmap = QPixmap.fromImage(image)  # QPixmap(fp).scaledToWidth(self.cam_viewport.width())
        self.cam_viewport.setPixmap(pixmap)

        if SAVE:
            assert os.path.exists(SCREENSHOT_SAVE_PATH), f'{SCREENSHOT_SAVE_PATH} does not exist.'
            save_path = os.path.join(SCREENSHOT_SAVE_PATH, f'{self.current_camtype}_{self.current_yaw}.png')
            pixmap.save(save_path)
            print(f'Saved {save_path}')

    def reset_gamma(self):
        """Resets the gamma slider."""
        self.gamma = 1

    def _change_camtype(self, delta):
        ncamtypes = len(self.avail_camtypes)
        self.current_camtype_idx = (self.current_camtype_idx + delta) % ncamtypes
        self.current_camtype = self.avail_camtypes[self.current_camtype_idx]
        # check gamma in case it is sticky
        if self.gamma_stick:
            if self.current_camtype in self.gamma_correction_for_camtype:
                self.gamma = self.gamma_correction_for_camtype[self.current_camtype]
            else:
                self.reset_gamma()
        else:
            self.reset_gamma()
        self.refresh_pixmap()

    def _change_projection_color(self, delta):
        if not self.show_point_cloud_projection:
            return
        nprojection_colors = len(self.avail_projection_colors)
        self.current_projection_color_idx = (
                (self.current_projection_color_idx + delta) % nprojection_colors
                )
        self.current_projection_color = self.avail_projection_colors[
                self.current_projection_color_idx]
        self.refresh_pixmap()

    def _change_yaw(self, delta):
        nyaws = len(self.avail_yaws)
        # do inverse here as yaw (in this left handed system) turns clockwise
        self.current_yaw_idx = (self.current_yaw_idx - delta) % nyaws
        self.current_yaw = self.avail_yaws[self.current_yaw_idx]
        self.refresh_pixmap()
