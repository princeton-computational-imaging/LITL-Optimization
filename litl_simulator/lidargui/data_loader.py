"""Data loading / caching module."""
import os

import numpy as np

from .error_dialog_wrapper import error_dialog
from ..bases import BaseUtility
from ..data_descriptors import KittiDescriptorCollection
from ..settings import (
        ANNOTATIONS_FOLDER,
        BBOX_YAW_PATH,
        # BBOX_2D_EDGES_FILENAME,
        # BBOX_2D_IDS_FILENAME,
        # BBOX_3D_EDGES_FILENAME,
        # BBOX_3D_IDS_FILENAME,
        BBOX_KITTI_FORMAT_FILENAME,
        BBOX_VERTICES_OCCLUSIONS_FILENAME,
        BBOX_3D_VERTICES_FILENAME,
        METADATA_FILENAME,
        PRESET_SUBFOLDER_BASENAME,
        PROCESSED_COLORS_FILENAME,
        PROCESSED_DATA_FOLDER,
        PROCESSED_PC_FILENAME,
        )
from ..utils import (
        get_available_camtypes,
        get_available_yaws,
        get_camera_array,
        get_camera_array_from_qimage,
        get_camera_image,
        get_camera_matrix,
        get_lidar_matrix,
        sanitize_preset_name,
        )


class DataLoader(BaseUtility):
    """Utility for loading / caching data used by different viewports."""

    def __init__(self, mainhub, **kwargs):
        """Data loader init method."""
        super().__init__(**kwargs)
        self.mainhub = mainhub
        self._pc = None
        self._pc_colors = None
        self._cam_images = {}
        self._cam_arrays = {}
        self._cam_matrices = {}
        self._lidar_matrix = None
        self._cam_bboxes = {}
        self._lidar_bboxes = None
        # if frame / preset changes -> delete loaded data to reload if asked
        self.mainhub.current_frame_changed.connect(self.delete_all_cached_data)
        self.mainhub.current_preset_changed.connect(self.delete_lidar_cached_data)

    @property
    def current_frame(self):
        """The current frame."""
        return self.mainhub.current_frame

    @property
    def current_preset(self):
        """The current preset."""
        return self.mainhub.current_preset

    @property
    def cam_arrays(self):
        """The camera numpy arrays."""
        raise RuntimeError("Cannot access cam arrays directly use get().")

    @cam_arrays.deleter
    def cam_arrays(self):
        del self._cam_arrays
        self._cam_arrays = {}

    @property
    def cam_bboxes(self):
        """The cam bboxes data."""
        raise RuntimeError("Cannot access cam bboxes directly use get().")

    @cam_bboxes.deleter
    def cam_bboxes(self):
        del self._cam_bboxes
        self._cam_bboxes = {}

    @property
    def cam_images(self):
        """The camera QImages."""
        raise RuntimeError("Cannot access cam images directly use get().")

    @cam_images.deleter
    def cam_images(self):
        del self._cam_images
        self._cam_images = {}

    @property
    def cam_matrices(self):
        """The camera matrices."""
        raise RuntimeError("Cannot access cam matrices directly use get().")

    @cam_matrices.deleter
    def cam_matrices(self):
        del self._cam_matrices
        self._cam_matrices = {}

    @property
    def pc(self):
        """The point cloud."""
        if self._pc is None:
            # attemps at loading the pc
            self.load_pc()
        return self._pc

    @pc.setter
    def pc(self, data):
        self._pc = data

    @pc.deleter
    def pc(self):
        del self._pc
        self._pc = None

    @property
    def pc_colors(self):
        """The point cloud colors."""
        if self._pc_colors is None:
            self.load_pc_colors()
        return self._pc_colors

    @pc_colors.setter
    def pc_colors(self, colors):
        self._pc_colors = colors

    @pc_colors.deleter
    def pc_colors(self):
        del self._pc_colors
        self._pc_colors = None

    @property
    def lidar_matrix(self):
        """The lidar matrix."""
        if self._lidar_matrix is None:
            self._lidar_matrix = self.load_lidar_matrix(self.current_frame)
        return self._lidar_matrix

    @lidar_matrix.setter
    def lidar_matrix(self, matrix):
        self._lidar_matrix = matrix

    @lidar_matrix.deleter
    def lidar_matrix(self):
        del self._lidar_matrix
        self._lidar_matrix = None

    @property
    def lidar_bboxes(self):
        """The lidar bboxes."""
        if self._lidar_bboxes is None:
            self._lidar_bboxes = self.load_lidar_bboxes(self.current_frame)
        return self._lidar_bboxes

    @lidar_bboxes.deleter
    def lidar_bboxes(self):
        del self._lidar_bboxes
        self._lidar_bboxes = None

    def get_available_camtypes(self, frame):
        """Retrieve available cam types for current frame."""
        framedir = os.path.join(self.mainhub.frame_dirs[frame])
        return get_available_camtypes(framedir)

    def get_available_presets(self, frame):
        """Return the list of available presets for this frame."""
        presets = []
        framedir = os.path.join(self.mainhub.frame_dirs[frame], PROCESSED_DATA_FOLDER)
        if not os.path.isdir(framedir):
            return presets
        for filename in os.listdir(framedir):
            if not filename.startswith(PRESET_SUBFOLDER_BASENAME):
                continue
            # make sure processing was successfull
            metadatapath = os.path.join(framedir, filename, METADATA_FILENAME)
            if not os.path.isfile(metadatapath):
                # not successfull
                continue
            presets.append(sanitize_preset_name(filename, reverse=True))
        return sorted(presets)

    def get_available_yaws(self, frame, camtype):
        """Retrieve available cam types for current frame / camtype."""
        framedir = self.mainhub.frame_dirs[frame]
        return get_available_yaws(framedir, camtype)

    def get_cam_array(self, camtype, yaw, **kwargs):
        """Get the camera QImage for given camtype and yaw."""
        return self._get_cam_data(camtype, yaw, self.load_cam_array, self._cam_arrays, **kwargs)

    def get_cam_bboxes(self, yaw, **kwargs):
        """Get the cam bboxes data."""
        return self._get_cam_data(
                yaw=yaw, load_func=self.load_cam_bboxes,
                data_dict=self._cam_bboxes, **kwargs)

    def get_cam_image(self, camtype, yaw, **kwargs):
        """Get the camera QImage for given camtype and yaw."""
        return self._get_cam_data(camtype, yaw, self.load_cam_image, self._cam_images, **kwargs)

    def get_cam_matrix(self, camtype, yaw, **kwargs):
        """Get the camera matrix for given camtype and yaw."""
        return self._get_cam_data(camtype, yaw, self.load_cam_matrix, self._cam_matrices, **kwargs)

    def delete_all_cached_data(self):
        """Delete all cached data."""
        self.delete_lidar_cached_data()
        self.delete_camera_cached_data()

    def delete_camera_cached_data(self):
        """Deletes camera related data."""
        del self.cam_images
        del self.cam_matrices
        del self.cam_arrays
        del self.cam_bboxes

    def delete_lidar_cached_data(self):
        """Deletes lidar related data."""
        del self.pc
        del self.pc_colors
        del self.lidar_matrix
        del self.lidar_bboxes

    def load_all_data(self):
        """(Re)loads all data."""
        self.load_pc()
        self.load_cam_images()

    def load_cam_bboxes(self, yaw, frame=None):
        """Load cam bboxes."""
        if frame is None:
            frame = self.current_frame
        datadir = os.path.join(
                self.mainhub.frame_dirs[frame], ANNOTATIONS_FOLDER,
                BBOX_YAW_PATH, yaw)
        # bboxes_path = os.path.join(datadir, BBOX_2D_EDGES_FILENAME)
        kitti_data_path = os.path.join(datadir, BBOX_KITTI_FORMAT_FILENAME)
        vertices_3D_path = os.path.join(datadir, BBOX_3D_VERTICES_FILENAME)
        occlusions_path = os.path.join(datadir, BBOX_VERTICES_OCCLUSIONS_FILENAME)
        # ids_path = os.path.join(datadir, BBOX_2D_IDS_FILENAME)
        # allow loading pickles as non-uniform-shaped array can happen
        # (e.g.: when some edges are out of camera or behind for instance)
        kitti_coll = KittiDescriptorCollection.from_file(kitti_data_path, loglevel=self._loglevel)
        vertices_3D = np.load(vertices_3D_path)["arr_0"]
        occlusions = np.load(occlusions_path)["arr_0"]
        return {"kitti": kitti_coll, "vertices": vertices_3D, "occlusions": occlusions}
        # try:
        #     bboxes = np.load(bboxes_path, allow_pickle=True)["arr_0"]
        # except ValueError:
        #     self._logger.error(f"An error occured when loading edges: '{bboxes_path}'.")
        #     raise
        # try:
        #     ids = np.load(ids_path, allow_pickle=True)["arr_0"]
        # except ValueError:
        #     self._logger.error(f"An error occured when loading ids: '{ids_path}'.")
        #     raise
        # return {"bboxes": bboxes, "ids": ids}

    def load_cam_image(self, camtype, yaw, frame=None):
        """Load camera image for given yaw and camtype."""
        if frame is None:
            frame = self.current_frame
        framedir = self.mainhub.frame_dirs[frame]
        return get_camera_image(framedir, camtype, yaw)

    def load_cam_matrix(self, camtype, yaw, frame=None):
        """Loads camera matrix."""
        if frame is None:
            frame = self.current_frame
        framedir = self.mainhub.frame_dirs[frame]
        return get_camera_matrix(framedir, camtype, yaw)

    def load_cam_array(self, camtype, yaw, new_img=False, **kwargs):
        """Load camera array."""
        if not new_img:
            # load the one already in memory
            img = self.get_cam_image(camtype, yaw, **kwargs)
            return get_camera_array_from_qimage(img)
        else:
            # load a new one
            framedir = self.mainhub.frame_dirs[self.current_frame]
            return get_camera_array(framedir, camtype, yaw)

    def load_lidar_bboxes(self, frame=None):
        """Loads the lidar bboxes."""
        if frame is None:
            frame = self.current_frame
        framedir = os.path.join(
                self.mainhub.frame_dirs[frame], ANNOTATIONS_FOLDER)
        kitti_path = os.path.join(framedir, BBOX_KITTI_FORMAT_FILENAME)
        kitti_coll = KittiDescriptorCollection.from_file(kitti_path, loglevel=self._loglevel)
        return kitti_coll
        # bbox_path = os.path.join(framedir, BBOX_3D_EDGES_FILENAME)
        # ids_path = os.path.join(framedir, BBOX_3D_IDS_FILENAME)
        # try:
        #     bboxes = np.load(bbox_path, allow_pickle=True)["arr_0"]
        # except ValueError:
        #     self._logger.error(f"An error occured when loading edges: '{bbox_path}'.")
        #     raise
        # try:
        #     ids = np.load(ids_path, allow_pickle=True)["arr_0"]
        # except ValueError:
        #     self._logger.error(f"An error occured when loading ids: '{ids_path}'.")
        #     raise
        # return {"bboxes": bboxes, "ids": ids}

    def load_lidar_matrix(self, frame=None):
        """Loads the lidar matrix."""
        if frame is None:
            frame = self.current_frame
        return get_lidar_matrix(self.mainhub.frame_dirs[frame])

    @error_dialog(
            text="Error when loading point cloud from disk.", title="DataLoader error")
    def load_pc(self, *args):
        """(Re)load pc."""
        self.pc = self._load_pc_data(PROCESSED_PC_FILENAME)

    def load_pc_colors(self):
        """Loads the pc colors."""
        self.pc_colors = self._load_pc_data(PROCESSED_COLORS_FILENAME)

    def _load_pc_data(self, filename):
        framedir = os.path.join(self.mainhub.frame_dirs[self.current_frame], PROCESSED_DATA_FOLDER)
        if not self.current_preset:
            raise ValueError("No preset set.")
        path = os.path.join(
                framedir, sanitize_preset_name(self.current_preset),
                filename)
        # load point cloud for this frame
        if not os.path.isfile(path):
            self._logger.error(f"No point cloud data to load: {path}")
            raise FileNotFoundError(path)
        return np.load(path)

    def _get_cam_data(self, camtype=None, yaw=None, load_func=None, data_dict=None, **kwargs):
        """Generic function to load or get data already loaded from a data dict."""
        if load_func is None:
            raise ValueError("Need to specify 'load_func'.")
        if not callable(load_func):
            raise TypeError("load_func not callable...")
        if data_dict is None:
            raise ValueError("Need to specify 'data_dict'.")
        if camtype is not None:
            if camtype not in data_dict:
                data_dict[camtype] = {}
            if yaw not in data_dict[camtype]:
                data_dict[camtype][yaw] = load_func(camtype, yaw, **kwargs)
            return data_dict[camtype][yaw]
        else:
            if yaw not in data_dict:
                data_dict[yaw] = load_func(yaw, **kwargs)
            return data_dict[yaw]
