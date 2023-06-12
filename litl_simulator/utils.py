"""Utils module."""
import json
import os

import carla
from PyQt5.QtGui import QImage
import numpy as np
from pyquaternion import Quaternion
import qimage2ndarray

from .exceptions import DirectoryNotFoundError
from .settings import (
        CAM_MATRIX_FILENAME,
        METADATA_FILENAME,
        PC_MATRIX_FILENAME,
        PRESET_SUBFOLDER_BASENAME,
        PROCESSED_CAM_FILENAME,
        PROCESSED_DATA_FOLDER,
        RAW_DATA_FOLDER,
        YAW_BASENAME,
        )


def carla2spherical(pts):
    """Translate from carla spherical system of coordinates to the usual one."""
    # assume theta is 1st column, phis is 2nd and rho is 3rd
    # I don't know why but all angles are in degrees / 100 ??!?
    # to convert in rads => x pi / 1.8
    # compute good spherical angles theta = pi/2 - vert angle
    pts[..., 0] = np.pi / 2 - pts[..., 0] * np.pi / 1.8
    pts[..., 1] *= np.pi / 1.8
    # column 1 are horiz angle
    pts[..., 1] += np.pi  # start angles at 0 (they span - Horiz FOV / 2 to + Horiz FOV / 2)
    # for some reason we need to mirror y axis
    # actually Unreal uses left-handed coordinate system. See example here:
    # https://github.com/carla-simulator/carla/blob/fe3cb6863a604f5c0cf8b692fe2b6300b45b5999/PythonAPI/examples/open3d_lidar.py#L99
    where_less_pi = pts[..., 1] <= np.pi
    where_more_pi = np.logical_not(where_less_pi)
    pts[where_less_pi, 1] = np.pi - pts[where_less_pi, 1]
    pts[where_more_pi, 1] = 3 * np.pi - pts[where_more_pi, 1]
    return pts


# GETTER FUNCTIONS
def get_available_camtypes(framedir):
    """For a given framedir, get the available camtypes."""
    types = []
    if not os.path.isdir(framedir):
        return types
    dirname = os.path.join(framedir, PROCESSED_DATA_FOLDER)
    if not os.path.isdir(dirname):
        return types
    for subname in os.listdir(dirname):
        # all subdirectories in the raw data folder are camera folders.
        fp = os.path.join(dirname, subname)
        if os.path.isfile(fp):
            continue
        if subname.startswith(PRESET_SUBFOLDER_BASENAME):
            continue
        types.append(subname)
    return sorted(types)


def get_available_yaws(framedir, camtype):
    """Return the list of available yaws for a given framedir and camtype."""
    yaws = []
    camtypedir = os.path.join(framedir, PROCESSED_DATA_FOLDER, camtype)
    if not os.path.exists(camtypedir):
        raise DirectoryNotFoundError(camtypedir)
    for subname in os.listdir(camtypedir):
        fp = os.path.join(camtypedir, subname)
        if os.path.isfile(fp):
            continue
        yaws.append(subname)
    # sort them now according to angle
    yaws_angles = [int(yaw.replace(YAW_BASENAME, "", 1)) for yaw in yaws]
    return np.array(yaws)[np.argsort(yaws_angles)].tolist()


def get_camera_array(framedir, camtype, yaw):
    """Return the camera numpy array for a given framedir, camtype and yaw."""
    qimage = get_camera_image(framedir, camtype, yaw)
    return get_camera_array_from_qimage(qimage)


def get_camera_array_from_qimage(qimage):
    """Return numpy ndarray of image from a QImage object."""
    return qimage2ndarray.byte_view(qimage).copy()[..., [2, 1, 0, 3]]  # swap BGRA->RGBA


def get_camera_image(framedir, camtype, yaw):
    """Return the camera QImage object for a given framedir/camtype/yaw."""
    fp = os.path.join(
            framedir, PROCESSED_DATA_FOLDER,
            camtype, yaw,
            PROCESSED_CAM_FILENAME)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    return QImage(fp)


def get_camera_matrix(framedir, camtype, yaw):
    """Return the camera matrix for a given framedir, camtype and yaw."""
    fp = os.path.join(
            framedir, RAW_DATA_FOLDER, camtype, yaw,
            CAM_MATRIX_FILENAME)
    return np.load(fp)["arr_0"]


def get_lidar_matrix(framedir):
    """For a given framedir, get the lidar matrix."""
    return np.load(os.path.join(framedir, RAW_DATA_FOLDER, PC_MATRIX_FILENAME))["arr_0"]


def get_metadata(framedir, preset):
    """Returns the metadata for given framedir and preset."""
    path = os.path.join(
            framedir, PROCESSED_DATA_FOLDER, sanitize_preset_name(preset),
            METADATA_FILENAME,
            )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


def pixel_in_image(x, y, image):
    """Returns if a given pixel coordinate is inside the image or not."""
    w, h = image.shape[0], image.shape[1]
    if x < 0:
        return False
    if y < 0:
        return False
    if x >= w:
        return False
    if y >= h:
        return False
    return True


def quaternion_from_roll_pitch_yaw(roll, pitch, yaw, degrees=False):
    """Return a Quaternion from roll, pitch and yaw rotation angles."""
    if degrees:
        qx = Quaternion(axis=[1.0, 0.0, 0.0], degrees=roll)
        qy = Quaternion(axis=[0.0, 1.0, 0.0], degrees=pitch)
        qz = Quaternion(axis=[0.0, 0.0, 1.0], degrees=yaw)
    else:
        qx = Quaternion(axis=[1.0, 0.0, 0.0], radians=roll)
        qy = Quaternion(axis=[0.0, 1.0, 0.0], radians=pitch)
        qz = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
    return qx * qy * qz


def is_list_like(obj):
    """Return True if object looks like a list."""
    if isinstance(obj, list):
        return True
    if isinstance(obj, tuple):
        return True
    if isinstance(obj, carla.ActorList):
        return True
    return False


def sanitize_preset_name(name, reverse=False):
    """Sanitize a preset subfolder name.

    Args:
        name: str
            The subfolder name to sanitize.
        reverse: bool, optional
            If True: we go from actual_name -> display name
            If False: we go from display_name -> actual name
    """
    if reverse:
        name = name.replace(PRESET_SUBFOLDER_BASENAME, "", 1)
        while name.startswith("_"):
            name = name[1:]
        return name
    if not name.startswith(PRESET_SUBFOLDER_BASENAME):
        name = PRESET_SUBFOLDER_BASENAME + "_" + name
    name = name.replace(" ", "_")
    return name


def spherical2cartesian(rhos, thetas, phis):
    """Convert spherical coordinates to cartesian."""
    rhosintheta = np.multiply(rhos, np.sin(thetas))
    X = np.multiply(rhosintheta, np.cos(phis))
    Y = np.multiply(rhosintheta, np.sin(phis))
    Z = np.multiply(rhos, np.cos(thetas))
    # we want first axis at the end
    return np.moveaxis(np.asarray([X, Y, Z]), 0, -1)
