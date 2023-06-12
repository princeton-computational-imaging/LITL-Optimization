"""Data descriptors.

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in lidar coordinates (in meters)
   [REMOVED]  1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   4    quaternion values qw, qx, qy, qz [for bbox in lidar coordinate system]
                     real part = qw.
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""
import os
from typing import List

import carla
from numpy import pi
import numpy as np
from pyquaternion import Quaternion

from .bases import BaseUtility
from .settings import KITTI_TYPES
from .utils import quaternion_from_roll_pitch_yaw


# taken from:
# https://github.com/Ozzyz/carla-data-export/blob/26c0bec203a2f3d370ff8373ca6371b7eef35300/datadescriptor.py#L26
class KittiDescriptor(BaseUtility):
    """Kitti data descriptor for 3D bboxes."""

    def __init__(
            self, type_=None, bbox=None, dimensions=None,
            location=None, extent=None, pitch=None, roll=None, yaw=None,
            **kwargs,
            ):
        """Kitti descriptor's init method."""
        super().__init__(**kwargs)
        self.type = type_
        self.truncated = 0
        self.occluded = 0
        self.alpha = -10
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.score = 1.0
        self.rotation_y = -1.0
        self.extent = extent
        self.pitch, self.roll, self.yaw = pitch, roll, yaw
        self.quaternion = None
        if pitch is not None and roll is not None and yaw is not None:
            self.build_quaternion()

    def build_quaternion(self):
        """Builds the quaternion from rotation angles."""
        # in UE, the rotation order seems to be XYZ Roll, pitch then yaw. See:
        # https://forums.unrealengine.com/t/what-is-the-rotation-order-for-components-or-bones/286910
        if self.roll is None:
            raise ValueError("Need to set roll.")
        if self.pitch is None:
            raise ValueError("Need to set pitch.")
        if self.yaw is None:
            raise ValueError("Need to set yaw.")
        # assume angles are in radians and in correct coordinate system
        self.set_quaternion(
                quaternion_from_roll_pitch_yaw(
                    self.roll, self.pitch, self.yaw))

    def set_quaternion(self, quaternion):
        """Sets the quaternion."""
        if not isinstance(quaternion, Quaternion):
            raise TypeError("Not a quaternion.")
        self.quaternion = quaternion

    def set_type(self, obj_type: str):
        """Set type of object."""
        assert obj_type in KITTI_TYPES, "Object must be of types {} but got {}".format(
            KITTI_TYPES, obj_type)
        self.type = obj_type

    def set_truncated(self, truncated: float):
        """Set truncation value."""
        assert 0 <= truncated <= 1, """Truncated must be Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries """
        self.truncated = truncated

    def set_occlusion(self, occlusion: int):
        """Set occlusion flag."""
        assert occlusion in range(0, 5), """Occlusion must be Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown, 4 = fully occluded [CUSTOM]"""
        self.occluded = occlusion

    def set_alpha(self, alpha: float):
        """Set alpha value."""
        assert -pi <= alpha <= pi, "Alpha must be in range [-pi..pi]"
        self.alpha = alpha

    def set_bbox(self, bbox: List[int]):
        """Set bbox coordinates (2D)."""
        assert len(bbox) == 4, """ Bbox must be 2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates (two points)"""
        self.bbox = bbox

    def set_3d_object_dimensions(self, bbox_extent):
        """Set 3D object dimensions."""
        # Bbox extent consists of x,y and z.
        # The bbox extent is by Carla set as
        # x: length of vehicle (driving direction)
        # y: to the right of the vehicle
        # z: up (direction of car roof)
        # However, Kitti expects height, width and length (z, y, x):
        if isinstance(bbox_extent, carla.Vector3D):
            height, width, length = bbox_extent.z, bbox_extent.x, bbox_extent.y
        else:
            # assume they are already the box size
            height, width, length = bbox_extent[0] / 2, bbox_extent[1] / 2, bbox_extent[2] / 2
        # Since Carla gives us bbox extent, which is a half-box, multiply all by two
        self.extent = np.array((width, length, height))
        self.dimensions = 2 * np.array((height, width, length))

    def set_3d_object_location(self, obj_location, height=0):
        """Set 3D object location.

        TODO: Change this to
            Converts the 3D object location from CARLA coordinates and saves them as KITTI coordinates in the object
            In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
            z
            ▲   ▲ x
            |  /
            | /
            |/____> y
            This is a left-handed coordinate system, with x being forward, y to the right and z up
            See also https://github.com/carla-simulator/carla/issues/498
            However, the camera coordinate system for KITTI is defined as
                ▲ z
               /
              /
             /____> x
            |
            |
            |
            ▼
            y
            This is a right-handed coordinate system with z being forward, x to the right and y down
            Therefore, we have to make the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI:-X  -Y   Z
        """
        # Object location is four values (x, y, z, w). We only care about three of them (xyz)
        if isinstance(obj_location, carla.Location):
            obj_location = [obj_location.x, obj_location.y, obj_location.z]
        x, y, z = [float(x) for x in obj_location][0:3]

        z -= height / 2  # Kitti bbox at bottom not center

        # assert None not in [
        #     self.extent, self.type], "Extent and type must be set before location!"
        # if self.type == "Pedestrian":
        #     # Since the midpoint/location of the pedestrian is in the middle of the agent,
        #     # while for car it is at the bottom
        #     # we need to subtract the bbox extent in the height direction when adding location of pedestrian.
        #     y -= self.extent[0]
        # self.location = " ".join(map(str, [x, y, z]))
        # self.location = np.array([x, y, z])
        # Convert from Carla coordinate system to KITTI
        # This works for AVOD (image)
        # x *= -1
        # y *= -1
        # self.location = " ".join(map(str, [y, -z, x]))
        # self.location = " ".join(map(str, [-x, -y, z]))
        # This works for SECOND (lidar)
        # self.location = " ".join(map(str, [z, x, y]))
        # self.location = " ".join(map(str, [z, x, -y]))

        self.location = np.array([y, -z, x])

    def set_rotation_y(self, rotation_y: float):
        """Set y rotation."""
        # assert - \
        # pi <= rotation_y <= pi, "Rotation y must be in range [-pi..pi] - found {}".format(
        # rotation_y)
        self.rotation_y = rotation_y

    def __str__(self):
        """Returns the kitti formatted string of the datapoint.

        If it is valid only (all critical variables filled out), else it returns an error.
        """
        if self.bbox is None:
            bbox_format = " "
        else:
            bbox_format = " ".join([str(x) for x in self.bbox])
        if self.quaternion is None:
            quaternion_format = " "
        else:
            quaternion_format = " ".join([str(self.quaternion.elements[i]) for i in range(4)])
        dimensions = " ".join((str(x) for x in self.dimensions))
        location = " ".join((str(x) for x in self.location))
        return "{} {} {} {} {} {} {} {} {} {}".format(
                self.type, self.truncated, self.occluded, self.alpha, bbox_format,
                dimensions,
                location,
                self.rotation_y,
                1.0,  # score
                quaternion_format)


class KittiDescriptorCollection(BaseUtility):
    """Kitti descriptor data collection."""

    def __init__(self, **kwargs):
        """Kitti descriptor collection init method."""
        super().__init__(**kwargs)
        self.kitti_descriptors = []

    def __iter__(self):
        """Iterates over kitti descriptors."""
        for descriptor in self.kitti_descriptors:
            yield descriptor

    def add_data_point(self, descriptor):
        """Adds a data point descriptor."""
        if not isinstance(descriptor, KittiDescriptor):
            raise TypeError("Not a KittiDescriptor.")
        self.kitti_descriptors.append(descriptor)

    def read(self, path):
        """Read data descriptors from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.kitti_descriptors = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            elements = line.strip("\n").strip(" ").split(" ")
            kitti_descriptor = KittiDescriptor(loglevel=self._loglevel)
            kitti_descriptor.set_type(elements[0])
            kitti_descriptor.set_truncated(int(elements[1]))
            kitti_descriptor.set_occlusion(int(elements[2]))
            alpha = float(elements[3])
            if alpha == -10:
                # unset value
                kitti_descriptor.alpha = alpha
            else:
                kitti_descriptor.set_alpha(alpha)
            kitti_descriptor.set_bbox([int(x) for x in elements[4:8]])
            kitti_descriptor.set_3d_object_dimensions([float(x) for x in elements[8:11]])

            kitti_xyz = np.array(elements[11:14], dtype=np.float32)
            height, _, _ = kitti_descriptor.dimensions
            kitti_xyz[1] -= height / 2
            carla_xyz = np.asarray([kitti_xyz[2], kitti_xyz[0], -kitti_xyz[1]])
            kitti_descriptor.location = carla_xyz
            # kitti_descriptor.set_rotation_y(float(elements[14]))
            # rot -> 14
            # score _> 15
            if len(elements) > 15:
                # quaternions were written
                kitti_descriptor.quaternion = Quaternion(elements[16:20])
            self.add_data_point(kitti_descriptor)

    def write(self, path):
        """Write the data descriptor to a file."""
        if os.path.isfile(path):
            # overwrite
            os.remove(path)
        with open(path, "w") as f:
            f.write("\n".join([str(kd) for kd in self.kitti_descriptors]))

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        """Create a KittiDescriptorCollection object from a file path."""
        kitti_coll = cls(*args, **kwargs)
        kitti_coll.read(path)
        return kitti_coll
