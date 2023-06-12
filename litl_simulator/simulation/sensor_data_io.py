"""Sensor data IO classes."""
import abc
# import concurrent.futures
from itertools import product
# import json
import os
import queue

import carla
import numpy as np
# from pyquaternion import Quaternion

from ..bases import BaseUtility
from ..data_descriptors import KittiDescriptor, KittiDescriptorCollection
from ..settings import (
        # ANNOTATION_TAG_2_BBOX_ID,
        # ANNOTATION_TAG_2_CARLA_LABEL,
        # BBOX_ID_2_ANNOTATION_TAG,
        BBOX_KITTI_FORMAT_FILENAME,
        # BBOX_2D_EDGES_FILENAME, BBOX_2D_IDS_FILENAME,
        BBOX_VERTICES_OCCLUSIONS_FILENAME,
        BBOX_3D_VERTICES_FILENAME,
        # BBOX_3D_EDGES_FILENAME, BBOX_3D_IDS_FILENAME,
        CAM_FILENAME, CAM_MATRIX_FILENAME,
        DEPTH_CAMTYPE, OCCLUDED_FLAG,
        OK_FLAG,
        OPTICAL_FLOW_CAMTYPE,
        OUTSIDE_CAMERA_FOV_FLAG,
        PC_FILENAME, PC_MATRIX_FILENAME, TAGS_FILENAME,
        PROJECTION_MATRIX_FILENAME,
        # MIN_BBOX_AREA_IN_PX
        )
from ..utils import pixel_in_image, quaternion_from_roll_pitch_yaw


class BaseSensorDataHandler(BaseUtility):
    """Handles sensor data stream.

    Sends all data to a queue in order to get it later.
    """

    def __init__(self, actor_manager, **kwargs):
        """Base class init method."""
        super().__init__(**kwargs)
        self.queue = queue.Queue()
        self.actor_manager = actor_manager

    def __call__(self, data):
        """Call implemented as callback for sensor Actor."""
        self.put(data, block=True)

    def flush_data_queue(self):
        """Flush data from queue."""
        while True:
            try:
                self.queue.get(block=False)
            except queue.Empty:
                return

    def get(self, *args, **kwargs):
        """Get data from queue."""
        return self.queue.get(*args, **kwargs)

    def put(self, *args, **kwargs):
        """Put data into queue."""
        self.queue.put(*args, **kwargs)

    @abc.abstractmethod
    def process_data(self):
        """Process data."""
        pass


class BBoxDataHandler(BaseUtility):
    """Handles the 3D bounding boxes."""
    def __init__(self, actor_manager, **kwargs):
        """BBOX data handler init method."""
        super().__init__(**kwargs)
        self.actor_manager = actor_manager
        # edges pairs
        self.edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],
                [0, 4], [4, 5], [5, 1], [5, 7],
                [7, 6], [6, 4], [6, 2], [7, 3],
                ]
        self.visible_bboxes = {}
        self.all_bboxes = None

    def reset_bboxes(self):
        """Resets the bboxes."""
        del self.visible_bboxes
        self.visible_bboxes = {}
        del self.all_bboxes
        self.all_bboxes = None

    def get_bbox_set(self, flatten=False):
        """Return the list of bboxes of all objects."""
        if self.all_bboxes is not None:
            return self.all_bboxes
        ego_loc = self.actor_manager.ego_vehicle.get_transform().location
        ego_loc = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
        bboxes = {}
        for actor in self.actor_manager.world.get_actors():
            kitti_type = self.get_kitti_type_from_actor(actor)
            if kitti_type is None:
                continue
            bboxes.setdefault(kitti_type, [])
            actor_transform = actor.get_transform()
            actor_loc = actor_transform.location
            actor_rot = actor_transform.rotation
            actor_bbox = actor.bounding_box  # in local actor coordinates :(
            bbox_loc = actor_bbox.location
            bbox_loc_world = actor_transform.transform(bbox_loc)
            extent = actor_bbox.extent
            if kitti_type == "Car":
                loc = np.array([actor_loc.x, actor_loc.y, actor_loc.z])
                if np.allclose(loc, ego_loc, atol=1e-2, rtol=0):
                    # probably the ego vehicle (tolerance = 1cm)
                    continue
            # for pedestrians, sometimes the bbox is too small when pedestrian is moving
            # some limbs might be outside the box. To fix this, iterate through all the bone components
            # and compare with the original extent
            if kitti_type == "Pedestrian":
                # print(str([bbox_loc_world.x, bbox_loc_world.y, bbox_loc_world.z]))
                # get relative positions of all bones (ONLY in XY plane since Z is already good)
                # relative to bbox location***
                bones_relative_locs = []
                for bone in actor.get_bones().bone_transforms:
                    bones_relative_locs.append(
                            [
                                bone.world.location.x - bbox_loc_world.x,
                                bone.world.location.y - bbox_loc_world.y,
                                ],
                            )
                    # print(bone.name, str([bone.world.location.x, bone.world.location.y, bone.world.location.z]))
                # print(np.mean(bones_relative_locs, axis=0))
                #     bones_relative_locs.append(
                #             [
                #                 bone.component.location.x,
                #                 bone.component.location.y,
                #                 ]
                #             )
                bones_relative_locs = np.array(bones_relative_locs)
                # bones_relative_locs_mean = np.mean(bones_relative_locs, axis=0)
                # bbox_loc = bbox_loc + carla.Location(
                #         bones_relative_locs_mean[0], bones_relative_locs_mean[1], 0.0)
                bones_extent = np.abs(bones_relative_locs).max(axis=0)
                # compute new extent
                extent_x = max([extent.x, bones_extent[0]])
                extent_y = max([extent.y, bones_extent[1]])
                # print(bones_extent, str([extent.x, extent.y]), str([extent_x, extent_y]))
                extent = carla.Vector3D(extent_x, extent_y, extent.z)
            bbox = carla.BoundingBox(
                    bbox_loc_world,
                    # actor_transform.transform(bbox_loc),
                    # carla.Vector3D(*rotation.dot(extent_local)),
                    extent,
                    )
            bbox.rotation = actor_rot
            bboxes[kitti_type].append(bbox)
        # add parked cars
        bboxes["Car"] += list(self.actor_manager.parked_cars_bbs)
        bboxes["TrafficSign"] = list(self.actor_manager.traffic_signs_bbs)
        bboxes["TrafficLight"] = list(self.actor_manager.traffic_lights_bbs)
        self.all_bboxes = bboxes
        return bboxes

    def get_kitti_type_from_actor(self, actor):
        """Get the kitti label for a given carla actor."""
        id_ = actor.type_id
        if id_ == "spectator" or id_.startswith("sensor") or id_.startswith("controller"):
            return None
        if id_.startswith("traffic."):
            if id_.endswith("traffic_light"):
                return None
            return None
        elif id_.startswith("walker"):
            return "Pedestrian"
        elif id_.startswith("vehicle"):
            nwheels = int(actor.attributes["number_of_wheels"])
            if nwheels == 2:
                return "Cyclist"
            vtype = id_.replace("vehicle.", "")
            if vtype in [
                    "audi.a2", "audi.etron", "audi.tt",
                    "bmw.grandtourer",
                    "chevrolet.impala",
                    "citroen.c3",
                    "dodge.charger_2020", "dodge.charger_police", "dodge.charger_police_2020",
                    "ford.crown", "ford.mustang",
                    "jeep.wrangler_rubicon",
                    "lincoln.mkz_2017", "lincoln.mkz_2020",
                    "mercedes.coupe", "mercedes.coupe_2020",
                    "micro.microlino",
                    "mini.cooper_s", "mini.cooper_s_2021",
                    "nissan.micra", "nissan.patrol", "nissan.patrol_2021",
                    "seat.leon",
                    "tesla.cybertruck", "tesla.model3",
                    "toyota.prius",
                    ]:
                return "Car"
            elif vtype in [
                    "ford.ambulance",
                    "mercedes.sprinter",
                    "volkswagen.t2", "volkswagen.t2_2021",
                    ]:
                return "Van"
            elif vtype in [
                    "carlamotors.carlacola",
                    "carlamotors.firetruck",
                    ]:
                return "Truck"
        elif id_.startswith("static.prop"):
            return None
        raise NotImplementedError(f"{id_} {actor.attributes}")

    def process_3D_data(self, root, lidar):
        """Stores the 3D bboxes into root folder."""
        # start with traffic signs / lights
        os.makedirs(root, exist_ok=True)
        transf = lidar.get_transform()
        # l2w = np.array(transf.get_matrix())
        # w2l = np.array(transf.get_inverse_matrix())
        lidar_roll, lidar_pitch, lidar_yaw = (
                transf.rotation.roll, transf.rotation.pitch, transf.rotation.yaw)
        # convert visible bboxes vertices to lidar local coordinates
        kitti_descriptor_coll = KittiDescriptorCollection(loglevel=self._loglevel)
        # inverse quaternion (world to lidar frame of reference)
        quaternion = quaternion_from_roll_pitch_yaw(
                lidar_roll, lidar_pitch, lidar_yaw, degrees=True).conjugate
        for bbox_tag, bboxes_data in self.visible_bboxes.items():
            for bbox_data in bboxes_data:
                bbox = bbox_data["bbox"]
                occluded = bbox_data["occluded"]
                # in world coordinates
                roll, pitch, yaw = bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw
                local_quaternion = quaternion_from_roll_pitch_yaw(roll, pitch, yaw, degrees=True)
                tot_quaternion = quaternion * local_quaternion
                # relative to lidar
                kitti_descriptor = KittiDescriptor(loglevel=self._loglevel)
                kitti_descriptor.set_quaternion(tot_quaternion)
                kitti_descriptor.set_type(bbox_tag)
                kitti_descriptor.set_occlusion(occluded)
                # not all 0, because some evaluation code will ignore these boxes
                kitti_descriptor.set_bbox([0, 0, 99, 99])

                bloc = np.array([bbox.location.x, bbox.location.y, bbox.location.z])  # in world frame
                tloc = np.array([transf.location.x, transf.location.y, transf.location.z])  # lidar
                # inverse transform for location into lidar frame of reference
                loc = quaternion.rotate(bloc - tloc)  # bbox loc in lidar frame
                # extent in lidar frame of reference

                kitti_descriptor.set_3d_object_dimensions(bbox.extent)

                height = 2 * bbox.extent.z
                kitti_descriptor.set_3d_object_location(loc, height)
                kitti_descriptor.set_rotation_y(yaw * np.pi / 180.0)  # 0 is x-axis in kitti cam frame

                kitti_descriptor_coll.add_data_point(kitti_descriptor)
        kitti_descriptor_coll.write(os.path.join(root, BBOX_KITTI_FORMAT_FILENAME))

    def vertices_2_camera_2dpos(self, vertices, cam_matrix):
        """Convert vertices (3d world coords) to camera position."""
        verts2d = []
        for vertice in vertices:
            v2d = self.get_image_point(vertice, self.actor_manager.projection_matrix, cam_matrix)
            # need to transform to actual pixel frame of reference
            verts2d.append(np.array(v2d))
        return verts2d

    def get_projected_2d_bbox(self, vertices):
        """Return the 2 vertices which forms the projected 2d bbox."""
        vxs = [v[0] for v in vertices]
        vys = [v[1] for v in vertices]
        min_x, max_x = min(vxs), max(vxs)
        min_y, max_y = min(vys), max(vys)
        return np.round([min_x, min_y, max_x, max_y]).astype(int)

    def get_occlusion_flag(self, occlusions):
        """Return the occlusion flag."""
        # 0 = fully visible
        # 1 = partly occluded
        # 2 = largely occluded
        # 3 = unknown
        # 4 = fully occluded [CUSTOM]
        half_vertices = len(occlusions) // 2
        quarter_vertices = len(occlusions) // 4
        three_quarter_vertices = 3 * quarter_vertices
        num_visible = len([o for o in occlusions if o != OCCLUDED_FLAG])
        num_outside = len([o for o in occlusions if o == OUTSIDE_CAMERA_FOV_FLAG])
        if num_outside > three_quarter_vertices:
            return None  # fully outside cam FOV
        if num_visible <= quarter_vertices:
            return 4  # fully occluded
        # if 50% of vertices are outside FOV or occluded -> largely occluded
        if num_visible < half_vertices:
            return 2
        if num_outside >= half_vertices:
            return 2
        # if 25 % are outside FOV or occluded -> partly occluded
        if num_visible < three_quarter_vertices:
            return 1
        if num_outside >= quarter_vertices:
            return 1
        return 0

    def process_2D_data(self, root, depthcam, img):
        """Stores the 2D bboxes into root folder for cam."""
        os.makedirs(root, exist_ok=True)
        w2c = np.array(depthcam.get_transform().get_inverse_matrix())
        depth_buffer = img[:, :, 0] + 256 * img[:, :, 1] + 256 * 256 * img[:, :, 2]
        depth_buffer = depth_buffer.astype(float) * 1000 / (256 ** 3 - 1)  # results in m
        kitti_descriptor_coll = KittiDescriptorCollection(loglevel=self._loglevel)
        all_occlusions = []
        proj_3d_bboxes = []
        for bbox_tag, bboxes in self.get_bbox_set().items():
            for bbox in bboxes:
                verts = [v for v in bbox.get_world_vertices(carla.Transform())]  # in world frame of reference
                verts_2dpos = self.vertices_2_camera_2dpos(verts, w2c)
                # compute occlusion flag for each vertex
                occlusions = self.get_occlusions_for_vertices(
                        # round pixel coordinates before casting as int otherwise a floor happens
                        verts_2dpos,
                        depth_buffer,
                        )
                # only keep 2d bbox if at least 4 vertices are visible (not occluded)
                # and if at most 4 vertices are inside the camera FOV
                occluded = self.get_occlusion_flag(occlusions)
                if occluded is None:
                    # not in cam FOV at all
                    continue
                self.visible_bboxes.setdefault(bbox_tag, [])
                if bbox not in [b["bbox"] for b in self.visible_bboxes[bbox_tag]]:
                    # make sure no repetitions
                    self.visible_bboxes[bbox_tag].append({"bbox": bbox, "occluded": occluded})
                # compute 2d bbox
                proj_bbox2d_verts = self.get_projected_2d_bbox(verts_2dpos)
                # discard bbox if area too small
                # if abs((proj_bbox2d_verts[0] - proj_bbox2d_verts[2]) *
                #        (proj_bbox2d_verts[1] - proj_bbox2d_verts[3])) < MIN_BBOX_AREA_IN_PX:
                #     continue
                proj_3d_bboxes.append(np.round(np.array(verts_2dpos)[:, :2]).astype(int))
                kitti_descriptor = KittiDescriptor(loglevel=self._loglevel)
                kitti_descriptor.set_bbox(proj_bbox2d_verts)
                kitti_descriptor.set_3d_object_dimensions(bbox.extent)
                kitti_descriptor.set_type(bbox_tag)
                kitti_descriptor.set_3d_object_location(bbox.location)
                kitti_descriptor.set_occlusion(occluded)
                kitti_descriptor_coll.add_data_point(kitti_descriptor)
                all_occlusions.append(occlusions)
        # write data
        kitti_descriptor_coll.write(os.path.join(root, BBOX_KITTI_FORMAT_FILENAME))
        np.savez_compressed(
                os.path.join(root, BBOX_VERTICES_OCCLUSIONS_FILENAME), all_occlusions)
        np.savez_compressed(
                os.path.join(root, BBOX_3D_VERTICES_FILENAME), proj_3d_bboxes)
        np.savez_compressed(
                os.path.join(root, PROJECTION_MATRIX_FILENAME), w2c)
        return depth_buffer, w2c

    def get_image_point(self, loc, K, w2c):
        """Taken from carla docs."""
        pt = np.array([loc.x, loc.y, loc.z, 1])
        pt_cam = np.dot(w2c, pt)
        pt_cam = [pt_cam[1], -pt_cam[2], pt_cam[0]]
        pt_img = np.dot(K, pt_cam)
        pt_img[0] /= abs(pt_img[2])
        pt_img[1] /= abs(pt_img[2])
        return pt_img

    def get_occlusions_for_vertices(self, verts_uv, depth_buffer):
        """Compute occlusion flags for each vertex in a list of vertices."""
        occlusions = []
        for y, x, depth in verts_uv:
            x, y = np.round([x, y]).astype(int)
            if depth < 0:
                occlusions.append(OUTSIDE_CAMERA_FOV_FLAG)
                continue
            if not pixel_in_image(x, y, depth_buffer):
                occlusions.append(OUTSIDE_CAMERA_FOV_FLAG)
                continue
            if self.vertex_is_occluded(x, y, depth, depth_buffer):
                occlusions.append(OCCLUDED_FLAG)
                continue
            occlusions.append(OK_FLAG)
        return occlusions

    def vertex_is_occluded(self, x, y, depth, depth_buffer):
        """Return True if a vertex is entirely occluded."""
        # compare with 8 neighboring pixels and the pixel in itself
        neighbors = product((1, 0, -1), repeat=2)
        is_occluded = []
        for dy, dx in neighbors:
            # make sure neighboring pixel is inside img otherwise don't care
            if pixel_in_image(dx + x, dy + y, depth_buffer):
                if depth_buffer[dx + x, dy + y] < depth:
                    is_occluded.append(True)  # depth buffer is closer to camera plane than neighboring vertex
                else:
                    is_occluded.append(False)
        # pt is occluded if all adjacent pixels in depth buffer are closer
        return all(is_occluded)

    def _interpolate_outside_cam_fov_pt(self, p1, p2, w, h):
        """This methods interpolates a new pt on the edge of the camera FOV."""
        # p1 here is outside
        # assuming here p2 is not outside FOV
        p1x, p1y, p2x, p2y = p1[0], p1[1], p2[0], p2[1]
        deltay = p2y - p1y
        deltax = p2x - p1x
        if p1x >= 0 and p1x < h and p1y >= 0 and p1y < w:
            # pt is behind camera plane but inside camera FOV
            # compute pt interpolation on camera plane
            # r(t) = (p2 - p1)t + p1
            # we're looking for t' s.t.: r(t')[2] == 0
            tp = -p1[2] / (p2[2] - p1[2])
            newx = int(np.clip(round(deltax * tp + p1x), 0, h - 1))
            newy = int(np.clip(round(deltay * tp + p1y), 0, w - 1))
            return np.array([newx, newy], dtype=int)
        if deltay == 0:
            # either crosses d2 or d4
            if p1x < 0:
                return np.array([0, p1y], dtype=int)
            elif p1x >= h:
                return np.array([h - 1, p1y], dtype=int)
            else:
                raise LookupError(f"{p1} {p2} {w} {h}")
        # try crossing d1
        side = 1
        sstar = -p1y / deltay  # where r(t) cross d1
        tstar = deltax * sstar + p1x
        if tstar >= h or tstar < 0:
            # does not cross d1. then it might cross d3 at
            side = 3
            sstar = (w - 1 - p1y) / deltay
            tstar = deltax * sstar + p1x
            if tstar >= h or tstar < 0:
                # does not cross d3 either, try d4
                side = 4
                sstar = -p1x / deltax
                tstar = deltay * sstar + p1y
                if tstar >= w or tstar < 0:
                    # then it MUST cross d2
                    side = 2
                    sstar = (h - 1 - p1x) / deltax
                    tstar = deltay * sstar + p1y
                    if tstar >= w or tstar < 0:
                        raise LookupError(f"{p1} {p2} {w} {h}")
        if side == 1:
            toreturn = np.array([tstar, 0])
        elif side == 3:
            toreturn = np.array([tstar, w - 1])
        elif side == 2:
            toreturn = np.array([h - 1, tstar])
        else:
            toreturn = np.array([0, tstar])
        toreturn = np.round(toreturn).astype(int)
        toreturn[0] = np.clip(toreturn[0], 0, h - 1)
        toreturn[1] = np.clip(toreturn[1], 0, w - 1)
        return toreturn


class RGBCameraDataHandler(BaseSensorDataHandler):
    """RGB camera data handler."""

    def process_data(
            self, outdir,
            camtype,  # hack TODO: do better
            ):
        """Process data."""
        os.makedirs(outdir, exist_ok=True)
        try:
            data = self.get(block=True, timeout=None)
            if camtype == OPTICAL_FLOW_CAMTYPE:
                image = data.get_color_coded_flow().raw_data
            elif camtype == DEPTH_CAMTYPE:
                orig_raw_data = np.frombuffer(
                        data.raw_data, dtype=np.dtype("uint8")).copy()
                # can also use LogarithmicDepth conversion
                # this leads to better precision closer but less prevision farther away
                data.convert(carla.ColorConverter.Depth)
                image = data.raw_data
            else:
                image = data.raw_data
            im_array = np.frombuffer(image, dtype=np.dtype("uint8"))
            im_array = np.reshape(im_array, (data.height, data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]
            np.savez_compressed(os.path.join(outdir, CAM_FILENAME), im_array)
            # save matrix as well
            matrix = data.transform.get_inverse_matrix()
            np.savez_compressed(os.path.join(outdir, CAM_MATRIX_FILENAME), matrix)
        except queue.Empty:
            # do nothing else
            self._logger.error("Camera queue empty...")
            raise
        if camtype == DEPTH_CAMTYPE:
            # return the original data
            im_array = np.reshape(orig_raw_data, (data.height, data.width, 4))
            return im_array[:, :, :3][:, :, ::-1]


class SemanticLidarDataHandler(BaseSensorDataHandler):
    """Semantic lidar data handler."""

    def process_data(self, outdir):
        """Process data."""
        os.makedirs(outdir, exist_ok=True)
        data = self.get(block=True, timeout=None)
        lid_mat = data.transform.get_matrix()
        np.savez_compressed(os.path.join(outdir, "lidar_matrix"), lid_mat)

        try:
            # p_cloud_size = len(lid_data)
            raw_data = np.frombuffer(
                        data.raw_data,
                        dtype=np.dtype(
                            [('theta', np.float32), ('phi', np.float32), ('rho', np.float32),
                             ('CosAngle', np.float32), ('ObjIdx', np.uint32),
                             ('ObjTag', np.uint32),
                             ]))
            # float data
            p_cloud = np.asarray(
                    [raw_data["theta"],  raw_data["phi"], raw_data["rho"],
                     raw_data["CosAngle"]]).T
            p_cloud = p_cloud.reshape(
                    (self.actor_manager.channels,
                     self.actor_manager.points_per_rotation_per_laser, 4),
                    )
            # integers data
            tags = np.asarray([raw_data["ObjTag"], raw_data["ObjIdx"]]).T
            tags = tags.reshape(
                    (self.actor_manager.channels, self.actor_manager.points_per_rotation_per_laser, 2))
            np.savez_compressed(os.path.join(outdir, PC_FILENAME), p_cloud)
            np.savez_compressed(os.path.join(outdir, TAGS_FILENAME), tags)
            # save matrix as well for projection
            np.savez_compressed(os.path.join(outdir, PC_MATRIX_FILENAME),
                                data.transform.get_matrix())
            return p_cloud
        except queue.Empty:
            self._logger.error("Lidar queue empty...")
            raise
