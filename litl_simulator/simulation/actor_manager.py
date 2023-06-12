"""Actor Manager module."""
import os
import random
import shutil

import carla
import numpy as np

from ..bases import BaseUtility
from .sensor_data_io import (
        BBoxDataHandler, RGBCameraDataHandler, SemanticLidarDataHandler,
        )
from ..settings import (
        ANNOTATIONS_FOLDER,
        BBOX_YAW_PATH,
        DEPTH_CAMTYPE,
        DIFFUSE_COLOR_CAMTYPE,
        OPTICAL_FLOW_CAMTYPE,
        RAW_DATA_FOLDER,
        RGB_CAMTYPE,
        SPECULAR_METALLIC_ROUGHNESS_CAMTYPE,
        YAW_BASENAME,
        )
from ..utils import is_list_like


REGULAR_VEHICLE_MODELS = [
        'mercedes', 'dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius',
        'nissan', 'crown', 'impala',
        ]


class ActorManager(BaseUtility):
    """Manages actors for the simulation (except spectator)."""
    camera_yaws = (0, 90, 180, 270)
    camera_types = (
            SPECULAR_METALLIC_ROUGHNESS_CAMTYPE,
            DIFFUSE_COLOR_CAMTYPE,
            RGB_CAMTYPE,
            DEPTH_CAMTYPE,
            "semantic_segmentation",
            "instance_segmentation",
            OPTICAL_FLOW_CAMTYPE,
            )

    def __init__(
            self, world,
            **kwargs,
            ):
        """Actor manager init method.

        Args:
            world: Carla World object
            loglevel: int
                The logging level.
        """
        super().__init__(**kwargs)
        self.world = world
        self.map = self.world.get_map()

        self._set_blueprints()

        self.data_handlers = {
            "lidar": SemanticLidarDataHandler(self, loglevel=self._loglevel),
            "Bboxes": BBoxDataHandler(self, loglevel=self._loglevel),
            }
        #     "camera": queue.Queue(),
        #     "lidar_matrix": queue.Queue(),
        #     "camera_matrix": queue.Queue(),
        #     }
        for camtype in self.camera_types:
            self.data_handlers[camtype] = {}
            for yaw in self.camera_yaws:
                self.data_handlers[camtype][yaw] = RGBCameraDataHandler(
                        self, loglevel=self._loglevel)
        # destroy all actors if any
        self._logger.info("Destroying all already existing actors.")
        self.destroy_actors(self.get_initial_actors_to_destroy_from_world())
        # prepare specific attributes for simulation
        # pedestrians
        self._pedestrians, self._pedestrians_ai = [], []
        # vehicles
        self._vehicles = []
        self.ego_vehicle = None
        self.ego_vehicle_travel_distance = None
        self.ego_vehicle_last_location = None
        # world static bboxes
        self.parked_cars_bbs = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        self.traffic_lights_bbs = self.world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        self.traffic_signs_bbs = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        # sensors
        self._lidar = None
        self._lidar_z = None
        self._cameras = None
        self.projection_matrix = None
        self.lidar_range = None

    @property
    def lidar(self):
        """The Ego vehicle lidar."""
        return self._lidar

    @lidar.setter
    def lidar(self, lidar):
        self._lidar = lidar

    @lidar.deleter
    def lidar(self):
        if self.lidar is None:
            return
        self.destroy_actors(self.lidar)
        self.lidar = None
        del self.data_handlers["lidar"]
        # del self.data_queues["lidar_matrix"]
        self.data_handlers.update({"lidar": SemanticLidarDataHandler()})
        # , "lidar_matrix": queue.Queue()})

    @property
    def cameras(self):
        """The ego vehicle cameras."""
        return self._cameras

    @cameras.setter
    def cameras(self, cameras):
        self._cameras = cameras

    @cameras.deleter
    def cameras(self):
        if self.cameras is None:
            return
        for camtype, cameras in self.cameras.items():
            for yaw, camera in cameras.items():
                self.destroy_actors(camera)
                del self.data_handlers[camtype][yaw]
                self.data_handlers[camtype][yaw] = RGBCameraDataHandler()
        self.cameras = None

    @property
    def pedestrians(self):
        """The list of pedestrians."""
        return self._pedestrians

    @pedestrians.deleter
    def pedestrians(self):
        self.destroy_actors(self.pedestrians, self.pedestrians_ai)
        self._pedestrians, self._pedestrians_ai = [], []

    @property
    def pedestrians_ai(self):
        """The list of pedestrians AI."""
        return self._pedestrians_ai

    @property
    def vehicles(self):
        """The list of vehicles."""
        return self._vehicles

    @vehicles.deleter
    def vehicles(self):
        self.destroy_actors(self._vehicles)
        self._vehicles = []
        if self.ego_vehicle is not None:
            self.ego_vehicle = None
        self.ego_vehicle_travel_distance = None
        self.ego_vehicle_last_location = None
        if self.lidar is not None:
            self.lidar = None

    @property
    def regular_vehicles(self):
        """List of regular 4 wheels vehicles."""
        return self.get_regular_vehicles(self.vehicles)

    def flush_sensor_data_queues(self):
        """Flush data from queues."""
        if self.lidar is not None:
            self.data_handlers["lidar"].flush_data_queue()
        if self.cameras is not None:
            for camtype in self.camera_types:
                for handler in self.data_handlers[camtype].values():
                    handler.flush_data_queue()

    def get_regular_vehicles(self, vehicles):
        """Filter the vehicles to get the regular ones."""
        regs = []
        for v in vehicles:
            for model in REGULAR_VEHICLE_MODELS:
                if model in v.type_id:
                    regs.append(v)
                    break
        return regs

    def get_initial_actors_to_destroy_from_world(self):
        """Return the list of vehicles and pedestrian actors from world object."""
        actors = self.world.get_actors()
        filters = (
                ["*vehicle*", "*walker*", "*lidar*"] +
                [f"*{camtype}*" for camtype in self.camera_types]
                )
        rtn = []
        for filt in filters:
            rtn += list(actors.filter(filt))
        return rtn

    def spawn_cameras(self, width, height):
        """Spawns the ego vehicle's cameras."""
        self._logger.info("Spawning Ego vehicle's cameras.")
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle not set -> cannot spawn cameras.")
        if self.lidar is None:
            raise RuntimeError("Lidar not set -> cannot set cameras (they have same z).")
        del self.cameras
        cameras = {}
        for camtype in self.camera_types:
            cameras[camtype] = {}
            bp = self.blueprints[camtype]
            bp.set_attribute("image_size_x", str(width))
            bp.set_attribute("image_size_y", str(height))
            for yaw in self.camera_yaws:
                camera = self.spawn_actor_at(
                    bp,
                    carla.Transform(carla.Location(z=self._lidar_z), carla.Rotation(yaw=yaw)),
                    self.ego_vehicle,
                    )
                camera.listen(self.data_handlers[camtype][yaw])
                cameras[camtype][yaw] = camera
        self.cameras = cameras
        self.cam_width = width
        self.cam_height = height
        matrixK = np.identity(3)
        focal = width / (2.0)  # FOV = 90deg -> tan(fov * pi/360) = 1
        matrixK[0, 0] = matrixK[1, 1] = focal
        matrixK[0, 2] = focal  # width / 2
        matrixK[1, 2] = height / 2
        self.projection_matrix = matrixK

    def spawn_lidar(
            self, points_per_rotation_per_laser=None,
            channels=None, range_=None, lower_fov=None,
            upper_fov=None, horizontal_fov=None, rotation_frequency=None,
            points_per_second=None, z=2.4,
            ):
        """Spawn the lidar.

        Args:
            range_: float
                Lidar's range in m.
            channels: int
                Number of channels.
            lower_fov: float
                The lowest vertical angle.
            upper_fov: float
                The highest vertical angle.
            horizontal_fov: float
                The horizontal fov.
            rotation_frequency: float
                The rotation frequency.
            points_per_rotation_per_laser: int
                The number of pts per laser per rotation.
            points_per_second: int
                The total number of pts per second output by the lidar.
            z: float, optional
                Specifies how many meters above the ego vehicle's position we spawn
                the lidar.
        """
        self._logger.info("Spawning LiDAR.")
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle not set -> cannot spawn lidar.")
        del self.lidar
        bp = self.blueprints["lidar"]
        pprpl = points_per_rotation_per_laser
        if pprpl is not None:
            if points_per_second is not None:
                raise ValueError("Cannot set both 'points_per_second' and 'points_per_rotation_per_laser'.")
            if horizontal_fov is not None:
                if horizontal_fov != 360:
                    raise ValueError(
                            "horizontal_fov must be 360 if points_per_rotation_per_laser is given.")
            if rotation_frequency is None:
                raise ValueError(
                    "Need to define 'rotation_frequency' if 'points_per_rotation_per_laser' is set.")
            # if pprpl % rotation_frequency != 0:
            #     raise ValueError("pprpl must be a multiple of rotation_frequency.")
            if channels is None:
                raise ValueError(
                    "Need to define 'n_channels' if 'points_per_rotation_per_laser' is set.")
            points_per_second = int(pprpl * rotation_frequency * channels)
            self._logger.info(
                    f"LiDAR points/rot/laser set to {pprpl} gives {points_per_second} total pts/s.")
        else:
            pprpl = int(points_per_second / rotation_frequency / channels)
        attributes = {
                "range": range_, "channels": channels, "lower_fov": lower_fov, "rotation_frequency": rotation_frequency,
                "upper_fov": upper_fov, "horizontal_fov": horizontal_fov, "points_per_second": points_per_second,
                }
        for attr, value in attributes.items():
            if value is None:
                continue
            self._logger.info(f"Setting LiDAR's {attr} = {value}.")
            bp.set_attribute(attr, str(value))
        self.lidar = self.spawn_actor_at(
            bp,
            carla.Transform(carla.Location(z=z)),
            self.ego_vehicle,
        )
        self.channels = channels
        self.points_per_rotation_per_laser = pprpl
        self.lidar.listen(self.data_handlers["lidar"])
        self._lidar_z = z  # save it for cameras
        self.lidar_range = range_

    def spawn_pedestrians(
            self, max_n_pedestrians,
            percentage_pedestrians_running=10,
            percentage_pedestrians_crossing=50,
            ):
        """Try spawning 'max_n_pedestrians' NPC pedestrians.

        Args:
            max_n_pedestrians: int
                Maximum number of pedestrians.
            percentage_pedestrians_running: float > 0, < 100:
                Percentage of pedestrians running.
            percentage_pedestrians_crossing: float > 0, < 100:
                Percentage of pedestrians allowed to cross the roads.
                WARNING: this factor applies both to new and already existing pedestrians
                (world parameter).
        """
        del self.pedestrians  # this will delete the AI as well
        self._logger.info(
                f"Setting pedestrian_cross_factor to {percentage_pedestrians_crossing}%")
        self.world.set_pedestrians_cross_factor(percentage_pedestrians_crossing / 100)
        spawn_points = []
        i = 0
        while len(spawn_points) < max_n_pedestrians:
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point = carla.Transform()
                spawn_point.location = loc
                spawn_points.append(spawn_point)
            i += 1
            if i > 100 * max_n_pedestrians:
                # tried enough times...
                raise RuntimeError("Could not get spawn points for pedestrians...")
        blueprints = []
        for blueprint in self.blueprints['pedestrians']:
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            blueprints.append(blueprint)
        pedestrians = self.spawn_actors(blueprints, max_n_pedestrians, spawn_points)
        # tick to make sure all pedestrian spawned correctly
        self.world.tick()
        # spawn as many ai as there are pedestrians
        for pedestrian in pedestrians:
            ai = self.spawn_actor_at(
                self.blueprints['pedestrian_ai'], carla.Transform(), pedestrian)
            ai.start()
            ai.go_to_location(self.world.get_random_location_from_navigation())
            # first get the blueprint
            bp = self.blueprint_library.find(pedestrian.type_id)
            if bp.has_attribute('speed'):
                if 100 * random.random() > percentage_pedestrians_running:
                    # walking
                    ai.set_max_speed(float(bp.get_attribute('speed').recommended_values[1]))
                else:
                    # running
                    ai.set_max_speed(float(bp.get_attribute('speed').recommended_values[2]))
            else:
                ai.set_max_speed(0.0)
            self.pedestrians_ai.append(ai)
            self.pedestrians.append(pedestrian)
        self._logger.info(
                f"Managed to spawn n={len(pedestrians)} pedestrians! With "
                f"{percentage_pedestrians_running}% chance running.")

    def spawn_vehicles(self, max_n_vehicles):
        """Try spawning 'max_n_vehicles' NPC vehicles.

        Args:
            max_n_vehicles: int
                Maximum number of vehicles.
        """
        del self.vehicles
        # first spawn 1 regular vehicle
        self._logger.info(f"Spawning maximum {max_n_vehicles} vehicles.")
        self._logger.info("Spawning first 1 regular vehicle.")
        vehicles = self.spawn_actors(
            self.blueprints["regular_vehicles"], 1, self.map.get_spawn_points())
        if max_n_vehicles > 1:
            self._logger.info(f"Spawning maximum {max_n_vehicles-1} vehicles of any type.")
            vehicles += self.spawn_actors(
                self.blueprints['vehicles'], max_n_vehicles - 1,
                self.map.get_spawn_points())
        for vehicle in vehicles:
            vehicle.set_autopilot(True)
            self.vehicles.append(vehicle)
        # tick world to make sure everything spawned
        self.world.tick()
        self._logger.info(f"Managed to spawn a total of {len(vehicles)} vehicles!")

    def spawn_actors(self, blueprints, n, spawn_points):
        """Try spawning n actors from the given blueprints.

        Args:
            blueprints: list
                The list of blueprints to draw from randomly.
            n: int
                The number of actors to try spawning.
            spawn_points: list
                The list of locations available for spawning.

        Returns:
            list: The list of spawned actors.
        """
        if n <= 0:
            return []
        if n > len(spawn_points):
            raise ValueError(
                f"Cannot spawn more actors ({n}) than there are spawn pts ({len(spawn_points)})!")
        actors = []
        n_tries = 0
        while n_tries < 10 and len(actors) < n:
            actor = self.spawn_actor_at(
                    random.choice(blueprints), random.choice(spawn_points))
            if actor is not None:
                actors.append(actor)
                n_tries = 0
            else:
                # if it takes more than 10 tries in a row to spawn an actor
                # just give up...
                n_tries += 1
        if not len(actors):
            raise RuntimeError("Could not spawn any actor...")
        return actors

    def spawn_actor_at(self, blueprint, spawn_point, *args, **kwargs):
        """Spawn an actor from the given blueprint at the given spawn point.

        Then it returns the Carla Actor object or None if the actor could not be spawned.
        """
        return self.world.try_spawn_actor(blueprint, spawn_point, *args, **kwargs)

    def set_ego_vehicle(self):
        """Choose at random a regular vehicle to be the ego vehicle.

        Add a LiDAR on top of it.
        """
        # choose ego vehicle at random (with 4 wheels preferably)
        self._logger.info("Setting ego vehicle...")
        regs = self.regular_vehicles
        if not regs:
            raise RuntimeError("No regular vehicles to choose from. Spawn some!")
        self.ego_vehicle = random.choice(regs)
        self._logger.info(
            f"The ego vehicle is a '{self.ego_vehicle.type_id[8:].replace('.', ' ').replace('_', ' ')}'.")

    def destroy_actors(self, *actors):
        """Destroy the actors.

        Args:
            *actors:
                Lists of actors or Actor instances.
        """
        for actor in actors:
            if is_list_like(actor):
                self.destroy_actors(*actor)
                continue
            actor.destroy()

    def reset_ego_vehicle_travel_distance(self):
        """Reset the ego vehicle travel distance."""
        self.ego_vehicle_travel_distance = 0
        self.ego_vehicle_last_location = self.ego_vehicle.get_transform().location

    def tick(
            self, tick=0, frame=0, save_sensors_to=None,
            save_frame_over_distance=None, max_frames=None, save_every_tick=None):
        """Tick the actors (after world have been ticked).

        Args:
            tick: int, optional
                The tick number.
            frame: int, optional
                The frame number we're at.
            save_every_tick: int, optional
                Save every # ticks given by this argument. Default is None which means save
                at every tick. If set to e.g.: 2, data will be saved every 2 ticks instead.
                This argument is stronger than 'save_frame_over_distance'.
            save_sensors_to: str, optional
                If not None, specifies the directory where sensor data is rooted.
            save_frame_over_distance: float, optional
                If None, a frame is saved every tick.
                If not None, this tells the distance (in meters) that the ego
                vehicle needs to travel before saving a new frame.

        Returns:
            int:
                0 if we didn't save the frame.
                1 if we saved the frame.
        """
        if not self._save_this_frame(
                save_sensors_to, save_frame_over_distance, save_every_tick, tick):
            self.flush_sensor_data_queues()  # flush data to not clutter memory
            return 0
        root = os.path.join(save_sensors_to, f"frame{frame}")
        if os.path.isdir(root):
            # delete it
            shutil.rmtree(root)
        self.reset_ego_vehicle_travel_distance()
        # depth_buffers = []
        # depth_w2cs = []
        self.data_handlers["Bboxes"].reset_bboxes()
        if self.cameras is not None:
            for camtype, cams in self.cameras.items():
                for yaw, cam in cams.items():
                    self._logger.debug(f"Saving cam data for {camtype}/{yaw}")
                    campath = os.path.join(
                            root, RAW_DATA_FOLDER, camtype, f"{YAW_BASENAME}{yaw}")
                    img = self.data_handlers[camtype][yaw].process_data(
                            campath, camtype)
                    if camtype != DEPTH_CAMTYPE:
                        continue
                    # also record bboxes (with depth img)
                    self._logger.debug(f"Saving bboxes data for yaw {yaw}.")
                    bbox_path = os.path.join(
                        root, ANNOTATIONS_FOLDER, BBOX_YAW_PATH, f"{YAW_BASENAME}{yaw}")
                    self.data_handlers["Bboxes"].process_2D_data(
                        bbox_path, cam, img)
        if self.lidar is not None:
            self._logger.debug("Reading lidar data from queue.")
            self.data_handlers["lidar"].process_data(
                    os.path.join(root, RAW_DATA_FOLDER))
            # store the 3D bbox
            self.data_handlers["Bboxes"].process_3D_data(
                    os.path.join(root, ANNOTATIONS_FOLDER),
                    self.lidar)
        return 1

    def _save_this_frame(self, save_sensors_to, save_frame_over_distance, save_every_tick, tick):
        """This method decides whether or not with save data frame for given tick."""
        if save_sensors_to is None:
            return False
        if save_every_tick is not None:
            if tick % save_every_tick == 0:
                return True
            else:
                return False
        if save_frame_over_distance is None:
            # save on every tick
            return True
        # compute distance travelled since last frame save
        if self.ego_vehicle_last_location is None:
            self.reset_ego_vehicle_travel_distance()
            return False
        now_loc = self.ego_vehicle.get_transform().location
        delta = np.linalg.norm(
                [[now_loc.x - self.ego_vehicle_last_location.x],
                 [now_loc.y - self.ego_vehicle_last_location.y],
                 [now_loc.z - self.ego_vehicle_last_location.z]],
                )
        self.ego_vehicle_travel_distance += delta
        return self.ego_vehicle_travel_distance >= save_frame_over_distance

    def _set_blueprints(self):
        self.blueprint_library = self.world.get_blueprint_library()
        self.blueprints = {
            'vehicles': self.blueprint_library.filter('*vehicle*'),
            'pedestrians': self.blueprint_library.filter('*walker.pedestrian*'),
            'pedestrian_ai': self.blueprint_library.find('controller.ai.walker'),
            "lidar": self.blueprint_library.find("sensor.lidar.ray_cast_semantic"),
            "regular_vehicles": [],
        }
        for blueprint in self.blueprints["vehicles"]:
            if any(model in blueprint.id for model in REGULAR_VEHICLE_MODELS):
                self.blueprints["regular_vehicles"].append(blueprint)
        for camtype in self.camera_types:
            bp = self.blueprint_library.find(f"sensor.camera.{camtype}")
            # set height == width to have same FOV in both directions
            bp.set_attribute("fov", "90")
            self.blueprints[camtype] = bp
