"""Simulation module."""
import carla
import time

from .actor_manager import ActorManager
from ..bases import BaseUtility


def format_time(seconds):
    """Return a formatted time string."""
    nhrs = seconds // 3600
    seconds -= nhrs * 3600
    mins = seconds // 60
    seconds -= 60 * mins
    seconds = round(seconds, 1)
    if nhrs > 0:
        return f"{nhrs}h{mins}m{seconds}s"
    if mins > 0:
        return f"{mins}m{seconds}s"
    return f"{seconds}s"


class LiDARSim(BaseUtility):
    """LiDAR Sim class for easier simulation handling."""

    def __init__(
        self,
        fixed_delta_seconds=0.1,
        timeout=None,
        hostname='localhost',
        **kwargs,
    ):
        """Initialize the sim.

        Args:
            fixed_delta_seconds: float
                The physical time between each world tick in seconds.
            timeout: int, optional
                States the server timeout allowed for the client.
            hostname: str
                The hostname of the CARLA server.
            loglevel: int
                The logging level.
        """
        super().__init__(**kwargs)

        # connect to client and gather general objects
        self.client = carla.Client(hostname, 2000)
        if timeout is not None:
            self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.traffic_manager = self.client.get_trafficmanager()
        self.apply_initial_settings(fixed_delta_seconds=fixed_delta_seconds)
        self.actor_manager = ActorManager(self.world, loglevel=self._loglevel)
        self._last_progress_log = None

    @property
    def lidar(self):
        """The lidar actor object."""
        return self.actor_manager.lidar

    @property
    def ego_vehicle(self):
        """The simulated ego vehicle."""
        return self.actor_manager.ego_vehicle

    def apply_initial_settings(self, fixed_delta_seconds):
        """Apply initial settings."""
        self._logger.info(f"Synchro = True, fixed delta seconds = {fixed_delta_seconds}, TM synchronous")
        settings = self.world.get_settings()
        if fixed_delta_seconds > settings.max_substep_delta_time * settings.max_substeps:
            self._logger.error(f"max_substep_delta_time = {settings.max_substep_delta_time}")
            self._logger.error(f"max_substeps = {settings.max_substeps}")
            raise ValueError("fixed_delta_seconds should be <= max_substep_delta_time * max_substeps")
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.traffic_manager.set_synchronous_mode(True)
        self.world.apply_settings(settings)

    def run(self, watch=None, max_frames=None, max_ticks=None, **kwargs):
        """Run the simulation."""
        self._logger.info("Running simulation...")
        print_once = {"watch": False}
        n_ticks = 0
        n_frames = 0
        start_time = time.time()
        if max_ticks is not None and max_frames is not None:
            raise ValueError("Cannot have both a max number of frames and max number of ticks.")
        try:
            while True:
                self.watch_while_running(watch, print_once)
                self.world.tick()
                n_frames += self.actor_manager.tick(tick=n_ticks, frame=n_frames, **kwargs)
                n_ticks += 1
                if max_ticks is not None:
                    self._print_run_progress(n_ticks, max_ticks)
                    if n_ticks >= max_ticks:
                        break
                if max_frames is not None:
                    self._print_run_progress(n_frames, max_frames)
                    if n_frames >= max_frames:
                        break
        except KeyboardInterrupt:
            pass
        end = time.time()
        tot = end - start_time  # in seconds
        self._logger.info(f"Simulation stopped. Total simulation time: {format_time(tot)}.")

    def spawn_npc_pedestrians(self, *args, **kwargs):
        """Spawns npc pedestrians."""
        self.actor_manager.spawn_pedestrians(*args, **kwargs)

    def spawn_npc_vehicles(self, *args, ego=True, **kwargs):
        """Spawn npc vehicles."""
        self.actor_manager.spawn_vehicles(*args, **kwargs)
        if ego:
            self.actor_manager.set_ego_vehicle()

    def spawn_lidar(self, *args, **kwargs):
        """Spawns the lidar on top of ego vehicle."""
        self.actor_manager.spawn_lidar(*args, **kwargs)

    def spawn_cameras(self, *args, **kwargs):
        """Spawns a camera on the same spot as the lidar."""
        self.actor_manager.spawn_cameras(*args, **kwargs)

    def watch_while_running(self, watch, print_once):
        """Move spectator in order to watch a given actor."""
        if watch is None:
            return
        if watch == "ego" or watch is self.ego_vehicle:
            if print_once["watch"]:
                self._logger.info("Following ego vehicle.")
                print_once["watch"] = True
            watch = self.ego_vehicle
        if isinstance(watch, carla.Actor):
            if print_once["watch"]:
                self._logger.info(f"Following actor {watch.type_id}:{watch.id}.")
                print_once["watch"] = True
            watch = watch
            watch_error = self.watch_actor(watch)
            if watch_error:
                watch = None
        else:
            raise ValueError('Can only watch carla Actors.')

    def watch_actor(self, actor, third_person_view_dist_xy=5, third_person_view_dist_z=3):
        """Move the spectator in a 3rd person view around actor."""
        try:
            transform = actor.get_transform()  # R0
            forward = transform.get_forward_vector()
            # apply shift to get 3rd person view position
            transform.location.x -= forward.x * third_person_view_dist_xy
            transform.location.y -= forward.y * third_person_view_dist_xy
            transform.location.z = third_person_view_dist_z
            # transform.rotation.pitch = -16
            self.spectator.set_transform(transform)
        except RuntimeError as e:
            self._logger.error("Could not follow actor...")
            self._logger.exception(e)
            return 'error'
        return None

    def _print_run_progress(self, n, max_):
        five_percent = int(round(max_ / 20))
        if max_ >= 20:
            if n % five_percent != 0:
                return
            log = f"Simulation progress: {int(round(n / five_percent)) * 5} %"
        else:
            percent = int(round(n / max_ * 100, 0))
            log = f"Simulation progress: {percent} %"
        if log != self._last_progress_log:
            self._logger.info(log)
            self._last_progress_log = log
