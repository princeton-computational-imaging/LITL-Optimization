"""Module for point cloud intensities computers."""
import time

import cv2
import numpy as np

from .bases import BaseLidarProcessor
from .camera_processors import CameraProcessor
from .pc_projector import PointCloudProjector
from .utils import multiply_arrays
from ..utils import (
        get_available_camtypes,
        get_available_yaws,
        get_camera_array,
        get_camera_matrix,
        get_lidar_matrix,
        )
from ..settings import (
        DIFFUSE_COLOR_CAMTYPE,
        RGB_CAMTYPE,
        SPECULAR_METALLIC_ROUGHNESS_CAMTYPE,
        )


DIFFUSE_RGB_COLOR_INTENSITY_MODE = "Diffuse RGB color"
NIR_INTENSITY_MODE = "NIR model from diffuse color"


# Tags to material name see:
# https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/open3d_lidar.py#L34
TAGS_TO_MATERIAL = [
        None, "building", "fences", "other", "pedestrian", "pole",
        "road_lines", "road", "sidewalk", "vegetation", "vehicle",
        "wall", "traffic_sign", "sky", "ground", "bridge", "rail_track", "guard_rail",
        "traffic_light", "static", "dynamic", "water", "terrain",
        ]
ROAD_TAGS = [TAGS_TO_MATERIAL.index(item) for item in ("road_lines", "road")]
RETROREFLECTOR_TAGS = [TAGS_TO_MATERIAL.index(item) for item in ("road_lines", "traffic_sign")]

# albedos range from 0 to 1 and taken from:
# https://www.engineeringtoolbox.com/light-material-reflecting-factor-d_1842.html
# and from: https://www.artstation.com/blogs/shinsoj/Q9j6/pbr-color-space-conversion-and-albedo-chart
# metal albedos range from 0.5 (matte) to 0.98 (polished)

# probably it would be preferable to make these a probabilistic distributions where
# we randomly draw albedos from
MATERIAL_TO_ALBEDO = {
        None: 0,  # no hit
        "building": 0.175,     # brick: 0.1-0.15 / concrete: 0.2-0.3
        "fences": 0.5,         # assume matte metal
        "other": 0.2,          # average albedo
        "pedestrian": 0.3,     # assume skin (0.25-0.35)
        "pole": 0.5,           # same as fence
        "road_lines": 1.0,     # these are usually highly reflective
        "road": 0.14,          # aged asphalt: 0.1-0.18
        "sidewalk": 0.25,      # aged concrete: 0.2-0.3
        "vegetation": 0.1,     # summer foliage: 0.09-0.12
        "vehicle": 0.75,       # usually somewhat polished metal
        "wall": 0.175,         # same as building
        "traffic_sign": 1.0,   # usually highly reflective materials
        "sky": 0.0,            # sky does not reflect anything
        "ground": 0.14,        # ground is asphalt here
        "bridge": 0.14,        # same as road
        "rail_track": 0.5,     # same as fence
        "guard_rail": 0.5,     # same as fence
        "traffic_light": 1.0,  # same as traffic sign
        "static": 0.0,         # ???
        "dynamic": 0.0,        # ???
        "water": 0.08,         # water: 0.07-0.09
        "terrain": 0.14,       # assume same as road
        }
TAGS_TO_ALBEDO = np.array([MATERIAL_TO_ALBEDO[tag] for tag in TAGS_TO_MATERIAL])
WATER_REFRACTION_INDEX = 1.333


def load_camera_data(framedir):
    """Loads necessary camera data for intensity computing.

    Args:
        framedir: str
            The path of the framedir.
    """
    data = {}
    for camtype in [
            DIFFUSE_COLOR_CAMTYPE, SPECULAR_METALLIC_ROUGHNESS_CAMTYPE,
            RGB_CAMTYPE,
            ]:
        yaws = get_available_yaws(framedir, camtype)
        data[camtype] = {}
        for yaw in yaws:
            imarray = get_camera_array(framedir, camtype, yaw)
            camera_matrix = get_camera_matrix(framedir, camtype, yaw)
            data[camtype][yaw] = {
                    "camera_matrix": camera_matrix,
                    "imarray": imarray,
                    }
    return data


class BaseIntensityComputer(BaseLidarProcessor):
    """Base class for point cloud intensity computers."""
    intensity_model_name = None
    road_wetness_enabled = True
    ambiant_light_enabled = False

    def __init__(
            self, *args, wait_for_images=False, camera_data=None,
            snr=None, saturation=None, projection_data=None,
            **kwargs):
        """Init method for the base intensity computer."""
        super().__init__(*args, **kwargs)
        if self.intensity_model_name is None:
            raise ValueError("Intensity mode name is not set...")
        self.wait_for_images = wait_for_images
        self._cartesian_pc = None
        self.camera_data = camera_data
        self.projection_data = projection_data
        self._lidar_matrix = None
        self._available_camtypes = None
        self.ambiant_light = None
        self.snr = snr
        self.saturation = saturation
        self._cos_in_over_dist_squared = None

    @property
    def available_camtypes(self):
        """The list of available camtypes for the current frame."""
        if self._available_camtypes is None:
            if self.camera_data is None:
                self._available_camtypes = get_available_camtypes(self.framedir)
            else:
                self._available_camtypes = tuple(self.camera_data.keys())
        return self._available_camtypes

    @property
    def lidar_matrix(self):
        """The lidar matrix for the current frame."""
        if self._lidar_matrix is None:
            self._lidar_matrix = get_lidar_matrix(self.framedir)
        return self._lidar_matrix

    @lidar_matrix.setter
    def lidar_matrix(self, matrix):
        self._lidar_matrix = matrix

    @property
    def cartesian_pc(self):
        """The PC in cartesian coordinates (only where hitmask is True)."""
        if self._cartesian_pc is None:
            # copy is important here
            self._cartesian_pc = self.convert_to_cartesian(self.raw_pc.copy(), self.hitmask)
        return self._cartesian_pc

    def add_road_wetness(
            self, road_wetness_depth, road_thread_profile_depth):
        """Augment with road wetness noise."""
        if not self.road_wetness_enabled:
            raise RuntimeError("Tried to add road wetness but it is not enabled...")
        pc = self.raw_pc
        tags = self.raw_tags
        hitmask = self.hitmask
        road_wetness_ratio = np.clip(road_wetness_depth / road_thread_profile_depth, 0, 1)
        # only add wetness on ROADS!
        for tag_idx in ROAD_TAGS:
            where = np.logical_and(tags[..., 0] == tag_idx, hitmask)
            cos_in = pc[where, 3]
            alpha_in = np.arccos(cos_in)
            alpha_out = np.arcsin(np.sin(alpha_in) / WATER_REFRACTION_INDEX)
            cos_out = np.cos(alpha_out)
            # compute T_air
            Tair_para, Tair_perp = self.fresnel(
                    cos_in, cos_out, 1.0, WATER_REFRACTION_INDEX, reflections=False)
            # compute T_water and R_water (inverse of air parameters)
            Twater_para, Twater_perp, Rwater_para, Rwater_perp = self.fresnel(
                    cos_out, cos_in, WATER_REFRACTION_INDEX, 1.0, reflections=True)
            # now compute Ttotal
            rho0 = TAGS_TO_ALBEDO[tag_idx]  # this (or equivalent) is already applied
            Ttotal_perp = np.divide(np.multiply(Tair_perp, Twater_perp), 1 - rho0 * Rwater_perp)
            Ttotal_para = np.divide(np.multiply(Tair_para, Twater_para), 1 - rho0 * Rwater_para)
            # concatenate and keep the maximum for each
            Ttotal = np.max(np.concatenate([Ttotal_perp, Ttotal_para], axis=0).T, axis=0)
            newrho0 = (1 - road_wetness_ratio) * rho0 + road_wetness_ratio * np.divide(Ttotal, cos_in)
            self.processed_intensities[where] *= newrho0 / rho0  # just correction of initial albedo
        self.processed_intensities /= np.max(self.processed_intensities)

    @staticmethod
    def fresnel(
            cos_in, cos_out, n_in, n_out, reflections=True):
        """Compute the R and T coefficients from angles and refraction indices.

        Basically an implementation of Fresnel equations.
        """
        # start by t_perp and t_paral
        n_in_cos_in = n_in * cos_in
        n_in_cos_in_times_2 = 2 * n_in_cos_in
        n_out_cos_out = n_out * cos_out
        n_out_cos_in = n_out * cos_in
        n_in_cos_out = n_in * cos_out
        n_in_cos_in_times_2_squared = np.square(n_in_cos_in_times_2)
        n_in_cos_in_plus_n_out_cos_out_squared = np.square(n_in_cos_in + n_out_cos_out)
        n_out_cos_in_plus_n_in_cos_out_squared = np.square(n_out_cos_in + n_in_cos_out)
        t_perp2 = np.divide(n_in_cos_in_times_2_squared, n_in_cos_in_plus_n_out_cos_out_squared)
        t_paral2 = np.divide(n_in_cos_in_times_2_squared, n_out_cos_in_plus_n_in_cos_out_squared)
        t_factor = np.divide(n_in_cos_in, n_out_cos_out)
        Tparal = np.multiply(t_factor, t_paral2)
        Tperp = np.multiply(t_factor, t_perp2)
        if not reflections:
            return Tparal, Tperp
        # now for reflections
        Rperp = np.divide(np.square(n_in_cos_in - n_out_cos_out),
                          n_in_cos_in_plus_n_out_cos_out_squared)
        Rpara = np.divide(np.square(n_out_cos_in - n_in_cos_out),
                          n_out_cos_in_plus_n_in_cos_out_squared)
        return Tparal, Tperp, Rpara, Rperp

    def get_cos_in_over_dist_squared(self, **kwargs):
        """Return cos(alpha_in) / R^2 for given pt cloud (hit only)."""
        # times 2 as light travels twice the distance to the pt
        if self._cos_in_over_dist_squared is not None:
            return self._cos_in_over_dist_squared
        dist_squared = np.square(2 * self.get_distance(**kwargs))
        self._cos_in_over_dist_squared = np.divide(self.get_cos_in(**kwargs), dist_squared)
        return self._cos_in_over_dist_squared

    def get_cos_in(self, apply_hitmask=True):
        """Return cos(alpha_in) for hit rays."""
        if apply_hitmask:
            return self.raw_pc[self.hitmask, 3]
        return self.raw_pc[..., 3]

    def get_distance(self, apply_hitmask=True):
        """Get distance array."""
        if apply_hitmask:
            return self.raw_pc[self.hitmask, 2]
        # prevent divisions by 0
        dists = self.raw_pc[..., 2].copy()
        dists[dists == 0.0] = 1 / 10000
        return dists

    def process_camera_frames(self, camtype):
        """Process the camera frames."""
        self._logger.info("Need to process camera frames for '{camtype}'.")
        cam_processor = CameraProcessor(framedir=self.framedir, camtype=camtype)
        cam_processor.process(False)
        self._logger.info("Camera frames processed!")
        self.refresh_available_camtypes()

    def refresh_available_camtypes(self):
        """Refresh available camtypes list."""
        self._available_camtypes = None

    def saturate(self, signal):
        """Saturate a signal."""
        return np.clip(signal, 0.0, self.saturation)


class UniformPointCloudIntensityComputer(BaseIntensityComputer):
    """Assigns uniform intensity for each point."""
    intensity_model_name = "Lambertian"

    def process(self):
        """Uniform 'base intensity' for all pts.

        Intensity is scaled with cos(alpha_in) / R^2
        """
        intensities = np.zeros_like(self.raw_pc[..., 0])
        intensities[self.hitmask] += self.get_cos_in_over_dist_squared()
        intensities = self.saturate(intensities)
        intensities /= np.max(intensities)
        self.processed_intensities = intensities
        self.processed_pc = self.raw_pc


class SemanticTagAlbedoPointCloudIntensityComputer(BaseIntensityComputer):
    """Assigns an albedo according to the tag of the object hit."""
    intensity_model_name = "Semantic tag albedo"

    def process(self):
        """The intensity computed is rho0 x cos(alpha_in) / R^2.

        rho0 here is taken from a list of albedo values depending of the tag of the object hit.
        """
        intensity = np.zeros_like(self.raw_pc[..., 0])
        for tag_idx in np.unique(self.raw_tags[..., 0]):
            where = np.logical_and(self.raw_tags[..., 0] == tag_idx, self.hitmask)
            if not where.any():
                # nothing to do
                continue
            albedo = TAGS_TO_ALBEDO[tag_idx]
            intensity[where] = albedo * self.get_cos_in_over_dist_squared(
                    apply_hitmask=False,
                    )[where]
        if np.max(intensity) == 0.0:
            raise RuntimeError("No hits at all...")
        # normalize & saturate
        intensity = self.saturate(intensity)
        intensity /= np.max(intensity)
        self.processed_intensities = intensity


class DiffuseColorPointCloudIntensityComputer(BaseIntensityComputer):
    """Assigns an intensity which is the RGB color from the diffuse image."""
    intensity_model_name = DIFFUSE_RGB_COLOR_INTENSITY_MODE
    road_wetness_enabled = False

    def get_projection_intensities_for_camtype(self, camtype):
        """Return the projection intensities for the given camtype."""
        if camtype not in self.available_camtypes:
            if not self.wait_for_images:
                raise FileNotFoundError(
                    f"No camtype '{camtype}' for frame {self.framedir}. "
                    f"Available camtypes: {self.available_camtypes}")
            else:
                # wait until they are available
                self._available_camtypes = None  # reset
                while camtype not in self.available_camtypes:
                    time.sleep(5)
                    self._logger.info(f"Waiting for camtype to appear: '{camtype}'.")
        shape = list(self.cartesian_pc.shape[:-1]) + [3]  # RGB
        if self.camera_data is None:
            self.camera_data = load_camera_data(self.framedir)
        if self.projection_data is not None:
            return self.projection_data[camtype]
        intensities = np.zeros(shape, dtype=np.uint8)
        for data in self.camera_data[camtype].values():
            camera_matrix = data["camera_matrix"]
            imarray = data["imarray"]
            projector = PointCloudProjector(
                    self.cartesian_pc, self.lidar_matrix, camera_matrix,
                    imarray.shape[1],
                    imarray.shape[0],
                    hitmask=self.hitmask, loglevel=self._loglevel)
            uvcoords = projector.compute_uv_projection()
            inside_mask = projector.inside_mask
            intensities[inside_mask, :] = imarray[uvcoords[:, 0], uvcoords[:, 1], :3]
            del projector
        return intensities

    def process(self):
        """The intensity computed is the RGB color from diffuse images."""
        # here intensities are 8bit uints (0-255)
        intensities = self.get_projection_intensities_for_camtype(DIFFUSE_COLOR_CAMTYPE)
        # convert intensities to floats and normalize
        intensities = intensities.astype(float)
        # intensities /= np.max(intensities)
        # normalize by 255
        intensities /= 255
        # cos_over_r2 = self.get_cos_in_over_dist_squared()
        # for channel in range(intensities.shape[-1]):
        #     intensities[self.hitmask, channel] = np.multiply(intensities[self.hitmask, channel], cos_over_r2)
        #     # normalize for each channel
        #     intensities[self.hitmask, channel] /= np.max(intensities[self.hitmask, channel])
        opacity = np.zeros(intensities.shape[:-1], dtype=np.float64)
        opacity[self.hitmask] += 0.5
        rgba = np.concatenate((intensities, opacity[..., np.newaxis]), axis=-1)
        self.processed_intensities = rgba


class AIODRiveModelPointCloudIntensityComputer(DiffuseColorPointCloudIntensityComputer):
    """Compute lidar intensity from base color using a NIR model (source: mario's code on slack)."""
    # basically this takes the diffuse color and outputs a gray-level intensity
    intensity_model_name = "AIODrive shader"
    road_wetness_enabled = True

    def normalize_over_average(self, img):
        """Normalize an array by its average."""
        mean = img.mean()
        if mean == 0.0:
            raise ZeroDivisionError(f"img{img.shape} has 0 mean...")
        return img / img.mean()

    def process(self, *args, use_cosr2=True, normalize=True, **kwargs):
        """The intensity from the NIR model."""
        super().process(*args, **kwargs)
        intensities = np.zeros_like(self.raw_pc[..., 0])
        intensities[self.hitmask] = self.get_projection_intensities_for_camtype(
                RGB_CAMTYPE)[self.hitmask, 0].astype(float) / 255
        # rgba is an (RGBA) array for the point cloud colors corresponding to diffuse image
        # multiply by cos(incident angle) / R^2 for each channel
        if use_cosr2:
            intensities[self.hitmask] *= self.get_cos_in_over_dist_squared()
        if normalize:
            intensities[self.hitmask] = self.normalize_over_average(intensities[self.hitmask])
        self.processed_intensities = self.saturate(intensities)


class NIRModelPointCloudIntensityComputer(DiffuseColorPointCloudIntensityComputer):
    """Compute lidar intensity from base color using a NIR model (source: mario's code on slack)."""
    # basically this takes the diffuse color and outputs a gray-level intensity
    intensity_model_name = NIR_INTENSITY_MODE
    road_wetness_enabled = True

    def normalize_over_average(self, img):
        """Normalize an array by its average."""
        mean = img.mean()
        if mean == 0.0:
            raise ZeroDivisionError(f"img{img.shape} has 0 mean...")
        return img / img.mean()

    def process(self, *args, use_cosr2=True, normalize=True, **kwargs):
        """The intensity from the NIR model."""
        super().process(*args, **kwargs)
        rgba = self.processed_intensities
        # rgba is an (RGBA) array for the point cloud colors corresponding to diffuse image
        # multiply by cos(incident angle) / R^2 for each channel
        if use_cosr2:
            cos_over_r2 = self.get_cos_in_over_dist_squared()
            for channel in range(rgba.shape[-1]):
                rgba[self.hitmask, channel] = np.multiply(rgba[self.hitmask, channel], cos_over_r2)
                # normalize for each channel
                rgba[self.hitmask, channel] = self.saturate(rgba[self.hitmask, channel])
                rgba[self.hitmask, channel] /= np.max(rgba[self.hitmask, channel])
        inverse = np.copy(rgba[..., :-1])  # don't consider opacity (only contains hitmask info)
        gt0 = inverse > 0
        inverse[gt0] = 1 - inverse[gt0]
        img = np.maximum(rgba[..., :-1], inverse)  # [:, [2, 1, 0]]  # switch to BGR ordering? not sure why
        # irimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert to gray img (using cv2 algorithm)
        ir = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        # ir = np.concatenate(ir, img[:, 3], axis=-1)
        # adjust gamma
        gamma = 0.25
        invgamma = 1 / gamma
        table = np.array([((i / 255) ** invgamma) * 255 for i in range(256)], dtype=np.uint8)
        # convert img to np.uint8 for applying gamma
        ir = (255 * ir).astype(np.uint8)
        # bring back opacity in the end
        ir = cv2.LUT(ir.astype(np.uint8), table).astype(np.float64)
        # convert back to floats in [0, 1]
        ir /= 255
        # I dont know where this comes from. Diffuse reflectance is scaled between 0-1.
        # if normalize:
        #    ir = self.normalize_over_average(ir)
        self.processed_intensities = ir


class NIRWithOrenNayarPointCloudIntensityComputer(NIRModelPointCloudIntensityComputer):
    """Computes lidar intensity from base color using NIR model and adds on top the Oren-Nayar model for roughness."""
    intensity_model_name = "NIR + Oren-Nayar roughness models"

    def process(self, *args, **kwargs):
        """The intensity from the NIR + oren nayer model."""
        super().process(*args, normalize=False, **kwargs)
        intensities = self.processed_intensities
        # here intensities is npts x 2 where 2 = (gray intensity, opacity) in [0, 1] format
        # get roughness values
        specular_metal_roughness = self.get_projection_intensities_for_camtype(
                SPECULAR_METALLIC_ROUGHNESS_CAMTYPE).astype(np.float64)
        # ROUGHNESS IS SET TO G channel (index = 1)
        roughness = specular_metal_roughness[self.hitmask, 1] / 255  # G channel is roughness (from 0 to 255)
        # in the Oren-Nayar model, sigma is the variance of the slopes of the micro-facets [0, infinity)
        # 0 roughness = completely smooth (mirror) while 1 = very rough (matte)
        # for now lets assume sigma^2 == roughness
        A = 1 - 0.5 * np.divide(roughness, roughness + 0.33)
        B = 0.45 * np.divide(roughness, roughness + 0.09)
        cos_in = self.get_cos_in()
        theta_in = np.arccos(cos_in)
        sin_in = np.sin(theta_in)
        factor = A + B * np.divide(np.square(sin_in), cos_in)
        intensities[self.hitmask] *= factor * self.get_cos_in_over_dist_squared()
        intensities = self.saturate(intensities)
        # renormalize
        intensities[self.hitmask] /= np.max(intensities[self.hitmask])
        self.processed_intensities = intensities


class NIRWithUnrealEngineRoughnessShaderPointCloudIntensityComputer(NIRModelPointCloudIntensityComputer):
    """Computes lidar intensity from base color using NIR model and adds on top the UE's roughness shader."""
    intensity_model_name = "NIR + UE's roughness shader"
    ambiant_light_enabled = True

    def saturate_retroreflectors(self):
        """Augment with road wetness noise."""
        tags = self.raw_tags
        hitmask = self.hitmask
        # only add wetness on ROADS!
        for tag_idx in RETROREFLECTOR_TAGS:
            where = np.logical_and(tags[..., 0] == tag_idx, hitmask)
            self.processed_intensities[where] = 1
        # self.processed_intensities[self.hitmask] = (
        #   self.normalize_over_average(self.processed_intensities[self.hitmask]))

    def compute_ambiant_light(self, add_to_processed_intensities=True, snr=None):
        """Adds abiant light intensity to each point."""
        # only keep r data
        i_r = self.get_projection_intensities_for_camtype(RGB_CAMTYPE)[..., 0]
        i_r = self.normalize_over_average(i_r)
        self.ambiant_light = i_r
        if add_to_processed_intensities:
            self.processed_intensities = self.snr * self.processed_intensities + i_r

    def process(self, *args, **kwargs):
        """Process the intensity."""
        super().process(*args, normalize=False, use_cosr2=False, **kwargs)
        diffuse = self.processed_intensities[self.hitmask]  # this would correspond to NIR image in range [0, 1]
        # here intensities is gray img in uint8 format
        # get roughness values
        specular_metal_roughness = self.get_projection_intensities_for_camtype(
                SPECULAR_METALLIC_ROUGHNESS_CAMTYPE)
        # ROUGHNESS IS SET TO G channel (index = 1)
        # G channel is roughness (from 0 to 255)
        roughness = specular_metal_roughness[self.hitmask, 1].astype(float) / 255
        # metallic is B channel
        metallic = specular_metal_roughness[self.hitmask, 2].astype(float) / 255
        # diffuse term is basically lambert (diffuse color) unless metallic -> 0
        # we can thus just multiply the metallic values and diffuse term
        diffuse_term = np.multiply(diffuse, 1 - metallic)
        # shader is described here:
        # https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        # there are multiple factors to compute
        # see paper for more details
        alpha2 = np.power(roughness, 4)
        k = np.square(roughness + 1) / 8
        # specular term is R channel
        specular = specular_metal_roughness[self.hitmask, 0].astype(float) / 255
        cos_in = self.get_cos_in()
        numerator = multiply_arrays(alpha2, specular)
        cos_in2 = np.square(cos_in)
        denom1 = np.multiply(cos_in2, alpha2 - 1) + 1
        denom2 = np.multiply(cos_in, 1 - k) + k
        denom = 4 * np.square(np.multiply(denom1, denom2))
        # # prevent divisions by 0
        denom_eq_0 = denom == 0.0
        if (denom_eq_0).any():
            denom[denom_eq_0] = denom[denom > 0.0].min()
        specular_term = np.divide(numerator, denom)
        final = self.saturate(np.multiply(specular_term + diffuse_term, self.get_cos_in_over_dist_squared()))
        notfinite = np.logical_not(np.isfinite(final))
        if notfinite.any():
            final[notfinite] = final[np.logical_not(notfinite)].max()
        # renormalize
        # intensities[self.hitmask] /= np.max(intensities[self.hitmask])
        self.processed_intensities[self.hitmask] = self.normalize_over_average(final)


class NIRWithUnrealEngineRoughnessShaderPointCloudIntensityComputerv2(
        NIRModelPointCloudIntensityComputer):
    """Computes lidar intensity from base color using NIR model and adds on top the UE's roughness shader.

    Updated intensity scales.
    """
    intensity_model_name = "Cook-Torrance"
    ambiant_light_enabled = True

    def saturate_retroreflectors(self):
        """Augment with road wetness noise.

        Sources: https://highways.dot.gov/safety/other/visibility/sign-retroreflectivity
                 https://www.safetysign.com/determining-reflectivity
                 https://safety.fhwa.dot.gov/roadway_dept/night_visib/sign_visib/sheetguide/sheetguide2014.pdf
        """
        tags = self.raw_tags
        hitmask = self.hitmask
        # only add wetness on ROADS!
        for tag_idx in RETROREFLECTOR_TAGS:
            where = np.logical_and(tags[..., 0] == tag_idx, hitmask)
            self.processed_intensities[where] = 3
            #  choose from 1 (parking lots), 3 (low traffic), 10 (heavy traffic)
        # self.processed_intensities[self.hitmask] = (
        #   self.normalize_over_average(self.processed_intensities[self.hitmask]))

    def get_cos_in_over_area(self):
        """Return cos(alpha_in) / R^2 for given pt cloud (hit only)."""
        # times 2 as light travels twice the distance to the pt

        # assume a beamdivergence theta of max 360/horizontal_resolution=360/875=0.41°
        # the spehere surface then is O=2*pi*r^2(1-cos(theta))
        area = np.square(2*self.get_distance())*np.pi*(1-np.cos(0.41/180*np.pi))
        self._cos_area = np.divide(self.get_cos_in(), area)
        return self._cos_area

    def compute_ambiant_light(self, add_to_processed_intensities=False, snr=None):
        """Adds abiant light intensity to each point."""
        # only keep r data
        i_r = self.get_projection_intensities_for_camtype(RGB_CAMTYPE)[..., 0]/255
        # i_r = self.normalize_over_average(i_r)/100
        # Ambient light from 880nm - 920nm corresponds with a opening angle of 0.41 to 40Watt at 100m at 10m
        # this correspongs to 10Watt
        area = np.square(self.raw_pc[:, :, 2])*np.pi*(1-np.cos(0.41/180*np.pi))
        self.ambiant_light = i_r*area
        if add_to_processed_intensities:
            self.processed_intensities = self.snr * self.processed_intensities + i_r

    def process(self, *args, **kwargs):
        """Process the intensity."""
        super().process(*args, normalize=False, use_cosr2=False, **kwargs)
        # this would correspond to NIR image in range [0, 1]
        diffuse = self.processed_intensities[self.hitmask]
        # here intensities is gray img in uint8 format
        # get roughness values
        specular_metal_roughness = self.get_projection_intensities_for_camtype(
                SPECULAR_METALLIC_ROUGHNESS_CAMTYPE)
        # ROUGHNESS IS SET TO G channel (index = 1)
        # G channel is roughness (from 0 to 255)
        # roughness + metallic should lead to 1
        roughness = specular_metal_roughness[self.hitmask, 1]
        # r_idx = np.where(roughness==0)
        roughness = roughness.astype(float) / 255
        # metallic is B channel
        metallic = specular_metal_roughness[self.hitmask, 2].astype(float) / 255
        # diffuse term is basically lambert (diffuse color) unless metallic -> 0
        # we can thus just multiply the metallic values and diffuse term
        diffuse_term = np.multiply(diffuse, 1-metallic)
        # shader is described here:
        # https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        # there are multiple factors to compute
        # see paper for more details
        alpha2 = np.power(roughness, 4)
        k = np.square(roughness + 1) / 8
        # specular term is R channel
        specular = specular_metal_roughness[self.hitmask, 0].astype(float) / 255

        # cos of the ray incident angle
        cos_in = self.get_cos_in()
        numerator = np.multiply(alpha2, specular)
        cos_in2 = np.square(cos_in)
        denom1 = np.multiply(cos_in2, alpha2 - 1) + 1
        denom2 = np.multiply(cos_in, 1 - k) + k
        # introduce small constant for numerical stability
        denom = 4 * np.square(np.multiply(denom1, denom2)) + 0.004
        # # prevent divisions by 0
        denom_eq_0 = denom == 0.0
        if (denom_eq_0).any():
            denom[denom_eq_0] = denom[denom > 0.0].min()
        specular_term = np.divide(numerator, denom)

        final = self.saturate(np.multiply(specular_term + diffuse_term, self.get_cos_in_over_area()))
        # final = specular_term #numerator # _term
        notfinite = np.logical_not(np.isfinite(final))
        # check if value exceeds 1
        if notfinite.any():
            final[notfinite] = final[np.logical_not(notfinite)].max()
        # renormalize
        # intensities[self.hitmask] /= np.max(intensities[self.hitmask])
        self.processed_intensities[self.hitmask] = final  # self.normalize_over_average(final)
        self.processed_intensities[np.logical_not(self.hitmask)] = 0


# The following list/dict should be imported everywhere we want an intensity computer
# while letting the user choosing it.
AVAILABLE_INTENSITY_PROCESSOR_CLS = [
        UniformPointCloudIntensityComputer,
        SemanticTagAlbedoPointCloudIntensityComputer,
        DiffuseColorPointCloudIntensityComputer,
        AIODRiveModelPointCloudIntensityComputer,
        NIRModelPointCloudIntensityComputer,
        NIRWithOrenNayarPointCloudIntensityComputer,
        NIRWithUnrealEngineRoughnessShaderPointCloudIntensityComputer,
        NIRWithUnrealEngineRoughnessShaderPointCloudIntensityComputerv2,
        ]
INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS = {
        cls.intensity_model_name: cls for cls in AVAILABLE_INTENSITY_PROCESSOR_CLS}
DISABLED_ROAD_WETNESS_MODELS = [
        cls.intensity_model_name for cls in AVAILABLE_INTENSITY_PROCESSOR_CLS if not cls.road_wetness_enabled
        ]
