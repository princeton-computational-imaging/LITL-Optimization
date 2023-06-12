"""Pipeline module."""
import json
import os

import numpy as np

from .color_processors import ColorProcessor
from .hitmask_processors import BasicHitMaskProcessor
from .intensity_models import INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS
from .noise_models import NOISE_MODEL_2_NOISE_PROCESSOR_CLS
from .thresholding_algorithms import THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS
from .waveform_modelling import WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS

from .bases import BaseProcessor
from .defaults import GLOBAL_DEFAULTS, NOISE_MODELS_DEFAULTS, WAVEFORM_MODELS_DEFAULTS
from ..settings import (
        PC_FILENAME,
        PROCESSED_DATA_FOLDER,
        PROCESSED_COLORS_FILENAME, PROCESSED_INTENSITIES_FILENAME,
        PROCESSED_PC_FILENAME,
        METADATA_FILENAME,
        RAW_DATA_FOLDER,
        TAGS_FILENAME,
        )
from ..utils import carla2spherical, sanitize_preset_name


class ProcessPipeline(BaseProcessor):
    """Process pipeline for a raw point cloud -> final point cloud."""

    def __init__(
            self, processing_parameters, framedir=None, wait_for_images=False,
            **kwargs):
        """Process pipeline's init method."""
        super().__init__(framedir=framedir, **kwargs)
        # used in many pipeline layers
        self.waveform_parameters = WAVEFORM_MODELS_DEFAULTS.copy()
        self.noise_parameters = NOISE_MODELS_DEFAULTS.copy()
        self.global_parameters = GLOBAL_DEFAULTS.copy()
        self.processing_parameters = processing_parameters
        for k in self.global_parameters:
            if k in self.processing_parameters:
                # update
                self.global_parameters[k] = self.processing_parameters[k]
        if "waveform_model" in processing_parameters:
            self.waveform_parameters = processing_parameters["waveform_model"]
        if "noise_model" in processing_parameters:
            self.noise_parameters = processing_parameters["noise_model"]
        self.upsampling_ratio = self.waveform_parameters["upsampling_ratio"]
        self.preset_name = None
        self.metadatapath = None
        self.pc_out_path = None
        self.intensities_out_path = None
        self.colors_out_path = None
        self.pc_path = None
        self.tags_path = None
        self.raw_pc = None
        self.raw_tags = None
        if framedir is not None:
            self.preset_name = sanitize_preset_name(
                    self.processing_parameters["preset_name"])
            self.presetdir = os.path.join(self.framedir, PROCESSED_DATA_FOLDER, self.preset_name)
            self.metadatapath = os.path.join(self.presetdir, METADATA_FILENAME)
            self.pc_out_path = os.path.join(self.presetdir, PROCESSED_PC_FILENAME)
            self.intensities_out_path = os.path.join(
                    self.presetdir, PROCESSED_INTENSITIES_FILENAME)
            self.colors_out_path = os.path.join(
                    self.presetdir, PROCESSED_COLORS_FILENAME)
            self.pc_path = os.path.join(self.framedir, RAW_DATA_FOLDER, PC_FILENAME)
            self.tags_path = os.path.join(self.framedir, RAW_DATA_FOLDER, TAGS_FILENAME)
        # processors
        self.hitmask_processor = BasicHitMaskProcessor(loglevel=self._loglevel)
        intensity_processor_cls = INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS[self.intensity_model]
        self.intensity_processor = intensity_processor_cls(
                framedir=self.framedir,
                snr=self.global_parameters["snr"],
                saturation=self.global_parameters["saturation"],
                loglevel=self._loglevel,
                wait_for_images=wait_for_images,
                )
        thresholding_processor_cls = THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS[self.thresholding_algorithm]
        self.thresholding_processor = thresholding_processor_cls(
                framedir=self.framedir,
                loglevel=self._loglevel,
                # upsampling_ratio=self.upsampling_ratio,
                saturation=self.intensity_processor.saturation,
                snr=self.intensity_processor.snr,
                **self.processing_parameters["thresholding_algorithm"],
                **self.waveform_parameters,
                )
        noise_processor_cls = NOISE_MODEL_2_NOISE_PROCESSOR_CLS[self.noise_model]
        self.noise_processor = noise_processor_cls(
                loglevel=self._loglevel,
                **self.noise_parameters,
                )
        self.color_processor = ColorProcessor(
                loglevel=self._loglevel)
        self.waveform_processor = WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS[self.waveform_model](
                loglevel=self._loglevel,
                **self.waveform_parameters,
                snr=self.intensity_processor.snr,
                )

    @property
    def camera_data(self):
        """The camera data."""
        return self.intensity_processor.camera_data

    @camera_data.setter
    def camera_data(self, camera_data):
        self.intensity_processor.camera_data = camera_data

    @property
    def projection_data(self):
        """The camera projection data."""
        return self.intensity_processor.projection_data

    @projection_data.setter
    def projection_data(self, projection_data):
        self.intensity_processor.projection_data = projection_data

    @property
    def lidar_matrix(self):
        """The lidar matrix."""
        return self.intensity_processor.lidar_matrix

    @lidar_matrix.setter
    def lidar_matrix(self, matrix):
        self.intensity_processor.lidar_matrix = matrix

    @property
    def intensity_model(self):
        """The intensity model."""
        return self.intensity_model_parameters["model"]

    @property
    def intensity_model_parameters(self):
        """The intensity model parameters."""
        return self.processing_parameters["intensity_model"]

    @property
    def noise_model(self):
        """The noise model."""
        return self.noise_parameters["model"]

    @property
    def thresholding_algorithm(self):
        """The thresholding algorithm."""
        return self.processing_parameters["thresholding_algorithm"]["algorithm"]

    @property
    def waveform_model(self):
        """The waveform model."""
        return self.waveform_parameters["model"]

    @property
    def need_recompute(self):
        """Return True if we need to recompute."""
        # check metadata first
        if not os.path.isfile(self.metadatapath):
            return True
        try:
            with open(self.metadatapath, "r") as f:
                metadata = json.load(f)
        except json.decoder.JSONDecodeError:
            self._logger.warning(
                    f"Metadata file '{self.metadatapath}' was corrupt so we killed it VIVA LA REVOLUCION!")
            os.remove(self.metadatapath)
            return True
        return metadata != self.processing_parameters

    def add_noise(self):
        """Adds noise to processed pc."""
        self.noise_processor.raw_pc = self.thresholding_processor.processed_pc
        self.noise_processor.raw_intensities = self.thresholding_processor.processed_intensities
        self.noise_processor.hitmask = self.thresholding_processor.hitmask
        self.noise_processor.process()

    def apply_thresholding(self):
        """Apply thresholding algorithm."""
        if self.thresholding_processor.algorithm_name == "None":
            # No downsampling -> bypass waveform processor
            self.thresholding_processor.set_raw_data(
                    raw_pc=self.intensity_processor.processed_pc,
                    raw_intensities=self.intensity_processor.processed_intensities,
                    hitmask=self.intensity_processor.hitmask,
                    )
            self.thresholding_processor.process()
            return
        if self.waveform_model in (None, "None") or not self.thresholding_processor.waveform_based:
            # no waveform generated rely on raw pc
            self.thresholding_processor.set_raw_data(
                    raw_pc=self.waveform_processor.processed_pc,
                    raw_intensities=self.waveform_processor.processed_intensities,
                    hitmask=self.waveform_processor.hitmask,
                    )
        else:
            self.thresholding_processor.set_raw_data(
                    raw_pc=self.waveform_processor.processed_pc,  # rearranged into subrays
                    raw_waveforms=self.waveform_processor  # .processed_waveforms,
                    )
        self.thresholding_processor.process()

    def compute_colors(self):
        """Computes the colors."""
        self.color_processor.raw_intensities = self.noise_processor.processed_intensities
        self.color_processor.process()

    def compute_hitmask(self):
        """Computes the initial hitmask."""
        # for now there is only 1 possibility
        self.hitmask_processor.raw_pc = self.raw_pc
        self.hitmask_processor.process()
        self.hitmask = self.hitmask_processor.hitmask

    def compute_intensities(self):
        """Compute the intensities."""
        self.intensity_processor.set_raw_data(
                raw_pc=self.raw_pc,
                raw_tags=self.raw_tags,
                hitmask=self.hitmask,
                )
        self.intensity_processor.process()
        if self.global_parameters["ambiant_light"]:
            if self.intensity_processor.ambiant_light_enabled:
                # this will be used later when forming the waveform
                if self.processing_parameters["waveform_model"] == "None":
                    add = True
                else:
                    add = False  # added during waveform generation
                self.intensity_processor.compute_ambiant_light(
                        add_to_processed_intensities=add,
                        )
            else:
                self._logger.error("Ambiant light noise requested but not enabled of intensity processor [SKIPPED].")
        if "road_wetness_depth" in self.processing_parameters["intensity_model"]:
            road_wetness_depth = self.processing_parameters["intensity_model"]["road_wetness_depth"]
            if road_wetness_depth > 0.0:
                self.intensity_processor.add_road_wetness(
                        road_wetness_depth,
                        self.processing_parameters["intensity_model"]["road_thread_profile_depth"]
                        )
        if "saturate_retro_reflectors" in self.intensity_model_parameters:
            if self.intensity_model_parameters["saturate_retro_reflectors"]:
                self.intensity_processor.saturate_retroreflectors()

    def compute_waveform(self):
        """Compute the signal waveform."""
        self.waveform_processor.set_raw_data(
                raw_pc=self.raw_pc,
                raw_intensities=self.intensity_processor.processed_intensities,
                ambiant_light=self.intensity_processor.ambiant_light,
                hitmask=self.intensity_processor.hitmask,
                )
        self.waveform_processor.process()

    def load_raw_data(self):
        """Read raw data files."""
        pc = np.load(self.pc_path)["arr_0"]
        tags = np.load(self.tags_path)["arr_0"]
        self.raw_pc = carla2spherical(pc)
        self.raw_tags = tags
        # print('Data fraction set!!!!!!!')
        # data_fraction = 0.15
        # tot = self.raw_pc.shape[1]
        # start = int(np.floor(np.random.uniform(0, 1 - data_fraction) * tot))
        # end = start + int(np.floor(data_fraction * tot))
        # self.raw_pc = self.raw_pc[:, start:end, ...]

    def process(
            self, overwrite=False, write=True,
            compute_colors=True, compute_intensities=True,
            compute_hitmask=True,
            ):
        """Process full pipeline."""
        if self.raw_pc is None:
            self.load_raw_data()
        if compute_hitmask:
            self.compute_hitmask()
        if compute_intensities:
            self.compute_intensities()
        self.compute_waveform()
        self.apply_thresholding()
        self.add_noise()
        # assign processed final data
        self.processed_pc = self.noise_processor.processed_pc
        self.processed_intensities = self.noise_processor.processed_intensities
        self.processed_hitmask = self.thresholding_processor.hitmask  # final hitmask
        if compute_colors:
            self.compute_colors()
            self.processed_colors = self.color_processor.processed_colors
        if write:
            self.save_processed_data(overwrite=overwrite, save_colors=compute_colors)

    def rearrange_data_in_subrays(self, *args, **kwargs):
        """Rearrange given data into subrays."""
        return self.thresholding_processor.rearrange_data_in_subrays(
                *args, **kwargs)

    def save_processed_data(self, overwrite=False, save_colors=True):
        """Write final pt cloud, metadata file, colors and intensities."""
        os.makedirs(self.presetdir, exist_ok=True)
        self._logger.info(f"Saving processed data to: {self.presetdir}.")
        paths = [
                self.pc_out_path,
                self.intensities_out_path,
                self.metadatapath,
                self.colors_out_path,
                ]
        for path in paths:
            if os.path.isfile(path):
                if overwrite:
                    os.remove(path)
                else:
                    raise FileExistsError(path)
        np.save(self.pc_out_path, self.processed_pc)  # not compressed for faster reading
        if save_colors:
            np.save(self.colors_out_path, self.processed_colors)
        np.save(self.intensities_out_path, self.processed_intensities)
        with open(self.metadatapath, "w") as f:
            json.dump(self.processing_parameters, f, indent=4)
