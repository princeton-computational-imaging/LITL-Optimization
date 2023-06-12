"""Process viewport module."""
import concurrent.futures
import json
import os
import traceback

# from PyQt5.Qt import QApplication
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (
        QCheckBox, QComboBox, QGridLayout, QDialog, QLabel, QLineEdit,
        QMessageBox, QProgressBar, QPushButton, QSpinBox,
        )
from qtrangeslider import QRangeSlider

from .bases import BaseTaskManager
from .custom_widgets import (
        LineWidget, MathTextLabel,
        )
from .error_dialog_wrapper import error_dialog
from .main_app import kill_app
from .utils import LOADING_ICON, enable_lineeditor
from ..bases import BaseUtility
from ..process_pipelines.camera_processors import CameraProcessor
from ..process_pipelines.defaults import (
        GLOBAL_DEFAULTS,
        INTENSITY_MODELS_DEFAULTS,
        NOISE_MODELS_DEFAULTS,
        THRESHOLDING_ALGORITHMS_DEFAULTS,
        WAVEFORM_MODELS_DEFAULTS,
        )
from ..process_pipelines.pipeline import ProcessPipeline
from ..process_pipelines.intensity_models import (
        DISABLED_ROAD_WETNESS_MODELS,
        INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS,
        )
from ..process_pipelines.noise_models import (
        NoiseLessModelProcessor,
        NOISE_MODEL_2_NOISE_PROCESSOR_CLS,
        )
from ..process_pipelines.thresholding_algorithms import (
        THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS,
        WAVEFORM_BASED_THRESHOLDING_PROCESSORS,
        NON_WAVEFORM_BASED_THRESHOLDING_PROCESSORS,
        FindPeaksPointCloudProcessor,
        )
from ..process_pipelines.waveform_modelling import (
        NoWaveformModelProcessor,
        WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS,
        )
from ..settings import (
        METADATA_FILENAME,
        PRESET_SUBFOLDER_BASENAME,
        PROCESSED_DATA_FOLDER,
        )
from ..utils import (
        get_metadata, sanitize_preset_name,
        )


class ProcessProgressDialog(QDialog, BaseUtility):
    """Progress dialog for processor."""

    def __init__(self, parent, process_viewport, nframes, processing_params, **kwargs):
        """Prepocess custom progress dialog with 3 progress bars."""
        QDialog.__init__(self, parent)
        BaseUtility.__init__(self, **kwargs)
        self.processor = process_viewport
        self.processing_parameters = processing_params
        self.nframes = nframes
        self.max_total_bar_progress = 2 * nframes - 1
        self.build_gui()

    def build_gui(self):
        """Build the GUI."""
        # self.setWindowModality(Qt.WindowModal)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setModal(True)
        self.setWindowTitle("Processing imgs / point clouds... please wait")

        n_rows = 0
        self.desc_label = QLabel()
        params = self.processing_parameters
        text = "Processing settings:\n"
        if "saturation" in params:
            text += f" o saturation: {params['saturation']}\n"
        if "snr" in params:
            text += f" o snr: {params['snr']}\n"
        if "ambiant_light" in params:
            text += f" o ambiant light noise: {params['ambiant_light']}\n"
        text += f" o Thresholding method: {self.processing_parameters['thresholding_algorithm']['algorithm']}\n"
        text += f" o Intensity model: {self.processing_parameters['intensity_model']['model']}\n"
        if "noise_model" in params:
            text += f" o Noise model: {self.processing_parameters['noise_model']['model']}\n"
        if "waveform_model" in params:
            text += f" o Waveform model: {self.processing_parameters['waveform_model']['model']}\n"
            if params['waveform_model']['model'] != "None":
                text += (
                    f"    - range: {params['waveform_model']['waveform_range']}m\n"
                    f"    - resolution: {params['waveform_model']['waveform_resolution']}\n"
                    f"    - upsampling ratio: {params['waveform_model']['upsampling_ratio']}\n"
                    )
        self.desc_label.setText(text)
        self.layout.addWidget(self.desc_label, n_rows, 0, 1, 3)
        n_rows += 1

        self.total_label = QLabel("total:")
        self.layout.addWidget(self.total_label, n_rows, 0, 1, 1)

        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setRange(0, self.max_total_bar_progress)
        self.total_progress_bar.setValue(0)
        self.layout.addWidget(self.total_progress_bar, n_rows, 1, 1, 2)
        n_rows += 1

        self.cam_label = QLabel("camera imgs:")
        self.layout.addWidget(self.cam_label, n_rows, 0, 1, 1)

        self.cam_progress_bar = QProgressBar(self)
        self.cam_progress_bar.setRange(0, self.nframes - 1)
        self.cam_progress_bar.setValue(0)
        self.layout.addWidget(self.cam_progress_bar, n_rows, 1, 1, 2)
        n_rows += 1

        self.lidar_label = QLabel("LiDAR pcs:")
        self.layout.addWidget(self.lidar_label, n_rows, 0, 1, 1)
        self.lidar_progress_bar = QProgressBar(self)
        self.lidar_progress_bar.setRange(0, self.nframes - 1)
        self.lidar_progress_bar.setValue(0)
        self.layout.addWidget(self.lidar_progress_bar, n_rows, 1, 1, 2)
        n_rows += 1

        self.cancel_btn = QPushButton("Abort")
        self.cancel_btn.clicked.connect(self.cancel)
        self.layout.addWidget(self.cancel_btn, n_rows, 0, 1, 1)

        # process gif
        self.gif_label = QLabel()
        self.layout.addWidget(self.gif_label, 0, 3, -1, 1)
        self.gif_movie = QMovie(LOADING_ICON)
        self.gif_label.setMovie(self.gif_movie)
        self.gif_movie.start()

        # error box
        self.errorbox = None

    def cancel(self):
        """Cancel."""
        self.processor.task_manager.cancel()
        self.done(0)

    def close(self):
        """Close window."""
        self.done(0)

    def completely_done(self):
        """Everything was completely done."""
        self.processor.set_ready()
        self.processor.task_manager.shutdown()

    def error(self, exc):
        """An error occured, close everything..."""
        self.cancel()
        # raise an error msg
        if self.errorbox is None and exc != "cancel":
            self.errorbox = QMessageBox()
            self.errorbox.setIcon(QMessageBox.Critical)
            self.errorbox.setStandardButtons(QMessageBox.Close)
            self.errorbox.setText("An error occured while processing data...")
            self.errorbox.setWindowTitle("Processing error!")
            # taken from:
            # https://stackoverflow.com/a/35712784/6362595
            stack = ''.join(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__))
            self.errorbox.setInformativeText(stack)
            self.errorbox.buttonClicked.connect(self.close)
            self.errorbox.exec_()
            self.close()
            self.errorbox = None
            self.processor.done(0)
            self.processor.mainhub.close()
        else:
            return

    def increment_cam(self, error):
        """Increment camera progress bar."""
        if error == "cancel":
            self.close()
            return
        if error is not None:
            self.error(error)
            return
        self.increment(self.cam_progress_bar)
        self.increment(self.total_progress_bar)
        if self.total_progress_bar.value() == self.max_total_bar_progress:
            self.completely_done()

    def increment_lidar(self, error):
        """Increment lidar progress bar."""
        if error == "cancel":
            self.close()
            return
        if error is not None:
            self.error(error)
            return
        self.increment(self.lidar_progress_bar)
        self.increment(self.total_progress_bar)
        if self.total_progress_bar.value() == self.max_total_bar_progress:
            self.completely_done()

    def increment(self, bar):
        """Increment a progress bar."""
        val = bar.value()
        bar.setValue(val + 1)


class ProcessorTaskManager(BaseTaskManager):
    """Actual processor task manager."""
    finished_cam = pyqtSignal(object)
    finished_lidar = pyqtSignal(object)
    _executor_type = "process"

    @property
    def futures_cam(self):
        """The list of futures related to the cam signal."""
        return self.futures["finished_cam"]

    @property
    def futures_lidar(self):
        """The list of futures related to the lidar signal."""
        return self.futures["finished_lidar"]

    def submit_cam(self, fn, *args, **kwargs):
        """Submit a cam job."""
        self.submit(
                fn, *args, done_callback=self._internal_done_callback_cam,
                futures_name="finished_cam", **kwargs)

    def submit_lidar(self, fn, *args, **kwargs):
        """Submit a lidar job."""
        self.submit(
                fn, *args, done_callback=self._internal_done_callback_lidar,
                futures_name="finished_lidar", **kwargs)

    def _internal_done_callback_cam(self, future):
        self._internal_done_callback(future, self.finished_cam)

    def _internal_done_callback_lidar(self, future):
        self._internal_done_callback(future, self.finished_lidar)

    def _internal_done_callback(self, future, signal):
        try:
            exception = future.exception()
        except concurrent.futures.CancelledError:
            signal.emit("cancel")
            return
        if type(exception) is KeyboardInterrupt:
            signal.emit("cancel")
            return
        if exception:
            signal.emit(exception)
        else:
            signal.emit(future.result())


class ProcessViewPort(QDialog, BaseUtility):
    """Class that process lidar point cloud and camera images.

    This is to view them faster in the GUI.
    This Dialog shows all the options for processing.
    """
    available_thresholding_methods = tuple(
            THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS.keys())
    available_intensity_modes = tuple(
            INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS.keys())
    available_noise_models = tuple(
            NOISE_MODEL_2_NOISE_PROCESSOR_CLS.keys())

    def __init__(self, mainhub, **kwargs):
        """Processor init method."""
        QDialog.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        self.mainhub = mainhub  # main app

        # global attributes
        self.ready = False
        self.frame_dirs = self.mainhub.frame_dirs
        self.n_lasers_upsampled = 64 * 3
        self.build_gui()
        # call this once gui is completed in case some sections must be disabled
        self.on_thresholding_dropdown_menu_changed(self.thresholding_algorithm)
        self.on_waveform_model_dropdown_menu_changed(self.waveform_model)
        self.on_dsp_dropdown_menu_change(self.dsp_template)
        # progress bar / dialog
        self.progress_dialog = None
        self.task_manager = ProcessorTaskManager()

    def build_buttons_section(self, *args):
        """Build the buttons part of the GUI."""
        # SPANS 3 columns
        self.buttons_section_layout = QGridLayout()
        self.layout.addLayout(self.buttons_section_layout, *args)
        self.process_btn = QPushButton("Process")
        self.buttons_section_layout.addWidget(self.process_btn, 0, 0, 1, 1)
        self.process_btn.clicked.connect(self.process)

        self.cancel_btn = QPushButton("Cancel")
        self.buttons_section_layout.addWidget(self.cancel_btn, 0, 1, 1, 1)
        self.cancel_btn.clicked.connect(self.close)

    def build_frames_to_process_section(self, *args):
        """Builds the 'frames to process' GUI section."""
        # spans 3 columns
        self.frames_to_process_layout = QGridLayout()
        self.layout.addLayout(self.frames_to_process_layout, *args)

        self.max_process_frames_desc = QLabel()
        self.max_process_frames_desc.setText(
                "Frames to process:")
        self.frames_to_process_layout.addWidget(
                self.max_process_frames_desc, 0, 0, 1, 2)
        self.begin_frame_label = QLabel("0")
        self.begin_frame_label.setAlignment(Qt.AlignRight)
        self.frames_to_process_layout.addWidget(
                self.begin_frame_label, 0, 2, 1, 1)
        self.frames_to_process_range_slider = QRangeSlider(Qt.Horizontal)
        self.frames_to_process_range_slider.setMaximum(len(self.frame_dirs) - 1)
        self.frames_to_process_range_slider.setValue((0, len(self.frame_dirs) - 1))
        self.frames_to_process_range_slider.valueChanged.connect(self.on_frames_to_process_slider_value_changed)
        self.frames_to_process_layout.addWidget(
                self.frames_to_process_range_slider, 0, 3, 1, 6)
        self.end_frame_label = QLabel(str(len(self.frame_dirs) - 1))
        self.end_frame_label.setAlignment(Qt.AlignLeft)
        self.frames_to_process_layout.addWidget(self.end_frame_label, 0, 9, 1, 1)
        self.frames_to_process_section_line = LineWidget()
        self.frames_to_process_layout.addWidget(
                self.frames_to_process_section_line, 1, 0, 1, -1)

    def build_gui(self):
        """Builds the GUI."""
        # SETUP GUI
        # find defaults tau / range / res if previous run used them
        defaults = self.get_default_processing_parameters()

        self.setWindowTitle("Process data preset")
        self.layout = QGridLayout()  # main grid layout
        self.setLayout(self.layout)

        # BUILD GUI (done by sections)
        # build preset naming section
        self.build_preset_naming_section(defaults["preset_name"], 0, 0, 1, -1)
        # build frames to process section
        self.build_frames_to_process_section(1, 0, 1, -1)

        # modelling sections
        self.build_intensity_modelling_section(
                defaults["intensity_model"]["model"],
                3, 0, 1, 1,
                )
        self.build_wetness_model_parameters_section(
                defaults["intensity_model"]["road_wetness_depth"],
                defaults["intensity_model"]["road_thread_profile_depth"],
                4, 0, 1, 1,
                )
        self.build_waveform_model_parameters_section(
                defaults["waveform_model"], defaults["snr"],
                5, 0, 1, 1,
                )
        self.build_thresholding_modelling_section(
                defaults["thresholding_algorithm"],
                defaults["saturation"],
                3, 2, 3, 1)
        self.build_noise_modelling_section(
                defaults["noise_model"]["model"],
                defaults["noise_model"]["std"],
                6, 0, 1, 1)
        # add vline to separate sections (starts at thresholgin selection ends at road wetness)
        self.section_separator_vline = LineWidget("vertical")
        self.layout.addWidget(
                self.section_separator_vline, 3, 1, 4, 1)
        # overwriting options
        self.build_overwriting_section(7, 0, 1, -1)
        # buttons section
        self.build_buttons_section(8, 0, 1, -1)

    def build_intensity_modelling_section(self, default_intensity_mode, *args):
        """Builds the intensity modelling section."""
        self.intensity_modelling_section_layout = QGridLayout()
        self.layout.addLayout(self.intensity_modelling_section_layout, *args)
        self.intensity_mode_label = QLabel()
        self.intensity_mode_label.setText("<b>Select Intensity computation mode</b>:")
        self.intensity_modelling_section_layout.addWidget(
                self.intensity_mode_label, 0, 0, 1, -1)
        self.intensity_mode_dropdown_menu = QComboBox()
        self.intensity_mode_dropdown_menu.addItems(self.available_intensity_modes)
        self.intensity_mode_dropdown_menu.setCurrentText(default_intensity_mode)
        self.intensity_modelling_section_layout.addWidget(
                self.intensity_mode_dropdown_menu, 1, 0, 1, -1)
        self.intensity_mode_dropdown_menu.currentTextChanged.connect(self.on_intensity_mode_dropdown_menu_changed)

        self.add_ambiant_light_noise_chkbox = QCheckBox("Add ambiant light noise")
        self.intensity_modelling_section_layout.addWidget(
                self.add_ambiant_light_noise_chkbox, 2, 0, 1, -1)
        self.saturate_retroreflectors_chkbox = QCheckBox("Saturate retro reflectors")
        self.intensity_modelling_section_layout.addWidget(
                self.saturate_retroreflectors_chkbox, 3, 0, 1, -1)

    def build_noise_modelling_section(self, default_noise, default_noise_std, *args):
        """Builds the noise modelling section."""
        self.noise_modelling_section_layout = QGridLayout()
        self.layout.addLayout(self.noise_modelling_section_layout, *args)
        self.noise_section_line = LineWidget()
        self.noise_modelling_section_layout.addWidget(self.noise_section_line, 0, 0, 1, -1)
        self.noise_model_selection_label = QLabel("<b>Select Noise model</b>:")
        self.noise_modelling_section_layout.addWidget(
                self.noise_model_selection_label, 1, 0, 1, -1)
        self.noise_model_dropdown_menu = QComboBox()
        self.noise_model_dropdown_menu.currentTextChanged.connect(
                self.on_noise_model_dropdown_menu_changed)
        self.noise_modelling_section_layout.addWidget(
                self.noise_model_dropdown_menu, 2, 0, 1, -1)
        self.noise_std_label = QLabel("std (mm): ")
        self.noise_modelling_section_layout.addWidget(
                self.noise_std_label, 3, 0, 1, 1)
        self.noise_model_std_lineedit = QLineEdit()
        self.noise_model_std_lineedit.setText(str(default_noise_std))
        self.noise_modelling_section_layout.addWidget(
                self.noise_model_std_lineedit, 3, 1, 1, 1)
        self.noise_model_dropdown_menu.addItems(self.available_noise_models)

    def build_overwriting_section(self, *args):
        """Builds the overwriting section."""
        # spans 3 columns
        self.overwriting_section_layout = QGridLayout()
        self.layout.addLayout(self.overwriting_section_layout, *args)
        self.ending_section_line = LineWidget()
        self.overwriting_section_layout.addWidget(
                self.ending_section_line, 0, 0, 1, -1)
        self.overwrite_lidar_chkbox = QCheckBox("overwrite LiDAR")
        self.overwrite_lidar_chkbox.setChecked(False)
        self.overwriting_section_layout.addWidget(
                self.overwrite_lidar_chkbox, 1, 0, 1, 1)

        self.overwrite_cam_chkbox = QCheckBox("overwrite Camera")
        self.overwrite_cam_chkbox.setChecked(False)
        self.overwriting_section_layout.addWidget(
                self.overwrite_cam_chkbox, 1, 2, 1, 1)

    def build_preset_naming_section(self, default_preset_name, *args):
        """Build the preset naming section."""
        # spans 3 columns
        self.preset_naming_layout = QGridLayout()
        self.layout.addLayout(self.preset_naming_layout, *args)
        self.preset_label = QLabel()
        self.preset_label.setText("Preset name: ")
        self.preset_naming_layout.addWidget(self.preset_label, 0, 0, 1, 1)
        self.preset_lineeditor = QLineEdit()
        self.preset_lineeditor.setText(default_preset_name)
        self.preset_lineeditor.setAlignment(Qt.AlignRight)
        self.preset_naming_layout.addWidget(self.preset_lineeditor, 0, 1, 1, 1)

    def build_thresholding_modelling_section(
            self, defaults_dict, default_saturation, *args,
            ):
        """Build the thresholding modelling section."""
        self.thresholding_section_layout = QGridLayout()
        self.layout.addLayout(self.thresholding_section_layout, *args)
        self.description_label = QLabel()
        self.description_label.setText("<b>Select thresholding method</b>:")
        self.thresholding_section_layout.addWidget(
                self.description_label, 0, 0, 1, -1)

        self.thresh_dropdown_menu = QComboBox()
        self.thresh_dropdown_menu.addItems(self.available_thresholding_methods)
        self.thresh_dropdown_menu.setCurrentText(defaults_dict["algorithm"])
        self.thresh_dropdown_menu.currentTextChanged.connect(self.on_thresholding_dropdown_menu_changed)
        self.thresholding_section_layout.addWidget(
                self.thresh_dropdown_menu, 1, 0, 1, -1)

        self.thresholding_noise_floor_label = QLabel("Noise floor threshold:")
        self.thresholding_section_layout.addWidget(
                self.thresholding_noise_floor_label, 2, 0, 1, 1)
        self.thresholding_noise_floor_lineeditor = QLineEdit()
        self.thresholding_noise_floor_lineeditor.setText(str(
            defaults_dict["noise_floor_threshold"]))
        self.thresholding_section_layout.addWidget(
                self.thresholding_noise_floor_lineeditor, 2, 1, 1, -1)

        # power gain
        self.dsp_gain_label = QLabel("Gain [dB]:")
        self.dsp_gain_lineeditor = QLineEdit()
        self.dsp_gain_lineeditor.setText(
                str(defaults_dict["gain"]))
        self.thresholding_section_layout.addWidget(
                self.dsp_gain_label, 3, 0, 1, 1)
        self.thresholding_section_layout.addWidget(self.dsp_gain_lineeditor, 3, 1, 1, -1)

        # saturation level
        self.saturation_label = QLabel("Saturation")
        self.thresholding_section_layout.addWidget(self.saturation_label, 4, 0, 1, 1)
        self.saturation_lineeditor = QLineEdit()
        self.saturation_lineeditor.setText(str(default_saturation))
        self.thresholding_section_layout.addWidget(self.saturation_lineeditor, 4, 1, 1, -1)

        # gaussian denoizer sigma
        self.gaussian_denoizer_sigma_label = QLabel("Gaussian denoizer sigma")
        self.thresholding_section_layout.addWidget(self.gaussian_denoizer_sigma_label, 5, 0, 1, 1)
        self.gaussian_denoizer_sigma_lineeditor = QLineEdit()
        self.gaussian_denoizer_sigma_lineeditor.setText(str(defaults_dict["gaussian_denoizer_sigma"]))
        self.thresholding_section_layout.addWidget(self.gaussian_denoizer_sigma_lineeditor, 5, 1, 1, -1)

        # digitization
        self.digitization_label = QLabel("Digitization")
        self.thresholding_section_layout.addWidget(self.digitization_label, 6, 0, 1, 1)
        self.digitization_dropdown_menu = QComboBox()
        self.digitization_dropdown_menu.addItems(['float', "uint8", "uint16"])
        self.digitization_dropdown_menu.setCurrentText(defaults_dict["digitization"])
        self.thresholding_section_layout.addWidget(self.digitization_dropdown_menu, 6, 1, 1, -1)

        # DSP Template
        self.dsp_label = QLabel("DSP Template")
        self.thresholding_section_layout.addWidget(self.dsp_label, 7, 0, 1, 1)
        self.dsp_dropdown_menu = QComboBox()
        self.dsp_dropdown_menu.addItems(["GaussTemplate", "CosTemplate"])
        self.dsp_dropdown_menu.setCurrentText(defaults_dict["dsp_template"])
        self.dsp_dropdown_menu.currentTextChanged.connect(self.on_dsp_dropdown_menu_change)
        self.thresholding_section_layout.addWidget(self.dsp_dropdown_menu, 7, 1, 1, -1)

        self.correction_factor_label = QLabel("Correction factor")
        self.thresholding_section_layout.addWidget(self.correction_factor_label, 8, 0, 1, 1)
        self.correction_factor_lineeditor = QLineEdit()
        self.correction_factor_lineeditor.setText(str(defaults_dict["correction_factor"]))
        self.thresholding_section_layout.addWidget(self.correction_factor_lineeditor, 8, 1, 1, -1)

        # find peaks method
        # height
        self.height_label = QLabel("Min. Peak Height:")
        sec = self.thresholding_section_layout
        sec.addWidget(self.height_label, 9, 0, 1, 1)
        self.height_label_lineeditor = QLineEdit()
        self.height_label_lineeditor.setText(str(defaults_dict["height"]))
        sec.addWidget(self.height_label_lineeditor, 9, 1, 1, -1)
        # threshold
        self.min_threshold_label = QLabel("Min. Threshold:")
        sec.addWidget(self.min_threshold_label, 10, 0, 1, 1)
        self.min_threshold_lineeditor = QLineEdit()
        self.min_threshold_lineeditor.setText(str(defaults_dict["min_threshold"]))
        sec.addWidget(self.min_threshold_lineeditor, 10, 1, 1, -1)
        self.max_threshold_label = QLabel("Max. Threshold:")
        sec.addWidget(self.max_threshold_label, 11, 0, 1, 1)
        self.max_threshold_lineeditor = QLineEdit()
        self.max_threshold_lineeditor.setText(str(defaults_dict["max_threshold"]))
        sec.addWidget(self.max_threshold_lineeditor, 11, 1, 1, -1)
        # peak dist
        self.peak_dist_label = QLabel("Peak min. dist:")
        sec.addWidget(self.peak_dist_label, 12, 0, 1, 1)
        self.peak_dist_lineeditor = QLineEdit()
        self.peak_dist_lineeditor.setText(str(defaults_dict["distance"]))
        sec.addWidget(self.peak_dist_lineeditor, 12, 1, 1, -1)
        # prominence
        self.min_prominence_label = QLabel("Peak min. prominence:")
        sec.addWidget(self.min_prominence_label, 13, 0, 1, 1)
        self.min_prominence_lineeditor = QLineEdit()
        self.min_prominence_lineeditor.setText(str(defaults_dict["min_prominence"]))
        sec.addWidget(self.min_prominence_lineeditor, 13, 1, 1, -1)
        self.max_prominence_label = QLabel("Peak max. prominence:")
        sec.addWidget(self.max_prominence_label, 14, 0, 1, 1)
        self.max_prominence_lineeditor = QLineEdit()
        self.max_prominence_lineeditor.setText(str(defaults_dict["max_prominence"]))
        sec.addWidget(self.max_prominence_lineeditor, 14, 1, 1, -1)
        # peak width
        self.min_width_label = QLabel("Peak min. width:")
        sec.addWidget(self.min_width_label, 15, 0, 1, 1)
        self.min_width_lineeditor = QLineEdit()
        self.min_width_lineeditor.setText(str(defaults_dict["min_width"]))
        sec.addWidget(self.min_width_lineeditor, 15, 1, 1, -1)
        self.max_width_label = QLabel("Peak max. width:")
        sec.addWidget(self.max_width_label, 16, 0, 1, 1)
        self.max_width_lineeditor = QLineEdit()
        self.max_width_lineeditor.setText(str(defaults_dict["max_width"]))
        sec.addWidget(self.max_width_lineeditor, 16, 1, 1, -1)
        # wlen
        self.wlen_label = QLabel("Wlen:")
        sec.addWidget(self.wlen_label, 17, 0, 1, 1)
        self.wlen_lineeditor = QLineEdit()
        self.wlen_lineeditor.setText(str(defaults_dict["wlen"]))
        sec.addWidget(self.wlen_lineeditor, 17, 1, 1, -1)
        # rel_height
        self.rel_height_label = QLabel("Rel_height:")
        sec.addWidget(self.rel_height_label, 18, 0, 1, 1)
        self.rel_height_lineeditor = QLineEdit()
        self.rel_height_lineeditor.setText(str(defaults_dict["rel_height"]))
        sec.addWidget(self.rel_height_lineeditor, 18, 1, 1, -1)
        # plateau size
        self.min_plateau_size_label = QLabel("Min plateau size:")
        sec.addWidget(self.min_plateau_size_label, 19, 0, 1, 1)
        self.min_plateau_size_lineeditor = QLineEdit()
        self.min_plateau_size_lineeditor.setText(str(defaults_dict["min_plateau_size"]))
        sec.addWidget(self.min_plateau_size_lineeditor, 19, 1, 1, -1)
        self.max_plateau_size_label = QLabel("Max plateau size:")
        sec.addWidget(self.max_plateau_size_label, 20, 0, 1, 1)
        self.max_plateau_size_lineeditor = QLineEdit()
        self.max_plateau_size_lineeditor.setText(str(defaults_dict["max_plateau_size"]))
        sec.addWidget(self.max_plateau_size_lineeditor, 20, 1, 1, -1)

    def build_waveform_model_parameters_section(self, defaults_dict, default_snr, *args):
        """Builds the waveform model parameter section."""
        self.waveform_model_section_layout = QGridLayout()
        self.layout.addLayout(self.waveform_model_section_layout, *args)
        self.intensity_section_line = LineWidget()
        self.waveform_model_section_layout.addWidget(
                self.intensity_section_line, 0, 0, 1, -1)
        self.waveform_model_section_title = QLabel("<b>Waveform model</b>:")
        self.waveform_model_section_layout.addWidget(
                self.waveform_model_section_title, 1, 0, 1, 1)
        self.waveform_model_dropdown_menu = QComboBox()
        self.waveform_model_dropdown_menu.addItems(
                tuple(WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS.keys()))
        self.waveform_model_dropdown_menu.setCurrentText(defaults_dict["model"])
        self.waveform_model_dropdown_menu.currentTextChanged.connect(
                self.on_waveform_model_dropdown_menu_changed)
        self.waveform_model_section_layout.addWidget(
                self.waveform_model_dropdown_menu, 2, 0, 1, -1)
        # parameter section
        self.tauH_label = MathTextLabel(r"$\tau_H$ [ns]: ")
        self.waveform_model_section_layout.addWidget(
                self.tauH_label, 3, 0, 1, 1)
        self.tauH_lineeditor = QLineEdit()
        self.tauH_lineeditor.setText(str(defaults_dict["tauH"]))
        self.waveform_model_section_layout.addWidget(
                self.tauH_lineeditor, 3, 1, 1, 1)

        self.waveform_min_dist_label = QLabel("Waveform min dist [m]: ")
        self.waveform_model_section_layout.addWidget(
                self.waveform_min_dist_label, 4, 0, 1, 1)
        self.waveform_min_dist_lineeditor = QLineEdit()
        self.waveform_min_dist_lineeditor.setText(str(defaults_dict["waveform_min_dist"]))
        self.waveform_model_section_layout.addWidget(
                self.waveform_min_dist_lineeditor, 4, 1, 1, 1)

        self.waveform_range_label = QLabel()
        self.waveform_range_label.setText("Waveform range [m]")
        self.waveform_model_section_layout.addWidget(
                self.waveform_range_label, 5, 0, 1, 1)
        self.waveform_range_lineeditor = QLineEdit()
        self.waveform_range_lineeditor.setText(str(defaults_dict["waveform_range"]))
        self.waveform_model_section_layout.addWidget(
                self.waveform_range_lineeditor, 5, 1, 1, 1)

        self.waveform_resolution_label = QLabel()
        self.waveform_resolution_label.setText("Waveform resolution (npts): ")
        self.waveform_model_section_layout.addWidget(
                self.waveform_resolution_label, 6, 0, 1, 1)
        self.waveform_resolution_lineeditor = QLineEdit()
        self.waveform_resolution_lineeditor.setText(str(defaults_dict["waveform_resolution"]))
        self.waveform_model_section_layout.addWidget(
                self.waveform_resolution_lineeditor, 6, 1, 1, 1)

        self.poissonize_signal_chkbox = QCheckBox("Poissonize Signal:")
        self.waveform_model_section_layout.addWidget(
                self.poissonize_signal_chkbox, 7, 0, 1, 1)

        self.snr_label = QLabel("SNR:")
        self.waveform_model_section_layout.addWidget(self.snr_label, 8, 0, 1, 1)
        self.snr_lineeditor = QLineEdit()
        self.snr_lineeditor.setText(str(default_snr))
        self.waveform_model_section_layout.addWidget(self.snr_lineeditor, 8, 1, 1, 1)

        self.upsampling_ratio_label = QLabel("Upsampling ratio:")
        self.waveform_model_section_layout.addWidget(self.upsampling_ratio_label, 9, 0, 1, 1)
        self.upsampling_ratio_spinbox = QSpinBox()
        self.upsampling_ratio_spinbox.setMinimum(1)
        self.upsampling_ratio_spinbox.setSingleStep(2)
        self.upsampling_ratio_spinbox.setValue(defaults_dict["upsampling_ratio"])
        self.waveform_model_section_layout.addWidget(
                self.upsampling_ratio_spinbox, 9, 1, 1, 1)

    def build_wetness_model_parameters_section(
            self, default_road_wetness_depth, default_road_thread_profile_depth, *args):
        """Builds the road wetness model parameters section."""
        # road wetness options
        self.road_wetness_section_layout = QGridLayout()
        self.layout.addLayout(self.road_wetness_section_layout, *args)

        self.road_wetness_checkbox = QCheckBox("Add road wetness")
        self.road_wetness_checkbox.clicked.connect(self.on_road_wetness_checkbox_state_changed)
        self.road_wetness_section_layout.addWidget(self.road_wetness_checkbox, 0, 0, 1, -1)

        self.road_wetness_depth_label = QLabel()
        self.road_wetness_depth_label.setText("Road wetness depth (mm): ")
        self.road_wetness_section_layout.addWidget(
                self.road_wetness_depth_label, 1, 0, 1, 1)
        self.road_wetness_depth_lineeditor = QLineEdit()
        self.road_wetness_depth_lineeditor.setText(str(default_road_wetness_depth))
        self.road_wetness_section_layout.addWidget(
                self.road_wetness_depth_lineeditor, 1, 1, 1, -1)

        self.road_thread_profile_depth_label = QLabel()
        self.road_thread_profile_depth_label.setText("Road thread profile depth (mm): ")
        self.road_wetness_section_layout.addWidget(
                self.road_thread_profile_depth_label, 2, 0, 1, 1)
        self.road_thread_profile_depth_lineeditor = QLineEdit()
        self.road_thread_profile_depth_lineeditor.setText(str(default_road_thread_profile_depth))
        self.road_wetness_section_layout.addWidget(
                self.road_thread_profile_depth_lineeditor, 2, 1, 1, -1)

        # disable the above by default
        enable_lineeditor(self.road_wetness_depth_lineeditor, False)
        enable_lineeditor(self.road_thread_profile_depth_lineeditor, False)

    @property
    def add_ambiant_light_noise(self):
        """True if we want to add ambiant light noise."""
        return self.add_ambiant_light_noise_chkbox.isChecked()

    @property
    def begin_frame(self):
        """The first frame to process."""
        return self.frames_to_process_range_slider.value()[0]

    @property
    def current_frame(self):
        """The current frame number."""
        return self.mainhub.current_frame

    @current_frame.setter
    def current_frame(self, frame):
        self.mainhub.current_frame = frame

    @property
    def end_frame(self):
        """The last frame to process."""
        return self.frames_to_process_range_slider.value()[1]

    @property
    def gain(self):
        """The power gain."""
        return float(self.dsp_gain_lineeditor.text())

    @property
    def digitization(self):
        """The number of digitization bits."""
        return self.digitization_dropdown_menu.currentText()

    @property
    def dsp_template(self):
        """The DSP model used."""
        return self.dsp_dropdown_menu.currentText()

    @property
    def gaussian_denoizer_sigma(self):
        """The gaussian denoizer sigma."""
        return float(self.gaussian_denoizer_sigma_lineeditor.text())

    @property
    def height(self):
        """The peak minimal height."""
        return float(self.height_label_lineeditor.text())

    @property
    def min_threshold(self):
        """The peak min threshold."""
        return float(self.min_threshold_lineeditor.text())

    @property
    def max_threshold(self):
        """The peaks max threshold."""
        return float(self.max_threshold_lineeditor.text())

    @property
    def distance(self):
        """Min distance between peaks."""
        return int(self.peak_dist_lineeditor.text())

    @property
    def min_prominence(self):
        """Min peak prominence."""
        return float(self.min_prominence_lineeditor.text())

    @property
    def max_prominence(self):
        """Max peak prominence."""
        return float(self.max_prominence_lineeditor.text())

    @property
    def min_width(self):
        """Min peak width."""
        return int(self.min_width_lineeditor.text())

    @property
    def max_width(self):
        """Max peak width."""
        return int(self.max_width_lineeditor.text())

    @property
    def wlen(self):
        """Peak wlen."""
        return int(self.wlen_lineeditor.text())

    @property
    def rel_height(self):
        """Peak rel height."""
        return float(self.rel_height_lineeditor.text())

    @property
    def min_plateau_size(self):
        """Min plateau size."""
        return int(self.min_plateau_size_lineeditor.text())

    @property
    def max_plateau_size(self):
        """Max plateau size."""
        return int(self.max_plateau_size_lineeditor.text())

    @property
    def thresholding_algorithm(self):
        """The selected thresholding method."""
        return self.thresh_dropdown_menu.currentText()

    @property
    def intensity_model(self):
        """The selected intensity computation mode."""
        return self.intensity_mode_dropdown_menu.currentText()

    @property
    def noise_model(self):
        """The selected noise model."""
        return self.noise_model_dropdown_menu.currentText()

    @property
    def noise_model_std(self):
        """The noise model std."""
        return float(self.noise_model_std_lineedit.text())

    @property
    def poissonize_signal(self):
        """True if we poissonize signal."""
        return self.poissonize_signal_chkbox.isChecked()

    @property
    def preset(self):
        """Alias for preset name."""
        return self.preset_name

    @property
    def preset_name(self):
        """The preset name."""
        name = self.preset_lineeditor.text()
        return sanitize_preset_name(name)

    @property
    def road_thread_profile_depth(self):
        """The road thread profile depth in mm."""
        return float(self.road_thread_profile_depth_lineeditor.text())

    @property
    def road_wetness_depth(self):
        """The road wetness depth in mm."""
        if not self.road_wetness_checkbox.isChecked():
            return 0.0
        return float(self.road_wetness_depth_lineeditor.text())

    @property
    def saturation(self):
        """The saturation threshold."""
        return float(self.saturation_lineeditor.text())

    @property
    def saturate_retro_reflectors(self):
        """True if we saturate the retro reflectors."""
        return self.saturate_retroreflectors_chkbox.isChecked()

    @property
    def snr(self):
        """The SNR."""
        return float(self.snr_lineeditor.text())

    @property
    def tauH(self):
        """The tauH for laser beam simulation."""
        return float(self.tauH_lineeditor.text())

    @property
    def thresholding_noise_floor(self):
        """The thresholding noise floor."""
        return float(self.thresholding_noise_floor_lineeditor.text())

    @property
    def upsampling_ratio(self):
        """The upsampling ratio."""
        return self.upsampling_ratio_spinbox.value()

    @property
    def waveform_min_distance(self):
        """The minimal distance for the waveform."""
        return float(self.waveform_min_dist_lineeditor.text())

    @property
    def waveform_resolution(self):
        """The number of pts in the waveform."""
        return int(self.waveform_resolution_lineeditor.text())

    @property
    def waveform_range(self):
        """The maximal waveform range in m."""
        return float(self.waveform_range_lineeditor.text())

    @property
    def waveform_model(self):
        """The selected waveform model."""
        return self.waveform_model_dropdown_menu.currentText()

    def enable_signal_processing_lineeditors(self, enable):
        """Enables or disables the noise floor threshold lineeditor."""
        for lineeditor in [self.thresholding_noise_floor_lineeditor,
                           # self.saturation_lineeditor,
                           self.dsp_gain_lineeditor, self.gaussian_denoizer_sigma_lineeditor,
                           ]:
            enable_lineeditor(lineeditor, enable=enable)

    def enable_find_peaks_lineeditors(self, enable):
        """Enables find peaks lineeditors."""
        for lineeditor in [
                self.height_label_lineeditor,
                self.min_threshold_lineeditor,
                self.max_threshold_lineeditor,
                self.rel_height_lineeditor,
                self.min_prominence_lineeditor,
                self.max_prominence_lineeditor,
                self.peak_dist_lineeditor,
                self.min_width_lineeditor,
                self.max_width_lineeditor,
                self.wlen_lineeditor,
                self.min_plateau_size_lineeditor,
                self.max_plateau_size_lineeditor,
                ]:
            enable_lineeditor(lineeditor, enable=enable)

    def get_processing_parameters(self):
        """Return the selected processing parameters."""
        return {"preset_name": self.preset_name,
                "snr": self.snr,
                "ambiant_light": self.add_ambiant_light_noise,
                "saturation": self.saturation,
                "thresholding_algorithm": {
                    "algorithm": self.thresholding_algorithm,
                    "gain": self.gain,
                    "digitization": self.digitization,
                    "dsp_template": self.dsp_template,
                    "noise_floor_threshold": self.thresholding_noise_floor,
                    "gaussian_denoizer_sigma": self.gaussian_denoizer_sigma,
                    "height": self.height,
                    "min_threshold": self.min_threshold,
                    "max_threshold": self.max_threshold,
                    "distance": self.distance,
                    "min_prominence": self.min_prominence,
                    "max_prominence": self.max_prominence,
                    "min_width": self.min_width,
                    "max_width": self.max_width,
                    "wlen": self.wlen,
                    "rel_height": self.rel_height,
                    "min_plateau_size": self.min_plateau_size,
                    "max_plateau_size": self.max_plateau_size,
                    },
                "waveform_model": {
                    "model": self.waveform_model,
                    "poissonize_signal": self.poissonize_signal,
                    "tauH": self.tauH,
                    "upsampling_ratio": self.upsampling_ratio,
                    "waveform_min_dist": self.waveform_min_distance,
                    "waveform_range": self.waveform_range,
                    "waveform_resolution": self.waveform_resolution,
                    },
                "noise_model": {
                    "model": self.noise_model,
                    "std": self.noise_model_std,
                    },
                "intensity_model": {
                    "model": self.intensity_model,
                    "road_wetness_depth": self.road_wetness_depth,
                    "road_thread_profile_depth": self.road_thread_profile_depth,
                    "saturate_retro_reflectors": self.saturate_retro_reflectors,
                    },
                }

    def on_frames_to_process_slider_value_changed(self, values):
        """Method called when moving the frames selected slider."""
        self.begin_frame_label.setText(str(values[0]))
        self.end_frame_label.setText(str(values[1]))

    def on_noise_model_dropdown_menu_changed(self, noise_model):
        """Method called when the noise model selection changed."""
        if noise_model == NoiseLessModelProcessor.model_name:
            # disable all noise model parameters
            enable = False
        else:
            enable = True
        enable_lineeditor(self.noise_model_std_lineedit, enable=enable)

    def on_road_wetness_checkbox_state_changed(self, enable):
        """Method called when the road_wetness_checkbox is clicked."""
        enable_lineeditor(self.road_wetness_depth_lineeditor, enable=enable)
        enable_lineeditor(self.road_thread_profile_depth_lineeditor, enable=enable)

    def on_dsp_dropdown_menu_change(self, dsp):
        """Method called when we change the dsp model."""
        if dsp == "CosTemplate":
            enable = True
        else:
            enable = False
        enable_lineeditor(self.correction_factor_lineeditor, enable=enable)

    def on_thresholding_dropdown_menu_changed(self, method):
        """Method called when thresholding dropdown menu changed."""
        if method == "":
            return  # to avoid infinite recursion when clearing this menu
        enable_find_peaks = False
        if method in WAVEFORM_BASED_THRESHOLDING_PROCESSORS:
            if self.waveform_model == NoWaveformModelProcessor.model_name:
                # change waveform model
                self.waveform_model_dropdown_menu.setCurrentText(
                        tuple(WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS.keys())[1],
                        )
            if THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS[
                    method].has_signal_processing:
                enable_signal_processing = True
            else:
                enable_signal_processing = False
            if method == FindPeaksPointCloudProcessor.algorithm_name:
                enable_find_peaks = True
        else:
            if self.waveform_model != NoWaveformModelProcessor.model_name:
                # change waveform model
                self.waveform_model_dropdown_menu.setCurrentText(
                        NoWaveformModelProcessor.model_name)
            enable_signal_processing = False
        self.enable_signal_processing_lineeditors(enable_signal_processing)
        self.enable_find_peaks_lineeditors(enable_find_peaks)

    def on_waveform_model_dropdown_menu_changed(self, model):
        """Method called when waveform model dropdown menu changed."""
        if model != NoWaveformModelProcessor.model_name:
            enable = True
            options = tuple(WAVEFORM_BASED_THRESHOLDING_PROCESSORS.keys())
        else:
            enable = False
            options = tuple(NON_WAVEFORM_BASED_THRESHOLDING_PROCESSORS.keys())
        for lineedit in [self.tauH_lineeditor, self.waveform_range_lineeditor,
                         self.waveform_resolution_lineeditor,
                         self.waveform_min_dist_lineeditor,
                         ]:
            enable_lineeditor(lineedit, enable=enable)
        # filter out thresholding algorithm
        self.thresh_dropdown_menu.clear()
        self.thresh_dropdown_menu.addItems(options)
        if self.thresholding_algorithm not in options:
            self.thresh_dropdown_menu.setCurrentText(options[0])

    def on_intensity_mode_dropdown_menu_changed(self, mode):
        """Method called when intensity mode selection changed."""
        if mode in DISABLED_ROAD_WETNESS_MODELS:
            # disable wetness modelling
            enable = False
        else:
            enable = True
        if self.road_wetness_checkbox.isChecked() and not enable:
            self.road_wetness_checkbox.setChecked(False)
        self.road_wetness_checkbox.setEnabled(enable)

    @staticmethod
    def process_camera_image(frame_dir, overwrite, loglevel):
        """Load a pickled numpy array and save it as an image.

        Remove output img if already existing if needed.
        """
        processor = CameraProcessor(framedir=frame_dir, loglevel=loglevel)
        processor.process(overwrite)

    @staticmethod
    def process_frame(
            framedir, processing_parameters, overwrite,
            iframe, loglevel,
            ):
        """Process a single frame for point cloud.

        Args:
            framedir: str
                The path of the framedir containing raw data.
            overwrite: bool
                If True, overwrite data.
            iframe: int
                The frame number to process.
            processing_parameters: dict
                The processing parameters.
            loglevel: int
                The logging level for processor.
        """
        pipeline = ProcessPipeline(
                processing_parameters, framedir=framedir, loglevel=loglevel,
                wait_for_images=True,
                )
        if not pipeline.need_recompute and not overwrite:
            pipeline._logger.info(f"frame {iframe} already done and don't overwrite -> SKIP")
            return
        pipeline.process(overwrite)

    @error_dialog(
            text="An error occured while processing data...",
            callback=kill_app)
    def process(self, *args, processing_parameters=None, force_overwrite_lidar=False, **kwargs):
        """Process data based on selected thresholding method."""
        self.ready = False
        nframes = self.end_frame - self.begin_frame
        self._logger.info(f"Will process {nframes} frames: {self.begin_frame} - {self.end_frame}.")
        if processing_parameters is None:
            processing_parameters = self.get_processing_parameters()
        self._logger.info(f"The processing parameters are: {processing_parameters}")

        # create progress dialog
        if self.progress_dialog is not None:
            self.progress_dialog.done(0)
            del self.progress_dialog
            self.progress_dialog = None
        parent = self
        if not self.isVisible():
            parent = self.mainhub
        self.progress_dialog = ProcessProgressDialog(
                parent, self, self.end_frame - self.begin_frame + 1, processing_parameters)
        self.progress_dialog.show()
        self.task_manager.reset(nworkers=nframes)  # the final amount of workers is clipped to cpu counts anyway
        self.task_manager.finished_cam.connect(self.progress_dialog.increment_cam)
        self.task_manager.finished_lidar.connect(self.progress_dialog.increment_lidar)
        for framedir in self.frame_dirs[self.begin_frame:self.end_frame + 1]:
            self.task_manager.submit_cam(
                    self.process_camera_image, framedir,
                    self.overwrite_cam_chkbox.isChecked(),
                    self._logger.level,
                    )
        # wait for cam tasks to finish
        # self.task_manager.join()
        for iframe, framedir in enumerate(
                self.frame_dirs[self.begin_frame:self.end_frame + 1],
                start=self.begin_frame):
            self.task_manager.submit_lidar(
                    self.process_frame, framedir,
                    processing_parameters,
                    self.overwrite_lidar_chkbox.isChecked() or force_overwrite_lidar,
                    iframe,
                    self._logger.level,
                    )

    def set_ready(self):
        """Set processor ready and main hub as well."""
        self.ready = True
        self.progress_dialog.done(0)
        self.mainhub.set_ready()
        self.close()

    def get_default_processing_parameters(self):
        """Return the default processing parameters dict."""
        if not os.path.isdir(os.path.join(self.mainhub.root_dir, "frame0")):
            raise FileNotFoundError("No frames found...")
        # actual default
        defaults = {
                "waveform_model": WAVEFORM_MODELS_DEFAULTS.copy(),
                "thresholding_algorithm": THRESHOLDING_ALGORITHMS_DEFAULTS.copy(),
                "intensity_model": INTENSITY_MODELS_DEFAULTS.copy(),
                "noise_model": NOISE_MODELS_DEFAULTS.copy(),
                "preset_name": "new preset name",
                }
        defaults.update(GLOBAL_DEFAULTS)
        # find if there are any presets already
        processed_dir = os.path.join("frame0", PROCESSED_DATA_FOLDER)
        if not os.path.isdir(processed_dir):
            return defaults
        for filename in os.listdir(os.path.join("frame0", PROCESSED_DATA_FOLDER)):
            fp = os.path.join("frame0", PROCESSED_DATA_FOLDER, filename)
            if os.path.isdir(fp) and filename.startswith(PRESET_SUBFOLDER_BASENAME):
                defaults["preset_name"] = sanitize_preset_name(filename, reverse=True)
                break
        else:
            # no presets exists
            return defaults
        framedir = self.mainhub.frame_dirs[0]
        dirname = os.path.join(framedir, PROCESSED_DATA_FOLDER, sanitize_preset_name(defaults["preset_name"]))
        if METADATA_FILENAME not in os.listdir(dirname):
            return defaults
        metapath = os.path.join(dirname, METADATA_FILENAME)
        try:
            data = get_metadata(framedir, defaults["preset_name"])
        except json.decoder.JSONDecodeError:
            # remove file because it was corrupted somehow
            self._logger.warning(
                    f"Metadata file '{metapath}' was corrupt so we killed it VIVA LA REVOLUCION!")
            os.remove(metapath)
            return defaults
        # defaults.update(data)  # doing an update directly erases key that were not present previously
        for newdata_k, newdata_v in data.items():
            if newdata_k not in defaults or not isinstance(newdata_v, dict):
                defaults[newdata_k] = newdata_v
                continue
            defaults[newdata_k].update(newdata_v)
        defaults["preset_name"] = sanitize_preset_name(defaults["preset_name"], reverse=True)
        return defaults
