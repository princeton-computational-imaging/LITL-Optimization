"""Waveforms Window for LiDARGui."""
import os

import numpy as np
from pyqtgraph import PlotWidget
import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
        QCheckBox, QGridLayout, QLabel, QLineEdit, QPushButton, QSlider, QWidget,
        )
from scipy.constants import c

from .custom_widgets import SliderWithIncrementButtons
from .error_dialog_wrapper import error_dialog
from .main_app import kill_app
from .utils import enable_chkbox, enable_lineeditor
from ..bases import BaseUtility
from ..process_pipelines.thresholding_algorithms import (
        RisingEdgePointCloudProcessor,
        DSP as DSPGaussTemplate,
        DSPCosTemplate,
        DOWNSAMPLING_THRESHOLDING_ALGORITHMS,
        NON_WAVEFORM_BASED_THRESHOLDING_PROCESSORS,
        )
from ..process_pipelines import ProcessPipeline
from ..utils import (
        get_metadata, is_list_like,
        spherical2cartesian,
        )
from ..settings import WHITE_BACKGROUND


DEFAULT_YRANGE = 1.01


class WaveformsViewport(QWidget, BaseUtility):
    """Waveform widget."""

    def __init__(self, mainhub, **kwargs):
        """Waveform widget init method."""
        QWidget.__init__(self)
        BaseUtility.__init__(self, **kwargs)
        self.mainhub = mainhub
        self.build_gui()
        self.mainhub.current_frame_changed.connect(self.refresh_viewport)
        self.mainhub.current_preset_changed.connect(self.refresh_viewport)
        # self.refresh_viewport()
        # waveforms attributes
        self.loaded_frame = None
        self.loaded_pc = None
        self.loaded_preset = None
        self.process_pipeline = None
        self._current_pt = 0
        self._current_laser = 0
        self.axis = None
        self.current_pt_mesh = None
        self._thresholding_algorithm = None
        self.sub_waveforms_curves = None
        self.waveform_curve = None
        self.max_vline = None
        self.current_pt_coordinates = None
        self.highlight_subray_idx = None
        self.thresholding_noise_floor_curve = None

    def build_gui(self):
        """Build the interface."""
        # global window attribute
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Waveforms")
        # waveform graph
        self.build_graph_section(0, 0, 1, 1)
        # waveform controls
        self.build_waveform_controls_section(1, 0, 1, 1)
        # graph controls
        self.build_graph_controls(2, 0, 1, 1)
        # other controls
        self.build_other_controls(3, 0, 1, 1)
        # window controls
        self.build_window_controls(4, 0, 1, 1)

    def build_graph_section(self, *args):
        """Build the graph viewport."""
        self.graph_layout = QGridLayout()
        self.layout.addLayout(self.graph_layout, *args)
        self.graphWidget = PlotWidget()
        self.graphWidget.setBackground((255, 255, 255))
        self.graphWidget.setYRange(0, DEFAULT_YRANGE, padding=0)
        self.graphWidget.addLegend()
        self.graph_layout.addWidget(self.graphWidget, 1, 0, 1, 1)

    def build_graph_controls(self, *args):
        """Build the graph controls."""
        self.graph_controls_layout = QGridLayout()
        self.layout.addLayout(self.graph_controls_layout, *args)
        self.show_subwaveforms_chkbox = QCheckBox("Show sub-waveforms")
        self.show_subwaveforms_chkbox.clicked.connect(self.refresh_viewport)
        self.graph_controls_layout.addWidget(self.show_subwaveforms_chkbox, 0, 0, 1, 1)

        # pt slider
        self.pt_label = QLabel()
        self.pt_label.setText("Point: ")
        self.graph_controls_layout.addWidget(self.pt_label, 1, 0, 1, 1)
        self.previous_pt_btn = QPushButton("<")
        self.previous_pt_btn.clicked.connect(self.previous_pt)
        self.graph_controls_layout.addWidget(self.previous_pt_btn, 1, 1, 1, 1)
        self.pt_slider = QSlider(Qt.Horizontal)
        self.pt_slider.setMinimum(0)
        self.pt_slider.valueChanged.connect(self.refresh_viewport)
        self.graph_controls_layout.addWidget(self.pt_slider, 1, 2, 1, 2)
        self.next_pt_btn = QPushButton(">")
        self.next_pt_btn.clicked.connect(self.next_pt)
        self.graph_controls_layout.addWidget(self.next_pt_btn, 1, 4, 1, 1)
        self.pt_number_label = QLabel("0")
        self.graph_controls_layout.addWidget(self.pt_number_label, 1, 5, 1, 1)

        # laser slider
        self.laser_label = QLabel()
        self.laser_label.setText("Laser: ")
        self.graph_controls_layout.addWidget(self.laser_label, 2, 0, 1, 1)
        self.previous_laser_btn = QPushButton("<")
        self.previous_laser_btn.clicked.connect(self.previous_laser)
        self.graph_controls_layout.addWidget(self.previous_laser_btn, 2, 1, 1, 1)
        self.laser_slider = QSlider(Qt.Horizontal)
        self.laser_slider.setMinimum(0)
        self.laser_slider.valueChanged.connect(self.refresh_viewport)
        self.graph_controls_layout.addWidget(self.laser_slider, 2, 2, 1, 2)
        self.next_laser_btn = QPushButton(">")
        self.next_laser_btn.clicked.connect(self.next_laser)
        self.graph_controls_layout.addWidget(self.next_laser_btn, 2, 4, 1, 1)
        self.laser_number_label = QLabel("0")
        self.graph_controls_layout.addWidget(self.laser_number_label, 2, 5, 1, 1)

    def build_waveform_controls_section(self, *args):
        """Builds the waveform controls section."""
        self.waveform_controls_section_layout = QGridLayout()
        self.layout.addLayout(self.waveform_controls_section_layout, *args)

        # vanilla
        self.custom_chkbox = QCheckBox("Custom parameters")
        self.custom_chkbox.clicked.connect(self.on_custom_chkbox_clicked)
        self.waveform_controls_section_layout.addWidget(self.custom_chkbox, 0, 0, 1, 1)
        # gain
        self.gain_label = QLabel("Gain [dB]:")
        self.waveform_controls_section_layout.addWidget(self.gain_label, 0, 1, 1, 1)
        self.gain_lineeditor = QLineEdit()
        self.gain_lineeditor.setText("0.0")
        self.waveform_controls_section_layout.addWidget(self.gain_lineeditor, 0, 2, 1, 1)
        # saturation
        self.saturation_label = QLabel("Saturation:")
        self.waveform_controls_section_layout.addWidget(self.saturation_label, 0, 3, 1, 1)
        self.saturation_lineeditor = QLineEdit()
        self.saturation_lineeditor.setText("0.0")
        self.waveform_controls_section_layout.addWidget(self.saturation_lineeditor, 0, 4, 1, 1)

        # update btn
        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self.refresh_viewport)
        self.waveform_controls_section_layout.addWidget(self.update_btn, 2, 0, 1, 1)
        # poissonize
        self.poissonize_signal_chkbox = QCheckBox("Poissonize")
        self.waveform_controls_section_layout.addWidget(self.poissonize_signal_chkbox, 1, 0, 1, 1)
        # SNR
        self.snr_label = QLabel("SNR:")
        self.snr_lineeditor = QLineEdit()
        self.snr_lineeditor.setText("1.0")
        self.waveform_controls_section_layout.addWidget(self.snr_label, 1, 1, 1, 1)
        self.waveform_controls_section_layout.addWidget(self.snr_lineeditor, 1, 2, 1, 1)
        # gaussian denoizer sigma
        self.gaussian_denoizer_sigma_label = QLabel("Gaussian denoizer sigma:")
        self.waveform_controls_section_layout.addWidget(self.gaussian_denoizer_sigma_label, 1, 3, 1, 1)
        self.gaussian_denoizer_sigma_lineeditor = QLineEdit()
        self.gaussian_denoizer_sigma_lineeditor.setText("0.0")
        self.waveform_controls_section_layout.addWidget(self.gaussian_denoizer_sigma_lineeditor, 1, 4, 1, 1)
        self.on_custom_chkbox_clicked(False)

    def build_other_controls(self, *args):
        """Build other controls."""
        self.other_controls_layout = QGridLayout()
        self.layout.addLayout(self.other_controls_layout, *args)
        # buttons
        self.reset_btn = QPushButton("Reset graph view")
        self.reset_btn.clicked.connect(self.reset_view)
        self.other_controls_layout.addWidget(self.reset_btn, 0, 0, 1, 1)
        # pt size
        self.pt_size_widget = SliderWithIncrementButtons(
                increment_symbol="+", decrement_symbol="-",
                name="PC Point size:",
                )
        self.pt_size_widget.decrement_button.clicked.connect(self.decrement_pt_size)
        self.pt_size_widget.increment_button.clicked.connect(self.increment_pt_size)
        self.pt_size_widget.setMinimum(3)
        self.pt_size_widget.setMaximum(20)
        self.pt_size_widget.slider.valueChanged.connect(
                self.add_points_to_main_pc_viewer)
        self.other_controls_layout.addWidget(self.pt_size_widget, 0, 1, 1, 3)

    def build_window_controls(self, *args):
        """The window controls."""
        self.window_controls_layout = QGridLayout()
        self.layout.addLayout(self.window_controls_layout, *args)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.window_controls_layout.addWidget(self.close_btn, 0, 0, 1, 1)

    @property
    def current_frame(self):
        """The current frame number."""
        return self.mainhub.current_frame

    @property
    def current_preset(self):
        """The current preset."""
        return self.mainhub.current_preset

    @property
    def current_pt(self):
        """The current pt visualized."""
        return self.pt_slider.value()

    @property
    def current_laser(self):
        """The current laser visualized."""
        return self.laser_slider.value()

    @property
    def custom_signal(self):
        """Return true if we want a custom signal."""
        return self.custom_chkbox.isChecked()

    @property
    def gain(self):
        """The gain in dB."""
        if self.custom_signal:
            return float(self.gain_lineeditor.text())
        return self.processing_parameters["thresholding_algorithm"]["gain"]

    @property
    def digitization(self):
        """Return the digitization mode."""
        return self.processing_parameters["thresholding_algorithm"]["digitization"]

    @property
    def gaussian_denoizer_sigma(self):
        """The gaussian denoizer sigma."""
        if self.custom_signal or "gaussian_denoizer_sigma" not in self.processing_parameters["thresholding_algorithm"]:
            return float(self.gaussian_denoizer_sigma_lineeditor.text())
        return self.processing_parameters["thresholding_algorithm"]["gaussian_denoizer_sigma"]

    @property
    def poissonize_signal(self):
        """True if we poissonize signal."""
        if self.custom_signal:
            return self.poissonize_signal_chkbox.isChecked()
        return self.processing_parameters["waveform_model"]["poissonize_signal"]

    @property
    def processing_parameters(self):
        """The dict of vanilla processing parameters."""
        return self.process_pipeline.processing_parameters

    @property
    def saturation(self):
        """The saturation."""
        if self.custom_signal:
            return float(self.saturation_lineeditor.text())
        if "saturation" in self.processing_parameters:
            return self.processing_parameters["saturation"]
        return np.inf

    @property
    def snr(self):
        """The SNR."""
        return float(self.snr_lineeditor.text())

    @property
    def show_subwaveforms(self):
        """True if we want to show the sub waveforms."""
        return self.show_subwaveforms_chkbox.isChecked()

    @property
    def thresholding_algorithm(self):
        """Thresholding algorithm."""
        return self._thresholding_algorithm

    @thresholding_algorithm.setter
    def thresholding_algorithm(self, algo):
        self._thresholding_algorithm = algo
        self.setWindowTitle(f"Waveforms: thersholding algo = {self.thresholding_algorithm}")

    @property
    def thresholding_noise_floor(self):
        """The noise floor."""
        return self.process_pipeline.processing_parameters[
                "thresholding_algorithm"]["noise_floor_threshold"]

    @property
    def pt_size(self):
        """The pt size of the pts as displayed on the PC viewer."""
        return self.pt_size_widget.value()

    @property
    def waveform(self):
        """The total waveform for all points in the point cloud."""
        return self._waveform

    @property
    def waveform_range(self):
        """The max distance in the waveform."""
        return self.processing_parameters["waveform_model"]["waveform_range"]

    @property
    def waveform_resolution(self):
        """The waveform resolution."""
        return self.processing_parameters["waveform_model"]["waveform_resolution"]

    @property
    def tauH(self):
        """Pulse width."""
        # No customization implemented yet
        return self.processing_parameters["waveform_model"]["tauH"]

    @property
    def dsp_template(self):
        """Return the DSP template model."""
        return self.processing_parameters["thresholding_algorithm"]["dsp_template"]

    @property
    def sin2cst(self):
        """The constant inside the sin^2 function."""
        ctauH = c * self.tauH * 1e-9  # tau in ns
        return np.pi / ctauH

    def add_points_to_main_pc_viewer(self):
        """Add a big dot where we visualize the current pt."""
        pos = np.asarray(self.current_pt_coordinates)
        colors = np.asarray([[0, 1, 0, 1]] * len(self.current_pt_coordinates))
        if self.highlight_subray_idx is not None:
            colors[self.highlight_subray_idx] = np.array([1, 0, 0, 1])  # red
        where0 = np.linalg.norm(pos, axis=1) == 0.0
        colors[where0, -1] = 0
        if self.current_pt_mesh is None:
            self.current_pt_mesh = gl.GLScatterPlotItem(
                pos=pos, size=self.pt_size,
                color=colors)
            if WHITE_BACKGROUND:
                self.current_pt_mesh.setGLOptions('translucent')
        else:
            self.current_pt_mesh.setData(pos=pos, color=colors, size=self.pt_size)
            if WHITE_BACKGROUND:
                self.current_pt_mesh.setGLOptions('translucent')
        if self.current_pt_mesh not in self.mainhub.pc_viewport.pc_viewer.items:
            self.mainhub.pc_viewport.pc_viewer.addItem(self.current_pt_mesh)

    def add_delta_diracs_to_viewport(self):
        """Adds delta dirac(s) to viewport for given pc and intensities."""
        # height of delta = intensity
        pc = self.process_pipeline.waveform_processor.processed_pc[self.current_laser, self.current_pt]
        intensities = self.process_pipeline.waveform_processor.processed_intensities[
                self.current_laser, self.current_pt]
        if self.thresholding_algorithm not in DOWNSAMPLING_THRESHOLDING_ALGORITHMS:
            if self.sub_waveforms_curves is not None:
                for sub in self.sub_waveforms_curves:
                    sub.setData([], [])
            r = pc[2]
            if self.waveform_curve is None:
                self.waveform_curve = self.graphWidget.plot(
                    [r, r], [0, intensities],
                    pen={"color": "k", "width": 2},
                    )
            else:
                self.waveform_curve.setData([r, r], [0, intensities]),
            self.current_pt_coordinates = spherical2cartesian([pc[2]], [pc[0]], [pc[1]])
            self.highlight_subray_idx = None
        else:
            # show multiple delta diracs using sub waveform curves
            maxI = np.max(intensities)
            wheremaxI = np.where(intensities == maxI)[0][0]
            maxR = pc[wheremaxI][2]
            if self.sub_waveforms_curves is None:
                self.sub_waveforms_curves = []
                for idx, (subray, intensity) in enumerate(zip(pc, intensities)):
                    r = subray[2]
                    curve = self.graphWidget.plot(
                            [r, r], [0, intensity],
                            pen={"color": idx, "width": 2},
                            )
                    self.sub_waveforms_curves.append(curve)
            else:
                for idx, (subray, intensity) in enumerate(zip(pc, intensities)):
                    r = subray[2]
                    self.sub_waveforms_curves[idx].setData(
                            [r, r], [0, intensity],
                            )
            if self.waveform_curve is None:
                self.waveform_curve = self.graphWidget.plot(
                        [maxR, maxR], [0, maxI],
                        pen={"color": "k", "width": 2},
                        )
            else:
                self.waveform_curve.setData([maxR, maxR], [0, maxI])
            self.current_pt_coordinates = spherical2cartesian(pc[:, 2], pc[:, 0], pc[:, 1])
            self.highlight_subray_idx = wheremaxI

    def add_waveforms_to_viewport(self):
        """Adds full waveforms to viewport for given pc and intensities."""
        if self.custom_signal:
            self.process_pipeline.waveform_processor.snr = self.snr
            self.process_pipeline.waveform_processor.poissonize_signal = self.poissonize_signal
        if self.dsp_template == "CosTemplate":
            dsp = DSPCosTemplate(
                    pulse_width=0.299792*self.tauH,
                    sin2cst=self.sin2cst,
                    snr=self.snr,
                    max_range=self.waveform_range,
                    time_discretization=self.waveform_resolution,
                    correction_factor=1.0,  # unsettable at the moment
                    gain=self.gain,
                    noise_floor=self.thresholding_noise_floor,
                    saturation=self.saturation,
                    digitization=self.digitization,
                    loglevel=self._loglevel,
                    )
        else:
            dsp = DSPGaussTemplate(
                gain=self.gain, saturation=self.saturation,
                noise_floor=self.thresholding_noise_floor,
                gaussian_denoizer_sigma=self.gaussian_denoizer_sigma,
                loglevel=self._loglevel,
                )
        tot_waveform, subwaveforms = self.process_pipeline.waveform_processor(
                self.current_laser, ipt=self.current_pt, return_subwaveforms=True)
        if self.thresholding_algorithm == RisingEdgePointCloudProcessor.algorithm_name:
            dsp.noise_floor = 0.0
        if is_list_like(self.thresholding_noise_floor):
            dsp.noise_floor = self.thresholding_noise_floor[self.current_laser]
        tot_waveform = dsp.process(tot_waveform[0, :])
        subs = []
        for sub in subwaveforms:
            subs.append(dsp.process(sub[0, :]))
        subwaveforms = np.asarray(subs)
        wheremax = self.axis[np.argmax(tot_waveform)]
        norm = np.max(tot_waveform)
        self.graphWidget.setYRange(0, norm, padding=0)
        pc = self.process_pipeline.waveform_processor.processed_pc[
                :, self.current_laser, self.current_pt, :]
        pt = pc[4]
        # 1st pt to display is the one resulting from the wvfrm
        self.current_pt_coordinates = spherical2cartesian(
                [wheremax] + pc[:, 2].tolist(),
                [pt[0]] + pc[:, 0].tolist(),
                [pt[1]] + pc[:, 1].tolist())
        self.highlight_subray_idx = 0
        if self.sub_waveforms_curves is None:
            if self.show_subwaveforms:
                self.sub_waveforms_curves = []
                for idx, sub_waveform in enumerate(subwaveforms):
                    self.sub_waveforms_curves.append(self.graphWidget.plot(
                        self.axis, sub_waveform,
                        pen={"color": idx, "width": 2}))
        else:
            for idx, (line, ydata) in enumerate(zip(self.sub_waveforms_curves, subwaveforms)):
                if self.show_subwaveforms:
                    line.setData(self.axis, ydata)
                    line.setPen({"color": idx, "width": 2})
                else:
                    line.setPen({"color": "w"})
        if self.waveform_curve is None:
            self.waveform_curve = self.graphWidget.plot(
                    self.axis, tot_waveform, name="total",
                    pen={"color": "k", "width": 2},
                    )
        else:
            self.waveform_curve.setData(self.axis, tot_waveform)
        if self.thresholding_noise_floor_curve is None:
            self.thresholding_noise_floor_curve = self.graphWidget.plot(
                    [0, max(self.axis)], [self.thresholding_noise_floor] * 2,
                    pen={"color": "r", "width": 2, "stype": 2},
                    name="noise floor threshold",
                    )
        # if self.max_vline is None:
        #     self.max_vline = self.graphWidget.plot(
        #             [0, wheremax], [max_, max_],
        #             pen={"color": "k", "width": 2, "style": 2},
        #             )
        # else:
        #     self.max_vline.setData([0, wheremax], [max_, max_])

    def change_current_pt(self, value):
        """Change the current pt."""
        self.current_pt = value

    def change_current_laser(self, value):
        """Change the current laser."""
        self.current_laser = value

    def decrement_pt_size(self):
        """Decrement pt size."""
        size = self.pt_size_widget.value()
        if size == self.pt_size_widget.minimum():
            return
        self.pt_size_widget.setValue(size - 1)

    def increment_pt_size(self):
        """Increment pt size."""
        size = self.pt_size_widget.value()
        if size == self.pt_size_widget.maximum():
            return
        self.pt_size_widget.setValue(size + 1)

    def load_pc(self, pipeline):
        """Load or reload raw pc."""
        # check if current preset and frame is MAX WAVEFORM
        if self.loaded_frame == self.current_frame and self.process_pipeline is not None and (
                self.loaded_preset == self.current_preset):
            # nothing to reload
            return
        self.loaded_frame = self.current_frame
        self.loaded_preset = self.current_preset
        # use pipeline to load data and compute intensities
        # but not applying thresholding
        pipeline.load_raw_data()
        pipeline.compute_hitmask()
        pipeline.compute_intensities()
        # compute waveforms -> this will create a generator but the actual calculation
        # is done upon iteration / call only
        pipeline.compute_waveform()
        # # need to rearrange data in subrays if necessary
        self.process_pipeline = pipeline
        # TODO: we could set custom values here not necessarily the ones used to
        # process the waveform
        axis = pipeline.waveform_processor.range_axis
        if self.axis is None:
            self.axis = axis
            self.reset_view()
        else:
            self.axis = axis
        if pipeline.waveform_processor.model_name == "None":
            pc = pipeline.waveform_processor.raw_pc
            nlasers = pc.shape[0]
            npts = pc.shape[1]
        else:
            pc = pipeline.waveform_processor.processed_pc
            nlasers = pc.shape[1]
            npts = pc.shape[2]
        self.laser_slider.setMaximum(nlasers - 1)
        self.pt_slider.setMaximum(npts - 1)

    def on_custom_chkbox_clicked(self, state):
        """Method called when custom chkbox is clicked."""
        for lineeditor in [
                self.snr_lineeditor, self.gain_lineeditor, self.saturation_lineeditor,
                self.gaussian_denoizer_sigma_lineeditor,
                ]:
            enable_lineeditor(lineeditor, enable=state)
        enable_chkbox(self.poissonize_signal_chkbox, enable=state)

    def previous_laser(self):
        """Move to previous laser."""
        if self.current_laser == 0:
            return
        self.laser_slider.setValue(self.current_laser - 1)

    def previous_pt(self):
        """Move to previous pt."""
        current_pt = self.current_pt
        if current_pt == 0:
            return
        self.pt_slider.setValue(current_pt - 1)

    def next_laser(self):
        """Move to next laser."""
        if self.current_laser == self.laser_slider.maximum():
            return
        self.laser_slider.setValue(self.current_laser + 1)

    def next_pt(self):
        """Move to next pt."""
        current_pt = self.current_pt
        if current_pt == self.pt_slider.maximum():
            return
        self.pt_slider.setValue(current_pt + 1)

    def reset_view(self):
        """Resets the view to default."""
        self.graphWidget.setXRange(0, np.max(self.axis))
        self.graphWidget.setYRange(0, DEFAULT_YRANGE, padding=0)

    @error_dialog(
            text="An error occured when refreshing waveform viewport.",
            callback=kill_app)
    def refresh_viewport(self, *args, **kwargs):
        """Update the waveform graph."""
        if not self.isVisible():
            return
        # update pt label and laser label
        self.pt_number_label.setText(str(self.current_pt))
        self.laser_number_label.setText(str(self.current_laser))
        framedir = os.path.join(
                self.mainhub.frame_dirs[self.current_frame])
        processing_parameters = get_metadata(framedir, self.mainhub.current_preset)
        pipeline = ProcessPipeline(
                processing_parameters,
                framedir=framedir,
                loglevel=self._loglevel)
        # TODO: use waveform generation feature of pipeline (new)
        self.thresholding_algorithm = pipeline.thresholding_algorithm
        self.load_pc(pipeline)
        if self.process_pipeline is None:
            # could not load data
            return
        if self.thresholding_algorithm in NON_WAVEFORM_BASED_THRESHOLDING_PROCESSORS:
            # add delta dirac like function
            self.add_delta_diracs_to_viewport()
        else:
            self.add_waveforms_to_viewport()
        # call this at the end since the add methods above decide where should be the pts
        self.add_points_to_main_pc_viewer()
