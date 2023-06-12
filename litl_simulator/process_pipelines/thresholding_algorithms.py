"""Processors that create final data as viewed by the corresponding viewports."""
import abc
# from logging.handlers import WatchedFileHandler

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.constants import c

from .bases import BaseLidarProcessor
from .utils import concatenate_arrays, is_list_like
from ..bases import BaseUtility


MAX_SUBRAY_THRESHOLDING_ALGORITHM = "Max Subray Intensity"
MAX_WAVEFORM_THRESHOLDING_ALGORITHM = "PC = Max of total Waveform"
DOWNSAMPLING_THRESHOLDING_ALGORITHMS = [
        MAX_SUBRAY_THRESHOLDING_ALGORITHM,
        MAX_WAVEFORM_THRESHOLDING_ALGORITHM,
        ]


class DSP(BaseUtility):
    """Fake Digital Signal Processor."""
    name = "GaussTemplate"

    def __init__(
            self, waveform=None, gain=None, saturation=None, noise_floor=None,
            gaussian_denoizer_sigma=None, digitization=None,
            **kwargs,
            ):
        """Fake DSP init method."""
        super().__init__(**kwargs)
        self.waveform = waveform
        self.gaussian_denoizer_sigma = gaussian_denoizer_sigma
        self.gain = gain
        self.saturation = saturation
        self.noise_floor = noise_floor
        self.digitization = digitization

    @staticmethod
    def add_power_gain(waveform, gain):
        """Adds power gain to waveform."""
        if gain == 0.0 or gain is None:
            return waveform
        return waveform * 10 ** (gain / 10)  # gain in dB

    @staticmethod
    def add_saturation(waveform, floor, saturation):
        """Adds saturation to a waveform."""
        if saturation == 0.0 or saturation is None:
            saturation = None
        if floor > saturation:
            raise ValueError("Floor threshold is > saturation.")
        return np.clip(waveform - floor, 0.0, saturation - floor)

    @staticmethod
    def gaussian_denoizing(waveform, sigma):
        """Denoize signal using simple gaussian filter."""
        if sigma == 0.0 or sigma is None:
            # disabled
            return waveform
        return ndimage.gaussian_filter1d(waveform, sigma=sigma, mode="nearest")

    @staticmethod
    def do_digitization(waveform, digitization, saturation):
        """Digitizes waveform to integers."""
        if digitization == "float" or digitization is None:
            return waveform
        elif digitization == "uint16":
            scale = 2**16 - 1
            dtype = np.uint16
        elif digitization == "uint8":
            scale = 2**8 - 1
            dtype = np.uint8
        else:
            raise NotImplementedError
        return (scale * (waveform / saturation)).astype(dtype)

    def process(self, waveform=None):
        """Process raw waveform into signal processed waveform."""
        if waveform is None:
            if self.waveform is None:
                raise ValueError("No waveform to process.")
            waveform = self.waveform
        waveform = self.add_power_gain(waveform, self.gain)
        waveform = self.add_saturation(waveform, self.noise_floor, self.saturation)
        waveform = self.gaussian_denoizing(waveform, self.gaussian_denoizer_sigma)
        waveform = self.do_digitization(waveform, self.digitization, self.saturation)
        return waveform


class DSPCosTemplate(DSP):
    """DSP with cos function fitting."""
    name = "CosTemplate"

    def __init__(
            self, pulse_width=None, sin2cst=None, max_range=120,
            time_discretization=2400, snr=100,
            correction_factor=None,
            digitization=None,
            **kwargs,
            ):
        """Fake DSP init method with cos fitting.

        max_range is maximum simulation distance in [m].
        time discretization is the number of interpolation points of the wavefront.
        """
        super().__init__(**kwargs)

        resolution = max_range/time_discretization
        # create positions for kernel
        xpos = np.arange(-pulse_width/resolution/2, pulse_width/resolution/2)*resolution

        self.kernel_weights = np.square(np.cos(sin2cst*xpos))
        # normalize kernel

        self.kernel_norm = np.sum(self.kernel_weights)
        self.kernel_weights /= self.kernel_norm

        # store params
        self.pulse_width = pulse_width
        self.sin2cst = sin2cst
        self.snr = snr
        self.correction_factor = correction_factor

    def inverse_peak_denoising(self, waveform=None, pulse_width=None, sin2cst=None):
        """Peak denoising method."""
        if pulse_width == 0.0 or sin2cst is None or sin2cst == 0.0 or pulse_width is None:
            # disabled
            return waveform
        waveform = ndimage.convolve1d(waveform, self.kernel_weights, mode='nearest')
        return waveform

    @staticmethod
    def normalize_power_gain(waveform, gain):
        """Adds power gain to waveform."""
        if gain == 0.0 or gain is None:
            return waveform
        return waveform / 10 ** (gain / 10)  # gain in dB

    def rescale_intensities(self, waveform=None, snr=None,  correction_factor=None):
        """Estimating intensities becoms increasingly difficult with changing pulse widths.

        Each different layer therefore might need to be compensated.
        """
        if snr is None or correction_factor is None:
            return waveform
        power = self.pulse_width/2 * correction_factor * snr
        return waveform/power

    def estimate_ambient(self, waveform=None):
        """Estimate ambiant light illumination.

        substract the ambient illumination
        Estimate the ambient from the median
        """
        if waveform.ndim > 1:
            median = np.median(waveform, axis=1)
            waveform = waveform - median[:, np.newaxis]
        else:
            median = np.median(waveform)
            waveform = waveform - median
        return np.clip(waveform, 0, np.inf)

    @staticmethod
    def remove_gain(waveform, gain):
        """Adds power gain to waveform."""
        if gain == 0.0 or gain is None:
            return waveform
        return waveform / (10 ** (gain / 10))  # gain in dB

    @staticmethod
    def do_digitization(waveform, digitization):
        """Digitizes waveform to integers."""
        if digitization == "float" or digitization is None:
            return waveform
        elif digitization == "uint16":
            scale = 2**16 - 1
            dtype = np.uint16
        elif digitization == "uint8":
            scale = 2**8 - 1
            dtype = np.uint8
        else:
            raise NotImplementedError
        return (scale * (waveform)).astype(dtype)

    def process(self, waveform=None):
        """Process raw waveform."""
        if waveform is None:
            if self.waveform is None:
                raise ValueError("No waveform to process.")
            waveform = self.waveform
        waveform = self.add_power_gain(waveform, self.gain)
        waveform = self.add_saturation(waveform, self.noise_floor, self.saturation)
        waveform = self.inverse_peak_denoising(waveform, pulse_width=self.pulse_width, sin2cst=self.sin2cst)
        waveform = self.estimate_ambient(waveform)
        waveform = self.rescale_intensities(waveform, self.snr, self.correction_factor)
        waveform = self.remove_gain(waveform, self.gain)
        waveform = self.do_digitization(waveform, self.digitization)
        return waveform


class BaseThresholdingAlgorithmProcessor(BaseLidarProcessor, abc.ABC):
    """Base class for point cloud processors."""
    algorithm_name = None
    waveform_based = None
    has_signal_processing = False

    def __init__(
            self,
            framedir=None,
            algorithm=None,
            noise_floor_threshold=None,
            gain=None, saturation=None,
            gaussian_denoizer_sigma=None,
            upsampling_ratio=5,
            digitization=None, dsp_template=None, correction_factor=None,
            # find_peaks arguments
            height=None, min_threshold=None, max_threshold=None,
            distance=None, min_prominence=None, max_prominence=None,
            min_width=None, max_width=None, wlen=None, rel_height=None,
            min_plateau_size=None, max_plateau_size=None,
            # Waveform params
            tauH=None, waveform_range=None, waveform_min_dist=None,
            waveform_resolution=None, poissonize_signal=None, snr=None, model=None,
            **kwargs
            ):
        """Base point cloud processor init method."""
        if self.waveform_based is None:
            raise ValueError("Waveform based or not?")
        if algorithm != self.algorithm_name:
            raise ValueError("Algorithm mismatch!")
        self.noise_floor_threshold = noise_floor_threshold
        self.gain = gain
        self.saturation = saturation
        self.gaussian_denoizer_sigma = gaussian_denoizer_sigma
        self.digitization = digitization
        self.dsp_template = dsp_template
        self.correction_factor = correction_factor
        self.upsampling_ratio = upsampling_ratio
        self.height = height
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.distance = distance
        self.min_prominence = min_prominence
        self.max_prominence = max_prominence
        self.min_width = min_width
        self.max_width = max_width
        self.wlen = wlen
        self.rel_height = rel_height
        self.min_plateau_size = min_plateau_size
        self.max_plateau_size = max_plateau_size
        # Waveform params
        self.tauH = tauH
        if is_list_like(self.tauH):
            ctauH = [c*i*1e-9 for i in self.tauH]
            self.sin2cst = (np.pi / np.asarray(ctauH)).tolist()
        else:
            ctauH = c * tauH * 1e-9  # tau in ns
            self.sin2cst = np.pi / ctauH
        self.waveform_range = waveform_range
        self.waveform_min_distance = waveform_min_dist
        self.waveform_resolution = waveform_resolution
        self.poissonize_signal = poissonize_signal
        self.snr = snr
        super().__init__(framedir=framedir, **kwargs)

    @abc.abstractmethod
    def apply_thresholding(self, *args, **kwargs):
        """Apply thresholding algorithm."""
        pass

    def process(self):
        """Actually do the processing."""
        pc, intensities, hitmask = self.apply_thresholding()
        pc = self.convert_to_cartesian(pc, hitmask)
        self.processed_pc = pc
        self.processed_intensities = intensities
        self.hitmask = hitmask


class NoThresholdingPointCloudProcessor(BaseThresholdingAlgorithmProcessor):
    """Point cloud processor with no thresholding method."""
    algorithm_name = "None"
    waveform_based = False

    def apply_thresholding(self):
        """Actually applies no thresholding."""
        return self.raw_pc, self.raw_intensities, self.hitmask


class DownsamplingPointCloudProcessor(BaseThresholdingAlgorithmProcessor):
    """Processor base class for downsampling algorithm."""
    pass


class MiddleRayPointCloudProcessor(DownsamplingPointCloudProcessor):
    """Point cloud thresholding processor that only retain the 'main' ray point.

    it's like if there was no upsampling in the first place.
    """
    algorithm_name = "Main Ray Only downsampling"
    waveform_based = False

    def apply_thresholding(self):
        """Return the processed point cloud only keeping the main ray."""
        subrays, subrays_I = self.rearrange_data_in_subrays(
                self.raw_pc, self.raw_intensities,
                upsampling_ratio=self.upsampling_ratio)
        main = self.upsampling_ratio ** 2 // 2
        phis = subrays[main, ..., 1]
        thetas = subrays[main, ..., 0]
        rhos = subrays[main, ..., 2]
        phis, thetas, rhos = concatenate_arrays(phis, thetas, rhos)
        pc = np.asarray((thetas, phis, rhos)).T
        intensities = concatenate_arrays(subrays_I[main])
        if intensities.ndim == 2 and intensities.shape[-1] in (3, 4):
            hitmask = np.ones(intensities.shape[0], dtype=bool)
        else:
            hitmask = np.ones_like(intensities, dtype=bool)
        return pc, intensities, hitmask


class MaxSubrayIntensityPointCloudProcessor(DownsamplingPointCloudProcessor):
    """Point cloud processor that only sets the intensity/distance of highest subray."""
    algorithm_name = MAX_SUBRAY_THRESHOLDING_ALGORITHM
    waveform_based = False

    def apply_thresholding(self):
        """Return the processed point cloud based on highest subray intensity.

        The subray with highest intensity will be picked, the others will be
        discarded.
        """
        subrays, subrays_I = self.rearrange_data_in_subrays(
                self.raw_pc, self.raw_intensities,
                upsampling_ratio=self.upsampling_ratio)
        # get phi and theta from middle ray
        ray = subrays[self.upsampling_ratio ** 2 // 2]
        phis = ray[..., 1]
        thetas = ray[..., 0]
        # subrays is n_subrays x n_lasers x npts_per_laser x nfeats
        # subrays_I is n_subrays x n_lasers x npts_per_laser
        npts_per_laser = ray.shape[1]
        n_lasers = subrays.shape[1]
        new = np.empty((n_lasers, npts_per_laser))  # , 2))  # only keep distances and cos_theta
        if subrays_I.ndim > 3 and subrays_I.shape[-1] > 1:
            # subrays_I =  n_subrays x n channels x n pts per channel x RGBA
            # decide the maximum accross all channels
            where_max = np.argmax(np.mean(subrays_I, axis=-1), axis=0)
            intensities = np.empty((n_lasers, npts_per_laser, subrays_I.shape[-1]))
        else:
            where_max = np.argmax(subrays_I, axis=0)
            intensities = np.empty((n_lasers, npts_per_laser))
        # where_max is n_lasers x npts_per_laser
        for ilaser in range(n_lasers):
            for ipt in range(npts_per_laser):
                subray_idx = where_max[ilaser, ipt]
                new[ilaser, ipt] = subrays[subray_idx, ilaser, ipt, 2]  # 2:4]
                intensities[ilaser, ipt] = subrays_I[subray_idx, ilaser, ipt]
        # flatten everything
        phis, thetas, rhos = concatenate_arrays(phis, thetas, new)  # [..., 0])
        # cos_theta_in = concatenate_arrays(new[..., 1])
        pc = np.array((thetas, phis, rhos)).T  # , cos_theta_in)).T
        intensities = concatenate_arrays(intensities)
        return pc, intensities, np.ones_like(pc[..., 0], dtype=bool)


class BaseWaveformPointCloudProcessor(DownsamplingPointCloudProcessor):
    """Base class for waveform based thresholding algorithms."""
    waveform_based = True

    def __init__(self, *args, **kwargs):
        """Base waveform based thresholding algorithm processor init method."""
        super().__init__(*args, **kwargs)
        self.raw_waveforms = None

    def get_dsp(self):
        """Return DSP object."""
        if self.dsp_template == DSP.name:
            return DSP(
                   gain=self.gain, noise_floor=self.noise_floor_threshold,
                   saturation=self.saturation,
                   gaussian_denoizer_sigma=self.gaussian_denoizer_sigma,
                   digitization=self.digitization,
                   loglevel=self._loglevel,
                   )
        elif self.dsp_template == DSPCosTemplate.name:
            return DSPCosTemplate(
                   pulse_width=self.tauH,
                   sin2cst=self.sin2cst,
                   max_range=self.waveform_range,
                   time_discretization=self.waveform_resolution, correction_factor=self.correction_factor,
                   gain=self.gain, noise_floor=self.noise_floor_threshold,
                   saturation=self.saturation,
                   digitization=self.digitization,
                   loglevel=self._loglevel,
                   )
        self._logger.error(f'{self.dsp_template} not implemented')
        raise NotImplementedError(str(self.dsp_template))

    def set_raw_data(self, raw_waveforms=None, **kwargs):
        """Sets raw data."""
        if raw_waveforms is not None:
            self.raw_waveforms = raw_waveforms
        super().set_raw_data(**kwargs)


class MaxWaveformPointCloudProcessor(BaseWaveformPointCloudProcessor):
    """Max waveform point cloud processor."""
    algorithm_name = MAX_WAVEFORM_THRESHOLDING_ALGORITHM
    has_signal_processing = True

    def apply_thresholding(self):
        """Return the processed point cloud based on highest subray intensity.

        The subray with highest intensity will be picked, the others will be
        discarded.
        """
        # raw pc is an array of size [n_subrays, n_lasers, npts, 3]
        ray = self.raw_pc[self.upsampling_ratio ** 2 // 2]
        phis = ray[:, :, 1]  # n_lasers x npts_per_laser
        thetas = ray[:, :, 0]
        # raw waveforms is an iterator that iterates over an array of size
        # n_lasers x npts x waveform_resolution
        nlasers = self.raw_waveforms.nlasers
        npts = self.raw_waveforms.npts
        final_dists = np.empty((nlasers, npts))
        final_intensities = np.empty((nlasers, npts))
        range_axis = self.raw_waveforms.range_axis
        final_hitmask = np.empty((nlasers, npts), dtype=bool)
        dsp = self.get_dsp()
        for ilaser, waveform in enumerate(self.raw_waveforms):
            # waveform this laser is an array of size [npts x waveform_resolution]
            waveform = dsp.process(waveform)
            # take max over last axis to get where the maximal peak is located
            final_dists[ilaser, :] = range_axis[np.argmax(waveform, axis=1)]
            max_ = np.max(waveform, axis=1)
            final_intensities[ilaser, :] = max_
            final_hitmask[ilaser, :] = max_ != 0.0

        thetas, phis, rhos = concatenate_arrays(thetas, phis, final_dists)
        final_pc = np.array([thetas, phis, rhos]).T
        intensities, hitmask = concatenate_arrays(final_intensities, final_hitmask)
        return final_pc, intensities, hitmask


class RisingEdgePointCloudProcessor(BaseWaveformPointCloudProcessor):
    """Rising edge signal processor. Should only give decent result if used with sin^2 waveform."""
    algorithm_name = "Rising Edge"
    has_signal_processing = True

    def __init__(self, *args, **kwargs):
        """Rising edge init method."""
        super().__init__(*args, **kwargs)
        if self.noise_floor_threshold is None:
            raise ValueError(
                    "Need to set 'noise_floor_threshold' in 'thresholding_algorithm' model..")

    def apply_thresholding(self):
        """Return the processed point cloud."""
        # raw pc is an array of size [n_subrays, n_lasers, npts, 3]
        ray = self.raw_pc[self.upsampling_ratio ** 2 // 2]
        phis = ray[:, :, 1]  # n_lasers x npts_per_laser
        thetas = ray[:, :, 0]
        # raw waveforms is an iterator that iterates over an array of size
        # n_lasers x npts x waveform_resolution
        nlasers = ray.shape[0]  # self.raw_waveforms.shape[0]#.nlasers
        npts = ray.shape[1]  # self.raw_waveforms.npts
        final_dists = np.empty((nlasers, npts))
        final_intensities = np.empty((nlasers, npts))
        range_axis = self.raw_waveforms.range_axis
        final_hitmask = np.empty((nlasers, npts), dtype=bool)
        dsp = self.get_dsp()
        dsp.noise_floor = 0.0  # set to 0 since it does not have the same meaning as usual
        for ilaser in range(nlasers):
            waveforms = self.raw_waveforms(ilaser)
            # waveform this laser is an array of size [npts x waveform_resolution]
            waveforms = dsp.process(waveforms)
            thresh = self.noise_floor_threshold
            if is_list_like(thresh):
                thresh = thresh[ilaser]
            for ipt, waveform in enumerate(waveforms):
                # cannot vectorize loop since find_peaks only works on 1D arrays...
                if np.isnan(waveform).any():
                    raise RuntimeError("there are nans in the waveform...")
                # idea from: https://stackoverflow.com/a/50365462/6362595
                # first find all rising edges
                rising_edges_idx = np.flatnonzero(
                        np.logical_and(
                            waveform[:-1] <= thresh,
                            waveform[1:] > thresh,
                            ))
                if not len(rising_edges_idx):
                    # no rising edge strong enough
                    final_intensities[ilaser, ipt] = 0.0
                    final_dists[ilaser, ipt] = 0.0
                    final_hitmask[ilaser, ipt] = False
                    continue
                # return the edge with highest intensity
                wheremax = np.argmax(waveform)
                # find closest to max on the right
                diff = wheremax - rising_edges_idx
                diff[diff < 0] = 1000000  # make negative values infinite so that we discard them
                closest = diff.argmin()
                dist_idx = rising_edges_idx[closest]
                final_intensities[ilaser, ipt] = waveform[wheremax]
                final_hitmask[ilaser, ipt] = True
                final_dists[ilaser, ipt] = range_axis[dist_idx]
        thetas, phis, rhos = concatenate_arrays(thetas, phis, final_dists)
        final_pc = np.array([thetas, phis, rhos]).T
        intensities, hitmask = concatenate_arrays(final_intensities, final_hitmask)
        return final_pc, intensities, hitmask


class RisingEdgeAndWidthPointCloudProcessor(BaseWaveformPointCloudProcessor):
    """Rising edge signal processor. Should only give decent result if used with sin^2 waveform.

    This processor expects different parameters for each ray.
    """
    algorithm_name = "Rising Edge and pulse params"
    has_signal_processing = True

    def get_dsp(self, nlasers):
        """Return DSP object."""
        # check if len of parameters is the same
        # if self.dsp_template == DSPCosTemplate.name:
        #    assert(self.tauH != self.raw_waveforms.nlasers)
        #    assert(self.sin2cst != self.raw_waveforms.nlasers)
        per_layer_dsp = []
        for nlaser in range(nlasers):
            tauH = self.tauH
            if is_list_like(tauH):
                tauH = tauH[nlaser]
            sin2cst = self.sin2cst
            if is_list_like(sin2cst):
                sin2cst = sin2cst[nlaser]
            snr = self.snr
            if is_list_like(snr):
                snr = snr[nlaser]
            noise_floor = self.noise_floor_threshold
            if is_list_like(noise_floor):
                noise_floor = noise_floor[nlaser]
            gaussian_sigma = self.gaussian_denoizer_sigma
            if is_list_like(self.gaussian_denoizer_sigma):
                gaussian_sigma = self.gaussian_denoizer_sigma[nlaser]
            if self.dsp_template == DSP.name:
                per_layer_dsp.append(DSP(
                                        gain=self.gain, noise_floor=noise_floor,
                                        saturation=self.saturation,
                                        gaussian_denoizer_sigma=gaussian_sigma,
                                        digitization=self.digitization,
                                        loglevel=self._loglevel,
                                        ))
            elif self.dsp_template == DSPCosTemplate.name:
                per_layer_dsp.append(DSPCosTemplate(
                                    pulse_width=0.299792*tauH,
                                    sin2cst=sin2cst,
                                    snr=snr,
                                    max_range=self.waveform_range,
                                    time_discretization=self.waveform_resolution,
                                    correction_factor=self.correction_factor,
                                    gain=self.gain, noise_floor=noise_floor,
                                    saturation=self.saturation,
                                    digitization=self.digitization,
                                    loglevel=self._loglevel,
                                    ))
        if len(per_layer_dsp) >= 1:
            return per_layer_dsp
        self._logger.error(f'{self.dsp_template} not implemented')
        raise NotImplementedError(str(self.dsp_template))

    def apply_thresholding(self):
        """Return the processed point cloud."""
        # raw pc is an array of size [n_subrays, n_lasers, npts, 3]
        ray = self.raw_pc[self.upsampling_ratio ** 2 // 2]
        phis = ray[:, :, 1]  # n_lasers x npts_per_laser
        thetas = ray[:, :, 0]
        # raw waveforms is an iterator that iterates over an array of size
        # n_lasers x npts x waveform_resolution
        nlasers = ray.shape[0]  # self.raw_waveforms.shape[0]#.nlasers
        npts = ray.shape[1]  # self.raw_waveforms.npts
        final_dists = np.empty((nlasers, npts))
        final_intensities = np.empty((nlasers, npts))
        range_axis = self.raw_waveforms.range_axis
        final_hitmask = np.empty((nlasers, npts), dtype=bool)
        dsp = self.get_dsp(nlasers)
        for ilaser in range(nlasers):
            waveforms = self.raw_waveforms(ilaser)
            # waveform this laser is an array of size [npts x waveform_resolution]
            dsp[ilaser].noise_floor = 0.0  # set to 0 since it does not have the same meaning as usual
            waveforms = dsp[ilaser].process(waveforms)
            # threshold is already allpied in the dsp
            # here assume to detect targets with at least 0.01
            thresh = self.noise_floor_threshold
            if is_list_like(thresh):
                thresh = thresh[ilaser]
            for ipt, waveform in enumerate(waveforms):
                # cannot vectorize loop since find_peaks only works on 1D arrays...
                if np.isnan(waveform).any():
                    raise RuntimeError("there are nans in the waveform...")
                # idea from: https://stackoverflow.com/a/50365462/6362595
                # first find all rising edges
                rising_edges_idx = np.flatnonzero(
                        np.logical_and(
                            waveform[:-1] <= thresh,
                            waveform[1:] > thresh,
                            ))
                faling_edges_idx = np.flatnonzero(
                        np.logical_and(
                            waveform[:-1] > thresh,
                            waveform[1:] <= thresh,
                            ))
                if not len(rising_edges_idx) or not len(faling_edges_idx):
                    # no rising edge strong enough
                    final_intensities[ilaser, ipt] = 0.0
                    final_dists[ilaser, ipt] = 0.0
                    final_hitmask[ilaser, ipt] = False
                    continue
                # return the edge with highest intensity
                wheremax = np.argmax(waveform)
                # find closest to max on the right
                diff_rise = wheremax - rising_edges_idx
                diff_fall = faling_edges_idx - wheremax
                diff_rise[diff_rise < 0] = 1000000  # make negative values infinite so that we discard them
                diff_fall[diff_fall < 0] = 1000000
                closest_rise = diff_rise.argmin()
                closest_fall = diff_fall.argmin()
                # if closest_fall!=closest_rise:
                if range_axis[faling_edges_idx[closest_fall]] < range_axis[rising_edges_idx[closest_rise]]:
                    # no rising edge strong enough
                    final_intensities[ilaser, ipt] = 0.0
                    final_dists[ilaser, ipt] = 0.0
                    final_hitmask[ilaser, ipt] = False
                    continue
                power_integral = waveform[wheremax]
                dist_idx = rising_edges_idx[closest_rise]
                final_intensities[ilaser, ipt] = power_integral  # waveform[wheremax]
                final_hitmask[ilaser, ipt] = True
                final_dists[ilaser, ipt] = range_axis[dist_idx]
        self.raw_waveforms.clean()
        thetas, phis, rhos = concatenate_arrays(thetas, phis, final_dists)
        final_pc = np.array([thetas, phis, rhos]).T
        intensities, hitmask = concatenate_arrays(final_intensities, final_hitmask)
        return final_pc, intensities, hitmask


class FindPeaksPointCloudProcessor(BaseWaveformPointCloudProcessor):
    """Find peaks according to the find_peaks scipy method with multiple parameters."""
    algorithm_name = "Scipy's find_peaks"
    has_signal_processing = True

    def apply_thresholding(self):
        """Return the processed point cloud based on highest subray intensity.

        The subray with highest intensity will be picked, the others will be
        discarded.
        """
        # raw pc is an array of size [n_subrays, n_lasers, npts, 3]
        ray = self.raw_pc[self.upsampling_ratio ** 2 // 2]
        phis = ray[:, :, 1]  # n_lasers x npts_per_laser
        thetas = ray[:, :, 0]
        # raw waveforms is an iterator that iterates over an array of size
        # n_lasers x npts x waveform_resolution
        nlasers = self.raw_waveforms.nlasers
        npts = self.raw_waveforms.npts
        final_dists = np.empty((nlasers, npts))
        final_intensities = np.empty((nlasers, npts))
        range_axis = self.raw_waveforms.range_axis
        final_hitmask = np.empty((nlasers, npts), dtype=bool)
        dsp = self.get_dsp()
        for ilaser, waveforms in enumerate(self.raw_waveforms):
            # waveform this laser is an array of size [npts x waveform_resolution]
            waveforms = dsp.process(waveforms)
            for ipt, waveform in enumerate(waveforms):
                # cannot vectorize loop since find_peaks only works on 1D arrays...
                if np.isnan(waveform).any():
                    raise RuntimeError("there are nans in the waveform...")
                if np.max(waveform) == 0.0:
                    peak_pos = []
                else:
                    peak_pos, _ = find_peaks(
                        waveform, height=self.height,
                        threshold=[self.min_threshold, self.max_threshold],
                        distance=self.distance,
                        prominence=[self.min_prominence, self.max_prominence],
                        width=[self.min_width, self.max_width],
                        wlen=self.wlen,
                        rel_height=self.rel_height,
                        plateau_size=[self.min_plateau_size, self.max_plateau_size],
                        )
                if len(peak_pos):
                    max_peak = np.argmax(waveform[peak_pos])
                    final_intensities[ilaser, ipt] = waveform[peak_pos[max_peak]]
                    final_dists[ilaser, ipt] = range_axis[peak_pos[max_peak]]
                    final_hitmask[ilaser, ipt] = True
                else:
                    final_intensities[ilaser, ipt] = 0.0
                    final_dists[ilaser, ipt] = 0.0
                    final_hitmask[ilaser, ipt] = False
        thetas, phis, rhos = concatenate_arrays(thetas, phis, final_dists)
        final_pc = np.array([thetas, phis, rhos]).T
        intensities, hitmask = concatenate_arrays(final_intensities, final_hitmask)
        return final_pc, intensities, hitmask


ALL_THRESHOLDING_PROCESSORS = [
        NoThresholdingPointCloudProcessor,
        MaxWaveformPointCloudProcessor,
        MaxSubrayIntensityPointCloudProcessor,
        MiddleRayPointCloudProcessor,
        FindPeaksPointCloudProcessor,
        RisingEdgePointCloudProcessor,
        RisingEdgeAndWidthPointCloudProcessor
        ]
NON_WAVEFORM_BASED_THRESHOLDING_PROCESSORS = {
        p.algorithm_name: p for p in ALL_THRESHOLDING_PROCESSORS if not p.waveform_based}
WAVEFORM_BASED_THRESHOLDING_PROCESSORS = {
        p.algorithm_name: p for p in ALL_THRESHOLDING_PROCESSORS if p.waveform_based}
THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS = {
        processor.algorithm_name: processor for processor in ALL_THRESHOLDING_PROCESSORS}
