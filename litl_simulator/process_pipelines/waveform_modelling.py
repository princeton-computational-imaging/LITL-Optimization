"""Processors that create the waveforms."""
import numpy as np
from scipy.constants import c
from numba import cuda

from .bases import BaseLidarProcessor
from .utils import is_list_like
from .waveform_modelling_gpu import waveform_with_poissson_sin as sin_squared_generator_with_poisson_gpu
from .waveform_modelling_gpu import waveform_without_poissson_sin as sin_squared_generator_without_poisson_gpu
from .waveform_modelling_gpu import waveform_with_poissson_cos as cos_squared_generator_with_poisson_gpu
from .waveform_modelling_gpu import waveform_without_poissson_cos as cos_squared_generator_without_poisson_gpu
from ..bases import BaseUtility
from ..settings import USE_GPU, GPU_THREADS_PER_BLOCK


class WaveformsIterator(BaseUtility):
    """Waveform generator iterator.

    When iterated over, it generated a waveform based on a given function.
    """

    def __init__(self, *args, **kwargs):
        """Waveforms iterator init method."""
        super().__init__(*args, **kwargs)
        self.nlasers = None
        self.npts = None
        self.waveform_generator_callable = None

    @property
    def range_axis(self):
        """The range axis."""
        return self.waveform_generator_callable.range_axis

    def __call__(self, *args, **kwargs):
        """Return the call to the callable."""
        return self.waveform_generator_callable(*args, **kwargs)

    def __iter__(self):
        """Iterates over waveforms one laser at a time."""
        if self.waveform_generator_callable is None:
            raise ValueError("No function to generate waveform given.")
        for ilaser in range(self.nlasers):
            yield self.waveform_generator_callable(ilaser)
        # once everything is done, clean data if necessary
        self.waveform_generator_callable.clean()

    def __len__(self):
        """The number of lasers."""
        if self.nlasers is None:
            raise ValueError("nlasers not set.")
        return self.nlasers


class BaseWaveformModelProcessor(BaseLidarProcessor):
    """Base class for waveform model processors."""
    model_name = None

    def __init__(
            self,
            framedir=None,
            tauH=None, waveform_range=None,
            waveform_resolution=None,
            waveform_min_dist=None,
            model=None,
            poissonize_signal=None,
            snr=None,  # signal to noise ratio
            upsampling_ratio=3,
            **kwargs):
        """Base waveform model processor init method."""
        super().__init__(framedir=framedir, **kwargs)
        if self.model_name is None:
            raise ValueError("No model name...")
        if model != self.model_name:
            print(f'{model} is not equal {self.model_name}')
            raise RuntimeError(model)
        self.tauH = tauH
        self.waveform_range = waveform_range
        self.waveform_min_distance = waveform_min_dist
        self.waveform_resolution = waveform_resolution
        self.poissonize_signal = poissonize_signal
        self.snr = snr
        self.raw_ambiant_light = None
        self.range_axis = np.linspace(self.waveform_min_distance, self.waveform_range, self.waveform_resolution)
        self.upsampling_ratio = upsampling_ratio

    def __call__(self, *args, **kwargs):
        """Call this processor to generate the waveforms."""
        raise NotImplementedError

    def process(self):
        """Generate waveform."""
        # rearrage data into subrays
        # self.processed_waveforms = WaveformsIterator(loglevel=self._loglevel)
        # self.processed_waveforms.waveform_generator_callable = lambda x: self.__call__(self, x)

        (self.processed_pc, self.processed_intensities,
         self.processed_ambiant_light, self.hitmask) = (
                self.rearrange_data_in_subrays(
                    self.raw_pc, self.raw_intensities, self.raw_ambiant_light,
                    self.hitmask,
                    upsampling_ratio=self.upsampling_ratio,
                    )
                )
        # self.nsubrays = self.processed_ambiant_light.shape[0]
        # self.processed_ambiant_light /= self.nsubrays  # normalize over subrays as well
        # this would mean that sum_subrays average = 1
        # downsampled shape
        # self.processed_waveforms.nlasers = self.processed_pc.shape[1]
        # self.processed_waveforms.npts = self.processed_pc.shape[2]

    def set_raw_data(self, *args, ambiant_light=None, **kwargs):
        """Sets raw data."""
        super().set_raw_data(*args, **kwargs)
        if ambiant_light is not None and ambiant_light is not False:
            self.raw_ambiant_light = ambiant_light


class NoWaveformModelProcessor(BaseWaveformModelProcessor):
    """Processor class when there is no waveform generation.

    There is no generated waveforms and the processed pc corresponds to the
    raw pc given as input.
    """
    model_name = "None"

    def process(self):
        """Process the 'no waveform'."""
        self.processed_pc = self.raw_pc
        self.processed_intensities = self.raw_intensities
        self.processed_waveforms = None


class SinCosWaveformModelProcessor(BaseWaveformModelProcessor):
    """Waveform model of a cos^2 at each place where this is a point.

    Parameters:
    -----------
    tau_H:
        pulse width in ns
    waveform_range:
        maximal range for the waveform
    waveform_resolution:
        number of discretization pts for waveform array
    poissonize_signal:
        final waveform will be used as lambdas (1 lambda for each pt in waveform)
        of a poisson distribution (mimicks photon counts on a SPAD sensor).
    """
    pulse_func = None

    # NOTE on waveform generation:
    # W(t) ~ Poisson(f(t))  final waveform in function of t is poisson distributed
    # f(t) = sum_i(G_i * f_i(t))  signal is sum of sub signals (downsampling)
    # G_i is a weight that accounts for pulse shape profile (gaussian)
    # f_i(t) = ambiant_i + SNR * g_i(t)  where ambiant is normalized to 1 (on all pt cloud)
    # g_i(t) = 0 if t < R_i / c and t > R_i / c + tauH
    #        = I_i cos^2( pi/tauH (t - R_i / c) ) otherwise
    # R_i is original distance of raycast
    # I_i is intensity computed by a model and is normalized to 1 (on all pt cloud)
    # SNR is signal-to-noise ratio

    def __init__(self, *args, **kwargs):
        """Cos squared waveform generator init method."""
        if self.pulse_func is None:
            raise ValueError("No pulse func defined.")
        super().__init__(*args, **kwargs)
        self.weights = self._get_weights(self.upsampling_ratio)

    def _get_weights(self, upsampling_ratio):
        # gaussian weights where 1st neighbors = 1/2 and 2nd = 1/4 etc
        # in a table, they corresponds to 2**(-(dist ** 2)) where dist
        # is the euclidian distance from the main ray (at center)
        # where the distance between 2 adjacent ray is = 1
        dists = []
        mid = self.upsampling_ratio // 2
        for row in range(-mid, mid + 1):
            for col in range(-mid, mid + 1):
                dists.append(np.linalg.norm([row, col]))
        weights = np.power(2, - np.power(np.array(dists), 2))
        return weights / np.sum(weights)  # normalize the weights

    def __call__(self, ilaser, ipt=None, return_subwaveforms=False):
        """Generates the waveforms for all pts of given laser idx.

        Args:
            ilaser: int
                The laser index to compute the waveform for.
            ipt: int, optional
                If not None, gives the pt index to compute the waveform for.
                (downsampled pt). If None, waveform is computed for all pts.
            return_subwaveforms: bool, optional
                If True, the subwaveforms are returned as well.
        """
        mainray = self.processed_pc[self.upsampling_ratio ** 2 // 2]
        # subrays is n_subrays x n_lasers x npts_per_laser x nfeats
        # subrays_I is n_subrays x n_lasers x npts_per_laser
        npts = mainray.shape[1]
        if ipt is not None:
            npts = 1
        intensities = np.zeros(npts)
        ambiant = np.zeros(npts)  # in case it is none
        tauH = self.tauH
        if is_list_like(self.tauH):
            tauH = tauH[ilaser]
        snr = self.snr
        if is_list_like(snr):
            snr = snr[ilaser]
        ctauH = c * tauH * 1e-9  # tau in ns
        sin2cst = np.pi / ctauH
        big_axis = np.broadcast_to(self.range_axis, (npts, len(self.range_axis)))
        r = np.arange(self.waveform_resolution)
        tot_waveform = np.zeros((npts, self.waveform_resolution))
        sub_waveforms = []
        # iterate over subrays
        for iray, (subray, intensity, weight) in enumerate(zip(
                self.processed_pc, self.processed_intensities, self.weights,
                )):
            dists = subray[ilaser, :, 2]  # len = npts
            intensities = intensity[ilaser]  # len = npts
            if self.processed_ambiant_light is not None:
                ambiant = self.processed_ambiant_light[iray, ilaser]
            if ipt is not None:
                dists = np.array([dists[ipt]])
                intensities = np.array([intensities[ipt]])
                if self.processed_ambiant_light is not None:
                    ambiant = np.array([ambiant[ipt]])
                else:
                    ambiant = np.asarray([0])
            # add ambiant noise at the start
            subray_wvfrm = np.multiply(
                    ambiant[:, np.newaxis], np.ones((npts, self.waveform_resolution))
                    )
            # numpy magic taken from
            # https://stackoverflow.com/a/46734201/6362595
            # where_non_zero = np.digitize(dists - ctauH/2, big_axis[0])
            where_non_zero = np.digitize(dists, big_axis[0])
            # -1 below because this is the upper bound which we don't want to include
            # where_non_zero_ctau = np.digitize(dists + ctauH/2, big_axis[0]) - 1
            where_non_zero_ctau = np.digitize(dists + ctauH, big_axis[0]) - 1
            mask = (where_non_zero[:, None] <= r) & (where_non_zero_ctau[:, None] >= r)
            big_d = np.broadcast_to(
                    dists[:, np.newaxis],
                    (len(dists), self.waveform_resolution))[mask]
            big_i = np.broadcast_to(
                    intensities[:, np.newaxis],
                    (len(intensities), self.waveform_resolution))[mask]
            # weight only applies to actual signal
            subray_wvfrm[mask] += weight * snr * np.multiply(
                big_i,
                np.square(
                    self.pulse_func(sin2cst * (big_axis[mask] - big_d))))
            # subray_wvfrm *= weight
            sub_waveforms.append(subray_wvfrm)
        tot_waveform = np.sum(sub_waveforms, axis=0)
        if self.poissonize_signal:
            tot_waveform = np.random.poisson(
                    lam=tot_waveform, size=tot_waveform.shape,
                    ).astype(np.float64)
        # waveforms with no hits will be constant ambient noise
        if not return_subwaveforms:
            return tot_waveform
        return tot_waveform, np.asarray(sub_waveforms)

    def clean(self):
        """Clean data."""
        pass


class SinCosWaveformModelProcessorGPU(SinCosWaveformModelProcessor):
    """Samething as sin cos waveform model but on gpu."""
    def __init__(self, *args, **kwargs):
        """Sin/Cos squared waveform generator init method."""
        super().__init__(*args, **kwargs)
        self.weights_cuda = self._to_cuda(self._get_weights(self.upsampling_ratio))
        assert cuda.is_available(), 'Cannot detect GPU'

    def process(self, *args, **kwargs):
        """Process."""
        super().process(*args, **kwargs)
        self.npts = self.processed_pc.shape[2]
        self.threads_per_block = GPU_THREADS_PER_BLOCK
        blocks_per_grid_x = int(np.ceil(self.npts / self.threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(self.waveform_resolution / self.threads_per_block[1]))
        self.blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        if isinstance(self.tauH, list):
            ctauH = [c*i*1e-9 for i in self.tauH]
            sin2cst = (np.pi / np.asarray(ctauH)).tolist()
        else:
            ctauH = c * self.tauH * 1e-9  # tau in ns
            sin2cst = np.pi / ctauH
        self.processed_pc_cuda = self._to_cuda(self.processed_pc)
        self.processed_intensities_cuda = self._to_cuda(self.processed_intensities)
        if self.processed_ambiant_light is None:
            n_subrays, n_lasers = self.processed_intensities.shape[:2]
            self.processed_ambiant_light_cuda = self._to_cuda(np.zeros((n_subrays, n_lasers, self.npts)))
        else:
            self.processed_ambiant_light_cuda = self._to_cuda(self.processed_ambiant_light)
        self.range_axis_cuda = self._to_cuda(self.range_axis)
        self.snr_cuda = self._to_cuda(self.snr)
        self.sin2cst_cuda = self._to_cuda(sin2cst)
        self.ctauH_cuda = self._to_cuda(ctauH)

        self.waveform_numba_cuda = self._to_cuda(np.zeros((self.npts, self.waveform_resolution)))

    def _to_cuda(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = np.asarray(x)
        if isinstance(x, np.ndarray):
            return cuda.to_device(x.astype(np.float64))
        elif isinstance(x, float) or isinstance(x, int):
            return cuda.to_device(np.asarray([x]).astype(np.float64))
        else:
            raise NotImplementedError(str(x))

    def clean(self):
        """Clean gpu data."""
        del self.processed_pc_cuda
        del self.processed_intensities_cuda
        del self.processed_ambiant_light_cuda
        del self.range_axis_cuda
        del self.snr_cuda
        del self.sin2cst_cuda
        del self.ctauH_cuda
        del self.waveform_numba_cuda
        del self.weights_cuda
        import cupy as cp
        cp._default_memory_pool.free_all_blocks()

    def __call__(self, ilaser, ipt=None, return_subwaveforms=False):
        """Generates the waveforms for all pts of given laser idx.

        Args:
            ilaser: int
                The laser index to compute the waveform for.
            ipt: int, optional
                If not None, gives the pt index to compute the waveform for.
                (downsampled pt). If None, waveform is computed for all pts.
            return_subwaveforms: bool, optional
                If True, the subwaveforms are returned as well.
        """
        if return_subwaveforms:
            raise NotImplementedError
        if ipt is not None:
            raise NotImplementedError

        if False:
            import pickle
            save = dict(
                processed_pc=self.processed_pc,
                processed_intensities=self.processed_intensities,
                weights=self._get_weights(self.upsampling_ratio),
                processed_ambiant_light=self.processed_ambiant_light,
                range_axis=self.range_axis,
                snr=self.snr,
            )
            with open('waveform.pkl', 'wb') as f:
                pickle.dump(save, f)

        if self.pulse_func == "sin":
            if self.poissonize_signal:
                func = sin_squared_generator_with_poisson_gpu
            else:
                func = sin_squared_generator_without_poisson_gpu
        elif self.pulse_func == "cos":
            if self.poissonize_signal:
                func = cos_squared_generator_with_poisson_gpu
            else:
                func = cos_squared_generator_without_poisson_gpu
        else:
            raise NotImplementedError

        return func(self.waveform_numba_cuda,
                    self.processed_pc_cuda,
                    self.processed_intensities_cuda,
                    self.weights_cuda,
                    self.processed_ambiant_light_cuda,
                    self.range_axis_cuda,
                    self.snr_cuda,
                    self.sin2cst_cuda,
                    self.ctauH_cuda,
                    ilaser,
                    self.blocks_per_grid,
                    self.threads_per_block,
                    )


class SinSquaredWaveformModelProcessor(SinCosWaveformModelProcessor):
    """Waveform is made of sin^2 functions."""
    model_name = "SinSquared"
    pulse_func = np.sin


class CosSquaredWaveformModelProcessor(SinCosWaveformModelProcessor):
    """Waveform is made of cos^2 functions."""
    model_name = "CosSquared"
    pulse_func = np.cos


class CosSquaredWaveformModelProcessorGPU(SinCosWaveformModelProcessorGPU):
    """Cos squared on gpu."""
    model_name = "CosSquaredGPU"
    pulse_func = "cos"


class SinSquaredWaveformModelProcessorGPU(SinCosWaveformModelProcessorGPU):
    """Cos squared on gpu."""
    model_name = "SinSquaredGPU"
    pulse_func = "sin"


ALL_WAVEFORM_PROCESSOR_CLS = [
        NoWaveformModelProcessor,
        CosSquaredWaveformModelProcessor,
        SinSquaredWaveformModelProcessor,
        ]
if USE_GPU:
    ALL_WAVEFORM_PROCESSOR_CLS += [
            CosSquaredWaveformModelProcessorGPU,
            SinSquaredWaveformModelProcessorGPU,
            ]
WAVEFORM_MODEL_2_WAVEFORM_PROCESSOR_CLS = {
        cls.model_name: cls for cls in ALL_WAVEFORM_PROCESSOR_CLS}
