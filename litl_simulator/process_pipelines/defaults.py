"""Default model values."""
from .intensity_models import INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS
from .noise_models import AVAILABLE_NOISE_PROCESSOR_CLS
from .thresholding_algorithms import THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS
from .waveform_modelling import ALL_WAVEFORM_PROCESSOR_CLS


# global parameters
GLOBAL_DEFAULTS = {
        # signal-to-noise ratio over ambiant noise
        # snr is shared with waveform model and intensity model
        "snr": 300.0,
        # ambiant light noise is shared the same way as snr
        "ambiant_light": False,
        # saturation is shared between intensities and thresholding method (if waveform)
        "saturation": 100.0,
        }


# Thresholding
THRESHOLDING_ALGORITHMS_DEFAULTS = {
        "algorithm": tuple(THRESHOLDING_ALGORITHM_2_THRESHOLDING_PROCESSOR_CLS.keys())[0],
        "noise_floor_threshold": 0.0,  # disabled by default and not used for find_peaks
        "gain": 0.0,  # disabled by default
        "gaussian_denoizer_sigma": 1.0,
        "digitization": 'float',
        "dsp_template": 'GaussTemplate',
        "correction_factor": 1.0,
        "height": 0.0,
        "min_threshold": 0.0,
        "max_threshold": 100.0,
        "distance": 1,
        "min_prominence": 0.0,
        "max_prominence": 100.0,
        "min_width": 0,
        "max_width": 2400,
        "wlen": 1000,
        "rel_height": 0.5,
        "min_plateau_size": 0,
        "max_plateau_size": 2400,
        }

# Intensity models
INTENSITY_MODELS_DEFAULTS = {
        "model": tuple(INTENSITY_MODEL_2_INTENSITY_PROCESSOR_CLS.keys())[0],
        "road_wetness_depth": 0.0,  # mm  (disabled by default)
        "road_thread_profile_depth": 1.2,  # mm
        "saturate_retro_reflectors": False,
        }

# Noise models
NOISE_MODELS_DEFAULTS = {
        "model": AVAILABLE_NOISE_PROCESSOR_CLS[0].model_name,
        "std": 6,  # mm
        }

# waveform models
WAVEFORM_MODELS_DEFAULTS = {
        "model": ALL_WAVEFORM_PROCESSOR_CLS[0].model_name,
        "tauH": 5,   # default for velodyne HDL64 (ns)
        # n points over range axis (~0.5cm precision if range ~100m)
        "waveform_resolution": 2400,
        "waveform_range": 120,  # in m
        "poissonize_signal": False,
        "waveform_min_dist": 1,  # in m
        "upsampling_ratio": 5,
        }
