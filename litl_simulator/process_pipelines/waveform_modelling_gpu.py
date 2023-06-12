"""Waveform modelling functions for gpu."""
try:
    import cupy as cp
    CUPY_AVAIL = True
    cupy_err = None
except ImportError as cupy_err:
    CUPY_AVAIL = False
    cupy_err = cupy_err
    print("cupy not working...")
import math
from numba import cuda


def to_cuda(x):
    """Sends data to gpu."""
    if isinstance(x, np.ndarray):
        return cuda.to_device(x.astype(np.float64))
    elif isinstance(x, float) or isinstance(x, int):
        return cuda.to_device(np.asarray([x]).astype(np.float64))
    else:
        raise NotImplementedError


@cuda.jit
def waveform_kernel_sin(waveform,
                        processed_pc,
                        processed_intensities,
                        weights,
                        processed_ambiant_light,
                        range_axis,
                        snr,
                        sin2cst,
                        ctauH,
                        iLaser,
                        ):
    """Waveform modelling main function on gpu.

    waveform : [npts=875, waveform_res=2400]

    processed_pc : [nSubrays=25, nLasers=128, npts=875, dims=4]

    ctauH is preferred parameter, sin2cst
    """
    dist_idx, r_idx = cuda.grid(2)

    # get values for iLaser
    if len(snr) > 1:
        snr_in = snr[iLaser]
    else:
        snr_in = snr[0]

    if len(ctauH) > 1:
        ctauH_in = ctauH[iLaser]
    else:
        ctauH_in = ctauH[0]

    if len(sin2cst) > 1:
        sin2cst_in = sin2cst[iLaser]
    else:
        sin2cst_in = sin2cst[0]
    #  get values for iLaser
    if len(snr) > 1:
        snr_in = snr[iLaser]
    else:
        snr_in = snr[0]

    # extract params according to laser index
    if len(ctauH) > 1:
        ctauH_in = ctauH[iLaser]
    else:
        ctauH_in = ctauH[0]

    if len(sin2cst) > 1:
        sin2cst_in = sin2cst[iLaser]
    else:
        sin2cst_in = sin2cst[0]

    if dist_idx < waveform.shape[0] and r_idx < waveform.shape[1]:
        tmp = 0
        for subray_idx in range(processed_pc.shape[0]):  # 25 -> Loop over subrays and directly sum

            subray, intensity, weight, ambiant = (
                    processed_pc[subray_idx], processed_intensities[subray_idx],
                    weights[subray_idx], processed_ambiant_light[subray_idx]
                    )
            intensities = intensity[iLaser]
            dists = subray[iLaser, :, 2]
            new_intens = ambiant[iLaser, dist_idx]
            cur_dist = dists[dist_idx]
            cur_int = intensities[dist_idx]
            if (cur_dist <= range_axis[r_idx]) and (range_axis[r_idx] <= cur_dist + ctauH_in):
                new_intens += weight * snr_in * cur_int * math.sin(sin2cst_in * (range_axis[r_idx] - cur_dist)) ** 2
            tmp += new_intens
            waveform[dist_idx, r_idx] = tmp


@cuda.jit
def waveform_kernel_cos(waveform,
                        processed_pc,
                        processed_intensities,
                        weights,
                        processed_ambiant_light,
                        range_axis,
                        snr,
                        sin2cst,
                        ctauH,
                        iLaser,
                        ):
    """Waveform modelling main function on gpu.

    waveform : [npts=875, waveform_res=2400]

    processed_pc : [nSubrays=25, nLasers=128, npts=875, dims=4]

    ctauH is preferred parameter, sin2cst
    """
    dist_idx, r_idx = cuda.grid(2)
    # get values for iLaser
    if len(snr) > 1:
        snr_in = snr[iLaser]
    else:
        snr_in = snr[0]

    if len(ctauH) > 1:
        ctauH_in = ctauH[iLaser]
    else:
        ctauH_in = ctauH[0]

    if len(sin2cst) > 1:
        sin2cst_in = sin2cst[iLaser]
    else:
        sin2cst_in = sin2cst[0]
    #  get values for iLaser
    if len(snr) > 1:
        snr_in = snr[iLaser]
    else:
        snr_in = snr[0]

    # extract params according to laser index
    if len(ctauH) > 1:
        ctauH_in = ctauH[iLaser]
    else:
        ctauH_in = ctauH[0]

    if len(sin2cst) > 1:
        sin2cst_in = sin2cst[iLaser]
    else:
        sin2cst_in = sin2cst[0]

    if dist_idx < waveform.shape[0] and r_idx < waveform.shape[1]:
        tmp = 0
        for subray_idx in range(processed_pc.shape[0]):  # 25 -> Loop over subrays and directly sum

            subray, intensity, weight, ambiant = (
                    processed_pc[subray_idx], processed_intensities[subray_idx],
                    weights[subray_idx], processed_ambiant_light[subray_idx]
                    )
            intensities = intensity[iLaser]
            dists = subray[iLaser, :, 2]
            new_intens = ambiant[iLaser, dist_idx]
            cur_dist = dists[dist_idx]
            cur_int = intensities[dist_idx]
            if (cur_dist - ctauH_in / 2 <= range_axis[r_idx]) and (range_axis[r_idx] <= cur_dist + ctauH_in / 2):
                new_intens += weight * snr_in * cur_int * math.cos(sin2cst_in * (range_axis[r_idx] - cur_dist)) ** 2
            tmp += new_intens
            waveform[dist_idx, r_idx] = tmp


def waveform_with_poissson_cos(waveform,
                               processed_pc,
                               processed_intensities,
                               weights,
                               processed_ambiant_light,
                               range_axis,
                               snr,
                               sin2cst,
                               ctauH,
                               iLaser,
                               blocks_per_grid,
                               threads_per_block,
                               ):
    """Waveform modelling function on gpu with poisson sampling."""
    waveform_kernel_cos[blocks_per_grid, threads_per_block](waveform,
                                                            processed_pc,
                                                            processed_intensities,
                                                            weights,
                                                            processed_ambiant_light,
                                                            range_axis,
                                                            snr,
                                                            sin2cst,
                                                            ctauH,
                                                            iLaser,
                                                            )
    # Convert to torch -> both default to 0 GPU
    if not CUPY_AVAIL:
        raise cupy_err
    waveform_cupy = cp.asarray(waveform)
    waveform_cupy_poisson = cp.random.poisson(100*waveform_cupy)/100.0
    return cp.asnumpy(waveform_cupy_poisson)


def waveform_without_poissson_cos(waveform,
                                  processed_pc,
                                  processed_intensities,
                                  weights,
                                  processed_ambiant_light,
                                  range_axis,
                                  snr,
                                  sin2cst,
                                  ctauH,
                                  iLaser,
                                  blocks_per_grid,
                                  threads_per_block,
                                  ):
    """Waveform modelling function on gpu without poisson sampling."""
    if not CUPY_AVAIL:
        raise cupy_err
    waveform_kernel_cos[blocks_per_grid, threads_per_block](waveform,
                                                            processed_pc,
                                                            processed_intensities,
                                                            weights,
                                                            processed_ambiant_light,
                                                            range_axis,
                                                            snr,
                                                            sin2cst,
                                                            ctauH,
                                                            iLaser,
                                                            )
    return waveform.copy_to_host()


def waveform_with_poissson_sin(waveform,
                               processed_pc,
                               processed_intensities,
                               weights,
                               processed_ambiant_light,
                               range_axis,
                               snr,
                               sin2cst,
                               ctauH,
                               iLaser,
                               blocks_per_grid,
                               threads_per_block,
                               ):
    """Waveform modelling function on gpu with poisson sampling."""
    waveform_kernel_sin[blocks_per_grid, threads_per_block](waveform,
                                                            processed_pc,
                                                            processed_intensities,
                                                            weights,
                                                            processed_ambiant_light,
                                                            range_axis,
                                                            snr,
                                                            sin2cst,
                                                            ctauH,
                                                            iLaser,
                                                            )
    # Convert to torch -> both default to 0 GPU
    if not CUPY_AVAIL:
        raise cupy_err
    waveform_cupy = cp.asarray(waveform)
    waveform_cupy_poisson = cp.random.poisson(waveform_cupy*100)/100.0
    return cp.asnumpy(waveform_cupy_poisson)


def waveform_without_poissson_sin(waveform,
                                  processed_pc,
                                  processed_intensities,
                                  weights,
                                  processed_ambiant_light,
                                  range_axis,
                                  snr,
                                  sin2cst,
                                  ctauH,
                                  iLaser,
                                  blocks_per_grid,
                                  threads_per_block,
                                  ):
    """Waveform modelling function on gpu without poisson sampling."""
    if not CUPY_AVAIL:
        raise cupy_err
    waveform_kernel_sin[blocks_per_grid, threads_per_block](waveform,
                                                            processed_pc,
                                                            processed_intensities,
                                                            weights,
                                                            processed_ambiant_light,
                                                            range_axis,
                                                            snr,
                                                            sin2cst,
                                                            ctauH,
                                                            iLaser,
                                                            )
    return waveform.copy_to_host()


if __name__ == "__main__":

    import pickle
    import copy
    import time
    import numpy as np
    from scipy.constants import c

    with open('/lhome/dscheub/ObjectDetection/lidar_carla_sim/tests/waveform_modelling.pkl', 'rb') as f:
        data = pickle.load(f)

    function_inputs = copy.deepcopy(data)
    function_inputs.pop('mainray')
    function_inputs.pop('npts')
    function_inputs['processed_ambiant_light'] = copy.deepcopy(function_inputs['processed_ambient_light'])
    function_inputs.pop('processed_ambient_light')

    num_subrays, num_lasers, npts, dims = function_inputs['processed_pc'].shape
    waveform_res = 2400
    print(f'{num_lasers=}, {num_subrays=}, {npts=}, {waveform_res=}')

    tauH = 5
    ctauH_cpu = c * tauH * 1e-9  # tau in ns
    ctauH_cuda = cuda.to_device(np.asarray([ctauH_cpu]))
    sin2cst_cpu = np.pi / ctauH_cpu
    sin2cst_cuda = cuda.to_device(np.asarray([sin2cst_cpu]))
    snr_cpu = 300
    # snr_cuda = cuda.to_device(np.asarray([snr_cpu]))
    snr_cuda = cuda.to_device(snr_cpu * np.ones(128))

    # iLaser = 10

    threads_per_block = (25, 24)
    blocks_per_grid_x = int(math.ceil(npts / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(waveform_res / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for k, v in function_inputs.items():
        if isinstance(v, np.ndarray):
            print(f'{k}, {v.shape}')

    # print(f'{processed_pc_cuda.nbytes / 1e6 =}')
    # print(f'{processed_intensities_cuda.nbytes / 1e6=}')
    # print(f'{weights_cuda.nbytes / 1e6 =}')
    # print(f'{range_axis_cuda.nbytes / 1e6=}')

    processed_pc_cuda = cuda.to_device(function_inputs['processed_pc'].astype(np.float64))
    processed_intensities_cuda = cuda.to_device(function_inputs['processed_intensities'].astype(np.float64))
    weights_cuda = cuda.to_device(function_inputs['weights'].astype(np.float64))
    processed_ambiant_light_cuda = cuda.to_device(function_inputs['processed_ambiant_light'].astype(np.float64))
    range_axis_cuda = cuda.to_device(function_inputs['range_axis'].astype(np.float64))
    waveform_numba_cuda = cuda.to_device(np.zeros((npts, waveform_res)).astype(np.float64))

    waveform_with_poisson_numpy = waveform_with_poissson_sin(waveform_numba_cuda,
                                                             processed_pc_cuda,
                                                             processed_intensities_cuda,
                                                             weights_cuda,
                                                             processed_ambiant_light_cuda,
                                                             range_axis_cuda,
                                                             snr_cuda,
                                                             sin2cst_cuda,
                                                             ctauH_cuda,
                                                             10,
                                                             blocks_per_grid,
                                                             threads_per_block)

    for iLaser in range(128):

        start = time.time()

        waveform_with_poisson_numpy = waveform_with_poissson_sin(waveform_numba_cuda,
                                                                 processed_pc_cuda,
                                                                 processed_intensities_cuda,
                                                                 weights_cuda,
                                                                 processed_ambiant_light_cuda,
                                                                 range_axis_cuda,
                                                                 snr_cuda,
                                                                 sin2cst_cuda,
                                                                 ctauH_cuda,
                                                                 iLaser,
                                                                 blocks_per_grid,
                                                                 threads_per_block)
        print('test')
        print(f'Elapsed time {time.time() - start}')
        print(f'{waveform_with_poisson_numpy=}')
