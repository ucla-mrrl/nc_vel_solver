import numpy as np
from scipy import fftpack, signal
import pyfftw
from encoding_matrix import gen_encoding
import matplotlib.pyplot as plt

from datetime import datetime

from womm_phantom import gen_phantom_parts

def precompute_2dt(A, v_max=95.0, s_res=1.0, oversamp=1, randu=0, randv=1, offsets=(0,0), diam = (4, 6, 8, 10), signal = (1.0, 1.0)):
    pyfftw.interfaces.cache.enable()
    
    # Generate actual flow data
    mask_c, mag, v_c = gen_phantom_parts(s_res=s_res, oversamp=oversamp, diam=diam, signal=signal)

    n_frames = v_c.shape[0]
    size_0 = mag.shape[0]
    size_1 = mag.shape[1]

    vel = np.zeros((n_frames, size_0, size_1))
    mask = np.zeros((size_0, size_1))

    # Place velocities in the data set (with offsets if required)
    d1 = v_c.shape[1] // 2
    d2 = v_c.shape[2] // 2
    do1 = offsets[0]
    do2 = offsets[1]
    vel[:, d1 + do1:-d1 + do1, d2 + do2:-d2 + do2] = v_c
    mask[d1 + do2:-d1 + do2, d2 + do2:-d2 + do2] = mask_c
    
    mag *= signal[0]
    mag[np.abs(vel).sum(0) > 0] = signal[1]

    # Each vessel's mask
    mask_diff = np.zeros((5, size_0, size_1))
    mask_diff[0] = (mask > 0)
    mask_diff[1] = (mask == 1)
    mask_diff[2] = (mask == 2)
    mask_diff[3] = (mask == 3)
    mask_diff[4] = (mask == 4)

    # Rotate velocities if required
    theta = 2.0 * np.pi * randu
    phi = np.arccos(2.0 * randv - 1)

    v3 = np.zeros((3, vel.shape[0], vel.shape[1], vel.shape[2]), np.float32)

    v3[0] = np.cos(theta) * np.sin(phi) * vel
    v3[1] = np.sin(theta) * np.sin(phi) * vel
    v3[2] = np.cos(phi) * vel

    v3 *= v_max

    # Reference velocity in k-space
    k_v3 = np.zeros(v3.shape, np.complex64)
    for it in range(v3.shape[1]):
        for dir in range(3):
            k_v3[dir, it] = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(v3[dir, it])))


    # Reference masks in k-space
    k_mask = np.zeros(mask_diff.shape, np.complex64)
    for it in range(5):
        k_mask[it] = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(mask_diff[it])))

    # Simulate complex data
    data_shape = (A.shape[0], vel.shape[0], vel.shape[1], vel.shape[2])
    data = np.zeros(data_shape, np.complex64)
    for ie in range(A.shape[0]):
        phase = (v3 * A[ie][:, np.newaxis, np.newaxis, np.newaxis]).sum(0)
        data[ie] = mag[np.newaxis, np.newaxis, ...] * np.exp(1j * phase)

    # Reference data in k-space
    k_data = np.zeros(data.shape, np.complex64)
    for it in range(data.shape[0]):
        for ie in range(data.shape[1]):
            k_data[it, ie] = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(data[it, ie])))

    return (s_res/oversamp), v3, k_v3, k_mask, k_data

def subsample_precomputed_2dt(input_file, s_res = 1.0, SNR = 20, mask_thresh = 0.01):
    
    pyfftw.interfaces.cache.enable()
    
    df = np.load(input_file)
    
    sim_res = df['sim_res']
    oversamp = s_res / sim_res

    k_v3 = df['k_v3']
    k_mask = df['k_mask']
    k_data = df['k_data']

    d00 = int(k_v3.shape[-2] // 2 - k_v3.shape[-2] // 2 // oversamp)
    d01 = int(k_v3.shape[-2] // 2 + k_v3.shape[-2] // 2 // oversamp)
    d10 = int(k_v3.shape[-1] // 2 - k_v3.shape[-1] // 2 // oversamp)
    d11 = int(k_v3.shape[-1] // 2 + k_v3.shape[-1] // 2 // oversamp)

    new_size0 = d01 - d00
    new_size1 = d11 - d10


    # Subsample velocity for reference standard
    v3ss = np.zeros((k_v3.shape[0], k_v3.shape[1], new_size0, new_size1), np.float32)
    for it in range(k_v3.shape[1]):
        for dir in range(3):
            k = k_v3[dir, it, d00:d01, d10:d11] / (oversamp * oversamp)
            filt = np.outer(signal.hamming(k.shape[0], False),
                            signal.hamming(k.shape[1], False))
            k *= filt
            v3ss[dir, it] = np.real(pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k))))

    
    mask_diff_ss = np.zeros((k_mask.shape[0], new_size0, new_size1), np.float32)
    for it in range(5):
        k = k_mask[it, d00:d01, d10:d11] / (oversamp * oversamp)
        filt = np.outer(signal.hamming(k.shape[0], False),
                        signal.hamming(k.shape[1], False))
        k *= filt
        mask_diff_ss[it] = np.real(pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k))))

    mask_diff_ss[mask_diff_ss < mask_thresh] = 0.0
    mask_diff_ss[mask_diff_ss >= mask_thresh] = 1.0


    data_ss = np.zeros((k_data.shape[0], k_data.shape[1], new_size0, new_size1), np.complex64)
    for it in range(k_data.shape[0]):
        for ie in range(k_data.shape[1]):
            k = k_data[it, ie, d00:d01, d10:d11] / (oversamp * oversamp)
            filt = np.outer(signal.hamming(k.shape[0], False),
                            signal.hamming(k.shape[1], False))
            k += 2 * np.sqrt(k.shape[0] * k.shape[1]) / SNR * (np.random.randn(k.shape[0], k.shape[1]) + 1j * np.random.randn(k.shape[0], k.shape[1]))
            k *= filt
            data_ss[it, ie] = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k)))

    return v3ss, mask_diff_ss, data_ss

def gendata_2dt(A, SNR=20, v_max=95.0, s_res=1.0, oversamp=8, randu=0, randv=1, offsets=(0,0), mask_thresh=0.01, diam = (4, 6, 8, 10), filt_strength=2):
    
    t_start = datetime.now()

    pyfftw.interfaces.cache.enable()
    
    mask_c, mag, v_c = gen_phantom_parts(s_res=s_res, oversamp=oversamp, diam=diam)

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    n_frames = v_c.shape[0]
    size_0 = mag.shape[0]
    size_1 = mag.shape[1]

    vel = np.zeros((n_frames, size_0, size_1))
    mask = np.zeros((size_0, size_1))

    d1 = v_c.shape[1] // 2
    d2 = v_c.shape[2] // 2
    do1 = offsets[0]
    do2 = offsets[1]
    vel[:, d1 + do1:-d1 + do1, d2 + do2:-d2 + do2] = v_c
    mask[d1 + do2:-d1 + do2, d2 + do2:-d2 + do2] = mask_c

    mask_diff = np.zeros((5, size_0, size_1))
    mask_diff[0] = (mask > 0)
    mask_diff[1] = (mask == 1)
    mask_diff[2] = (mask == 2)
    mask_diff[3] = (mask == 3)
    mask_diff[4] = (mask == 4)

    theta = 2.0 * np.pi * randu
    phi = np.arccos(2.0 * randv - 1)

    v3 = np.zeros((3, vel.shape[0], vel.shape[1], vel.shape[2]), np.float32)

    v3[0] = np.cos(theta) * np.sin(phi) * vel
    v3[1] = np.sin(theta) * np.sin(phi) * vel
    v3[2] = np.cos(phi) * vel

    v3 *= v_max

    d00 = v3.shape[-2] // 2 - v3.shape[-2] // 2 // oversamp
    d01 = v3.shape[-2] // 2 + v3.shape[-2] // 2 // oversamp
    d10 = v3.shape[-1] // 2 - v3.shape[-1] // 2 // oversamp
    d11 = v3.shape[-1] // 2 + v3.shape[-1] // 2 // oversamp

    v3ss = np.zeros((v3.shape[0], v3.shape[1], v3.shape[2] // oversamp, v3.shape[3] // oversamp), np.float32)

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    for it in range(v3.shape[1]):
        # print(it)
        for dir in range(3):
            k = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(v3[dir, it])))
            k = k[d00:d01, d10:d11] / (oversamp * oversamp)
            filt = np.outer(signal.gaussian(k.shape[0], k.shape[0] / filt_strength, False),
                            signal.gaussian(k.shape[1], k.shape[1] / filt_strength, False))
            # filt = np.outer(signal.hamming(k.shape[0], False),
                            # signal.hamming(k.shape[1], False))
            k *= filt
            v3ss[dir, it] = np.real(pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k))))

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    mask_diff_ss = np.zeros((mask_diff.shape[0], mask_diff.shape[1] // oversamp, mask_diff.shape[2] // oversamp), np.float32)
    for it in range(5):
        k = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(mask_diff[it])))
        k = k[d00:d01, d10:d11] / (oversamp * oversamp)
        # filt = np.outer(signal.gaussian(k.shape[0], k.shape[0] / 6, False),
                        # signal.gaussian(k.shape[1], k.shape[1] / 6, False))
        filt = np.outer(signal.hamming(k.shape[0], False),
                        signal.hamming(k.shape[1], False))
        k *= filt
        mask_diff_ss[it] = np.real(pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k))))

    mask_diff_ss[mask_diff_ss < mask_thresh] = 0.0
    mask_diff_ss[mask_diff_ss >= mask_thresh] = 1.0

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    data_shape = (A.shape[0], vel.shape[0], vel.shape[1], vel.shape[2])
    data = np.zeros(data_shape, np.complex64)
    for ie in range(A.shape[0]):
        phase = (v3 * A[ie][:, np.newaxis, np.newaxis, np.newaxis]).sum(0)
        data[ie] = mag[np.newaxis, np.newaxis, ...] * np.exp(1j * phase)

    d00 = data.shape[-2] // 2 - data.shape[-2] // 2 // oversamp
    d01 = data.shape[-2] // 2 + data.shape[-2] // 2 // oversamp
    d10 = data.shape[-1] // 2 - data.shape[-1] // 2 // oversamp
    d11 = data.shape[-1] // 2 + data.shape[-1] // 2 // oversamp

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    data_ss = np.zeros((data.shape[0], data.shape[1], data.shape[2] // oversamp, data.shape[3] // oversamp), np.complex64)
    phase_ss = np.zeros((data.shape[0], data.shape[1], data.shape[2] // oversamp, data.shape[3] // oversamp), np.float32)
    for it in range(data.shape[0]):
        # print(it)
        for ie in range(data.shape[1]):
            k = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.fftshift(data[it, ie])))
            k = k[d00:d01, d10:d11] / (oversamp * oversamp)
            k_ref = k.copy()
            filt = np.outer(signal.gaussian(k.shape[0], k.shape[0] / filt_strength, False),
                            signal.gaussian(k.shape[1], k.shape[1] / filt_strength, False))
            # filt = np.outer(signal.hamming(k.shape[0], False),
                            # signal.hamming(k.shape[1], False))
            k += np.sqrt(2) * np.sqrt(k.shape[0] * k.shape[1]) / SNR * (
                np.random.randn(k.shape[0], k.shape[1]) + 1j * np.random.randn(k.shape[0], k.shape[1]))
            k *= filt
            k_ref *= filt
            data_ss[it, ie] = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k)))
            phase_ss[it, ie] = np.angle(pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.fftshift(k_ref))))

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    phase_ss_uw = np.zeros((data.shape[0], data.shape[1], data.shape[2] // oversamp, data.shape[3] // oversamp), np.float32)
    for ie in range(A.shape[0]):
        phase_ss_uw[ie] = (v3ss * A[ie][:, np.newaxis, np.newaxis, np.newaxis]).sum(0)

    wraps = np.round( (phase_ss_uw-phase_ss)/(2*np.pi) )
    phase_ss2 = phase_ss + 2*np.pi * wraps

    v3ss2 = np.zeros_like(v3ss)

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)

    Ap = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    for ie in range(Ap.shape[0]):
        v3ss2[ie] = (phase_ss2 * Ap[ie][:, np.newaxis, np.newaxis, np.newaxis]).sum(0)

    t_stop = datetime.now()
    t_elapsed = t_stop - t_start
    print(t_elapsed.seconds + 1e-6*t_elapsed.microseconds)


    # Return stuff here and this is done !
    return data_ss, v3ss, v3ss2, mask_diff_ss, mask_diff, v3


if __name__ == "__main__":
    venc = 50
    A0 = gen_encoding('6pt-Zwart', True)
    A = A0 * np.pi / venc
    data_ss, v0_ss, mask_ss = gendata_2dt(A, s_res =2.0, oversamp = 4)
    for i in range(4):
        plt.figure()
        plt.imshow(mask_ss[i])