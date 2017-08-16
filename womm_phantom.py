import numpy as np
from scipy import special
import matplotlib.pyplot as plt


def pulsatile_flow(r, p0, pn, phi, timestep, grid, ru=1060, mu=.0035, freq=1.5):
    """Generate 2D cine vessel flow

    Most of this is copied from code I found on Matlab central, it needs to be cleaned up, a lot of this is slow and
    redundant, but it seems to work properly.  Comment better when I rewrite this.
    """
    ofst = int(np.round(grid / 2))
    rxl = int(np.round(3 * ofst / 4))
    h = r / rxl
    nw = pn.size
    omega = 2 * np.pi * freq
    u = np.zeros((timestep, grid, grid))
    zt = np.zeros(timestep + 1, np.complex)
    alpha = r * np.sqrt(omega * ru / mu)
    kapa = alpha * 1j ** 1.5 / r

    snw = nw * (nw + 1) / 2
    # alpha = alpha * np.sqrt(snw)
    for k in range(timestep):
        t = (k + 1) / timestep / freq
        for l in range(nw):
            zt[k] += pn[l] * np.exp(1j * (omega * t * (l + 1) - phi[l]))

    CJA = special.jv(0, kapa * r)
    for m in range(-rxl, rxl):
        for n in range(-rxl, rxl):
            for k in range(timestep):
                ri = np.sqrt(m ** 2 + n ** 2)
                if ri * h < r:
                    CBJ0 = special.jv(0, kapa * h * ri)
                    u[k, m + ofst, n + ofst] = p0 * ((ri * h) ** 2 - r ** 2) / 4 / mu + np.real(
                        1j / ru / omega / snw * (1 - CBJ0 / CJA) * zt[k])

    return u / u.max()


def gen_vessel(diam, s_res, oversamp, n_frames):
    """Creates a 2D cine flow profile with more useable input parameters

    Uses some pre-defined fudge factors to get a more or less physiological pressure waveform.  The waveform is highly
    simplified as an offset sine wave, which is hopefully realistic enough for our purposes.

    Args:
        diam: diameter of the vessel in mm
        s_res: desired spatial resolution of the grid in mm
        oversamp: oversampling used for proper grid size
        n_frames: number of timesteps

    Returns:
        A [time,y,x] array containing the velocity in the vessel normalized to a peak vel of 1.0
    """
    r = diam / 2000

    p0 = -1.5 / diam
    pn = np.array([-.1 * (diam ** 1.5)])
    phi = 3.021 * (diam ** (-1.187))
    phi = np.array([phi])

    grid = np.ceil(diam / 0.75 / s_res)
    grid = int(grid + (oversamp - grid % oversamp))
    u = pulsatile_flow(r, p0, pn, phi, n_frames, grid)

    return u


def gen_phantom(signal = (1.0, 1.0), s_res = .7, oversamp = 8, diam = (2, 4, 6, 8), n_frames = 20, offset=0):
    """Generate an oversampled magnitude and cine velocity profile for a set of vessels.

    Args:
        signal: tuple of signal in (static, flowing) tissues
        s_res: spatial resolution (without oversampling)
        oversamp: spatial resolution oversampling factor
        diam: list of vessel diameters in mm
        n_frames: number of time frames
        offset: oversampled offset of vessels (for multiple trials and possible intravoxel dephasing differences)

    Returns:
        mask: [y,x] array of ints specifying vessel identifiers
        mag: [y,x] array of magnitudes, defined by signal input
        vel: [t,y,x] array of velocities, normalized to have a peak velocity of 1.0
    """
    all_ves = []
    for d in diam:
        ves = gen_vessel(d, s_res/oversamp, oversamp, n_frames)
        all_ves.append(ves)

    total_width = 0
    max_height = 0
    for ves in all_ves:
        total_width += ves.shape[1]
        max_height = max(max_height, ves.shape[2])

    velv = np.zeros((n_frames, total_width, max_height))
    mask_crop = np.zeros((total_width, max_height))

    x_loc = 0
    ves_count = 1
    for ves in all_ves:
        velv[:, x_loc:x_loc+ves.shape[1], int(max_height/2 - ves.shape[2]/2):int(max_height/2 + ves.shape[2]/2)] = ves
        ves_mask = np.zeros((ves.shape[1], ves.shape[2]))
        ves_mask[np.abs(ves).sum(0) > 0] = ves_count
        mask_crop[x_loc:x_loc + ves.shape[1], int(max_height / 2 - ves.shape[2] / 2):int(max_height / 2 + ves.shape[2] / 2)] = ves_mask
        x_loc += ves.shape[1]
        ves_count += 1

    vel = np.zeros((n_frames, int(2*total_width), int(2*max_height)))
    mag = np.zeros(( int(2*total_width), int(2*max_height)))
    mask = np.zeros(( int(2*total_width), int(2*max_height)))

    d1 = velv.shape[1]//2
    d2 = velv.shape[2]//2
    do = offset
    vel[:,d1+do:-d1+do,d2+do:-d2+do] = velv
    mask[d1+do:-d1+do,d2+do:-d2+do] = mask_crop

    [xx, yy] = np.meshgrid(np.linspace(-1, 1, mag.shape[0]), np.linspace(-1, 1, mag.shape[1]), indexing='ij')
    rad = xx * xx + yy * yy
    mag[rad < .90] = signal[0]
    mag[np.abs(vel).sum(0) > 0] = signal[1]

    return mask, mag, vel


def gen_phantom_parts(signal = (1.0, 1.0), s_res = .7, oversamp = 8, diam = (4, 6, 8, 10), n_frames = 20):
    """Generate an oversampled magnitude and cine velocity profile for a set of vessels.

    Args:
        signal: tuple of signal in (static, flowing) tissues
        s_res: spatial resolution (without oversampling)
        oversamp: spatial resolution oversampling factor
        diam: list of vessel diameters in mm
        n_frames: number of time frames
        offset: oversampled offset of vessels (for multiple trials and possible intravoxel dephasing differences)

    Returns:
        mask: [y,x] array of ints specifying vessel identifiers
        mag: [y,x] array of magnitudes, defined by signal input
        vel: [t,y,x] array of velocities, normalized to have a peak velocity of 1.0
    """
    all_ves = []
    for d in diam:
        ves = gen_vessel(d, s_res/oversamp, oversamp, n_frames)
        all_ves.append(ves)

    total_width = 0
    max_height = 0
    for ves in all_ves:
        total_width += ves.shape[1]
        max_height = max(max_height, ves.shape[2])

    velv = np.zeros((n_frames, total_width, max_height))
    mask_crop = np.zeros((total_width, max_height))

    x_loc = 0
    ves_count = 1
    for ves in all_ves:
        velv[:, x_loc:x_loc+ves.shape[1], int(max_height/2 - ves.shape[2]/2):int(max_height/2 + ves.shape[2]/2)] = ves
        ves_mask = np.zeros((ves.shape[1], ves.shape[2]))
        ves_mask[np.abs(ves).sum(0) > 0] = ves_count
        mask_crop[x_loc:x_loc + ves.shape[1], int(max_height / 2 - ves.shape[2] / 2):int(max_height / 2 + ves.shape[2] / 2)] = ves_mask
        x_loc += ves.shape[1]
        ves_count += 1

    mag = np.zeros(( int(2*total_width), int(2*max_height)))

    [xx, yy] = np.meshgrid(np.linspace(-1, 1, mag.shape[0]), np.linspace(-1, 1, mag.shape[1]), indexing='ij')
    rad = xx * xx + yy * yy
    mag[rad < .90] = 1.0

    return mask_crop, mag, velv

if __name__ == "__main__":
    # mask, mag, v = gen_phantom()
    # plt.plot(v[5,:,:])
    # plt.show()
    diam = 8
    for phi in np.linspace(0.2,0.35,10):
        ves = gen_vessel(diam, 0.5 / 8, 8, 20, np.array([phi]))
        # plt.figure()
        # plt.plot(ves.mean((1, 2)))
        print(phi)
        print(np.argmax(ves.max((1, 2))))
        print(' ')

    for phi in np.linspace(1.14,1.16,3):
        ves = gen_vessel(diam, 0.5 / 8, 8, 200, np.array([phi]))
        # plt.figure()
        # plt.plot(ves.mean((1, 2)))
        print(phi)
        print(np.argmax(ves.max((1, 2)))/10)
        print(' ')









