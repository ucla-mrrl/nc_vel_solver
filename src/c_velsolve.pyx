import numpy as np
cimport numpy as np
import time

cdef extern from "solver.cpp":
    void vel_solve(int Ne, int n_voxels, float complex *data_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out)
    
    void vel_solve_weighted(int Ne, int n_voxels, float complex *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out, int grid)
    
    void vel_solve_weighted_knownvel(int Ne, int n_voxels, float complex *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float *kv_in, float max_v, float *v_out)
    
    void vel_solve_weighted_knownwraps(int Ne, int n_voxels, float complex *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, int *kk_in, float max_v, float *v_out)

    void reg_v2(int Ne, int n_dims, int *dims, int grid,
            int *mask, float complex *data_in, float *weights_in, int *wraps_in,
            float *A_in, float max_v, float lam, float *vel, float *vel_out,
            float *debug)

def c_vel_solve(data, A, max_v, mask=None):
    start1 = time.time()

    if mask is None:
        mask = np.ones(data.shape[1:], np.int)
    cdef np.ndarray[int, ndim=1, mode="c"] mask_c = np.ascontiguousarray(np.ravel(mask), np.intc)

    data = np.moveaxis(data,0,-1)
    data_n = data/np.abs(data)

    norms = np.sum(np.abs(A)**2,axis=1)**(1./2)
    wraps = np.round(0.99 * norms * max_v / 2.0 / np.pi)

    print('wraps = %s' % wraps)

    dshape = np.array(data.shape)

    Ne = dshape[-1]
    n_voxels = np.prod(dshape[:-1])

    cdef np.ndarray[np.complex64_t, ndim=1, mode="c"] data_c = np.ascontiguousarray(np.ravel(data_n), np.complex64)
    cdef np.ndarray[int, ndim=1, mode="c"] wraps_c = np.ascontiguousarray(np.ravel(wraps), np.intc)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] A_c = np.ascontiguousarray(np.ravel(A), np.float32)


    vshape = dshape.copy()
    vshape[-1] = 3
    v = np.zeros(vshape, np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] v_c = np.ascontiguousarray(np.ravel(v), np.float32)

    start2 = time.time()
    vel_solve(Ne, n_voxels, &data_c[0], &wraps_c[0], &mask_c[0], &A_c[0], max_v, &v_c[0])
    end2 = time.time()

    v_out = np.reshape(v_c, vshape)
    v_out = np.moveaxis(v_out,-1,0)

    print('v_solve took total = %f and the cpp part took %f' % ( (time.time()-start1), (end2-start2) ))

    return v_out

def c_vel_solve_weighted(data, A, max_v, weights=None, wnorm=2.0, weighting=True, mask=None, known_vel=None, known_wraps=None, grid = 0):
    start1 = time.time()

    if mask is None:
        mask = np.ones(data.shape[1:], np.int)
    cdef np.ndarray[int, ndim=1, mode="c"] mask_c = np.ascontiguousarray(np.ravel(mask), np.intc)

    data = np.moveaxis(data,0,-1)
    data_n = data/np.abs(data)

    norms = np.sum(np.abs(A)**2,axis=1)**(1./2)
    wraps = np.round(0.99 * norms * max_v / 2.0 / np.pi)
    norms /= norms.min()

    # print('wraps = %s' % wraps)
    print('norms = %s' % norms)

    if (weights is None) and weighting:
        weights = (norms * np.abs(data)) ** wnorm
    elif (weights is None) and (not weighting):
        weights = np.ones(data.shape, np.float32)
    else:
        weights = weights ** wnorm

    # print('weights.shape = %s' % (weights.shape,))

    dshape = np.array(data.shape)

    Ne = dshape[-1]
    n_voxels = np.prod(dshape[:-1])

    cdef np.ndarray[np.complex64_t, ndim=1, mode="c"] data_c = np.ascontiguousarray(np.ravel(data_n), np.complex64)
    cdef np.ndarray[int, ndim=1, mode="c"] wraps_c = np.ascontiguousarray(np.ravel(wraps), np.intc)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] A_c = np.ascontiguousarray(np.ravel(A), np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] weights_c = np.ascontiguousarray(np.ravel(weights), np.float32)

    vshape = dshape.copy()
    vshape[-1] = 3
    v = np.zeros(vshape, np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] v_c = np.ascontiguousarray(np.ravel(v), np.float32)

    if known_vel is not None:
        known_vel = np.moveaxis(known_vel,0,-1)
    else:
        known_vel = np.zeros(1)

    if known_wraps is not None:
        known_wraps = np.moveaxis(known_wraps,0,-1)
    else:
        known_wraps = np.zeros(1)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] kv_c = np.ascontiguousarray(np.ravel(known_vel), np.float32)
    cdef np.ndarray[int, ndim=1, mode="c"] kk_c = np.ascontiguousarray(np.ravel(known_wraps), np.intc)

    if known_vel.size>1:
        start2 = time.time()
        vel_solve_weighted_knownvel(Ne, n_voxels, &data_c[0], &weights_c[0], &wraps_c[0], &mask_c[0], &A_c[0], &kv_c[0], max_v, &v_c[0])
        end2 = time.time()
    elif known_wraps.size>1:
        start2 = time.time()
        vel_solve_weighted_knownwraps(Ne, n_voxels, &data_c[0], &weights_c[0], &wraps_c[0], &mask_c[0], &A_c[0], &kk_c[0], max_v, &v_c[0])
        end2 = time.time()
    else:
        start2 = time.time()
        vel_solve_weighted(Ne, n_voxels, &data_c[0], &weights_c[0], &wraps_c[0], &mask_c[0], &A_c[0], max_v, &v_c[0], grid)
        end2 = time.time()

    v_out = np.reshape(v_c, vshape)
    v_out = np.moveaxis(v_out,-1,0)

    # print('v_solve took total = %f and the cpp part took %f' % ( (time.time()-start1), (end2-start2) ))

    return v_out

def c_vel_reg(data, A, max_v, v_in=None, lam=1.0, weights=None, wnorm=2.0, weighting=True, mask=None, grid=0):
    start1 = time.time()

    if mask is None:
        mask = np.ones(data.shape[1:], np.int)
    cdef np.ndarray[int, ndim=1, mode="c"] mask_c = np.ascontiguousarray(np.ravel(mask), np.intc)

    data = np.moveaxis(data,0,-1)
    data_n = data/np.abs(data)

    norms = np.sum(np.abs(A)**2,axis=1)**(1./2)
    wraps = np.round(0.99 * norms * max_v / 2.0 / np.pi)
    norms /= norms.min()

    # print('wraps = %s' % wraps)
    print('norms = %s' % norms)

    if (weights is None) and weighting:
        weights = (norms * np.abs(data)) ** wnorm
    elif (weights is None) and (not weighting):
        weights = np.ones(data.shape, np.float32)
    else:
        weights = weights ** wnorm

    dshape = np.array(data.shape)

    Ne = dshape[-1]
    dims = dshape[:-1]
    n_voxels = np.prod(dims)

    cdef np.ndarray[np.complex64_t, ndim=1, mode="c"] data_c = np.ascontiguousarray(np.ravel(data_n), np.complex64)
    cdef np.ndarray[int, ndim=1, mode="c"] wraps_c = np.ascontiguousarray(np.ravel(wraps), np.intc)
    cdef np.ndarray[int, ndim=1, mode="c"] dims_c = np.ascontiguousarray(np.ravel(dims), np.intc)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] A_c = np.ascontiguousarray(np.ravel(A), np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] weights_c = np.ascontiguousarray(np.ravel(weights), np.float32)

    vshape = dshape.copy()
    vshape[-1] = 3
    if v_in is None:
        v_in = np.zeros(vshape, np.float32)
    else:
        v_in = np.moveaxis(v_in, 0, -1)
    v_out = np.zeros(vshape, np.float32)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] v_in_c = np.ascontiguousarray(np.ravel(v_in), np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] v_out_c = np.ascontiguousarray(np.ravel(v_out), np.float32)

    debug_shape = dshape.copy()
    debug_shape[-1] = 6
    debug = np.zeros(debug_shape, np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] debug_c = np.ascontiguousarray(np.ravel(debug), np.float32)

    reg_v2(Ne, len(dims), &dims_c[0], grid, 
            &mask_c[0], &data_c[0], &weights_c[0], &wraps_c[0],
            &A_c[0], max_v, lam, &v_in_c[0], &v_out_c[0],
            &debug_c[0])

    return np.moveaxis(v_out, -1, -0), np.moveaxis(debug, -1, 0)