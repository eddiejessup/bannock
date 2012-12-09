'''
Created on 27 Feb 2012

@author: ejm
'''

import numpy as np

def circle_intersect(r_1, R_1, r_2, R_2):
    return vector_mag(r_1 - r_2) < R_1 + R_2

# Index wrapping

def wrap_real(L, L_half, r):
    if r > L_half: 
        r -= L
    elif r < -L_half: 
        r += L
    return r

def wrap(M, i):
    if i >= M: 
        i -= M
    elif i < 0: 
        i += M
    return i

def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0

def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1

# Centre of mass

def rms_com(r, periodic=False, L=None):
    ''' RMS distance of array of cartesian vectors r from their 
    centre-of-mass vector. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    return rms(r, com(r, periodic, L), periodic, L)

def com(r, periodic=False, L=None):
    ''' Centre-of-mass vector of array of cartesian vectors r, possibly in
    a periodic system with system size L. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if not periodic:
        r_com = np.mean(r, 0)
    else:
        r_com = np.zeros([r.shape[r.ndim - 1]], dtype=np.float)
        for i_dim in range(len(r_com)):
            x_0 = r[:, i_dim]
            x_av = np.mean(x_0)
            rms_min = rms(x_0, x_av, True, L)
            steps_0 = 4
            steps = float(steps_0)
            while True:
                for x_base in np.arange(-L / 2.0 + L / steps, L / 2.0, L / steps):
                    x_new = x_0.copy()
                    x_new[np.where(x_new < x_base)] += L
                    x_av_new = np.mean(x_new)
                    if x_av_new > L / 2.0:
                        x_av_new -= L
                    rms_new = rms(x_new, x_av_new, True, L)
                    if rms_new < rms_min:
                        x_av = x_av_new
                        rms_min = rms_new

                if (rms(x_0, 0.99 * x_av, True, L) < rms_min or 
                    rms(x_0, 1.01 * x_av, True, L) < rms_min):
                    steps *= 2
                    print('Recalculating c.o.m. with steps=%i' % steps)
                else:
                    r_com[i_dim] = x_av
                    break
    return r_com

def rms(r, r_0, periodic=False, L=None):
    ''' RMS distance of array of cartesian vectors r from point r_0. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    r_sep_sq = (r - r_0) ** 2
    if periodic:
        if L is None:
            raise Exception('Require system size to be specified')
        r_sep_sq = np.minimum(r_sep_sq, (L - np.abs(r - r_0)) ** 2)
    return np.sqrt(np.mean(np.sum(r_sep_sq, r_sep_sq.ndim - 1)))

# Vectors

def vector_mag_sq(v):
    ''' Squared magnitude of array of cartesian vectors v.  
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    return np.sum(np.square(v), v.ndim - 1)

def vector_mag(v):
    ''' Magnitude of array of cartesian vectors v. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    return np.sqrt(vector_mag_sq(v))

def vector_unit_nonull(v):
    ''' Array of cartesian vectors v into unit vectors. 
    If null vector encountered, raise exception. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if v.size == 0: return v
    mag = vector_mag(v)
    if len(np.where(mag == 0.0)[0]) > 0:
        raise Exception('Can''t unitise the null(-ish) vector')
    return v / mag[..., np.newaxis]

def vector_unit_nullnull(v):
    ''' Array of cartesian vectors into unit vectors.
    If null vector encountered, print a warning but leave as null. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if v.size == 0: return v
    mag = vector_mag(v)
    i_nonzeros = np.where(mag > 0.0)[0]
#    if len(np.where(mag == 0.0)[0]) > 0:
#        print('Warning: leaving null vector unaltered when asked to unitise')
    v_new = v.copy()    
    v_new[i_nonzeros] /= mag[i_nonzeros][..., np.newaxis]
    return v_new

def vector_unit_nullrand(v):
    ''' Array of cartesian vectors into unit vectors. 
    If null vector encountered, pick new random unit vector. 
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if v.size == 0: return v
    v_new = v.copy()
    i_zeros = np.where(vector_mag(v) == 0.0)[0]
    v_new[i_zeros] = point_pick_cart(v.shape[-1], len(i_zeros))
    return v_new / vector_mag(v_new)[..., np.newaxis]

def vector_perp(v):
    if v.shape[-1] != 2: 
        raise Exception
    v_perp = np.empty_like(v)
    v_perp[..., 0] = v[:, 1]
    v_perp[..., 1] = -v[:, 0]
    return v_perp    

# Coordinate transformations

def polar_to_cart(arr_p):
    ''' Array of vectors arr_c corresponding to cartesian 
    representation of array of polar vectors arr_p.
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if arr_p.ndim != 2:
        raise Exception('Require 2d array for conversion')
    dim = arr_p.shape[1]
    if dim == 1: 
        arr_c = arr_p.copy()
    elif dim == 2:
        arr_c = np.zeros_like(arr_p)
        arr_c[..., 0] = arr_p[..., 0] * np.cos(arr_p[..., 1])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1])        
    elif dim == 3:
        arr_c = np.zeros_like(arr_p)
        arr_c[..., 0] = (arr_p[..., 0] * np.cos(arr_p[..., 1]) * 
                         np.sin(arr_p[..., 2]))
        arr_c[..., 1] = (arr_p[..., 0] * np.sin(arr_p[..., 1]) * 
                         np.sin(arr_p[..., 2]))
        arr_c[..., 2] = arr_p[..., 0] * np.cos(arr_p[..., 2])
    else:
        raise Exception('Invalid vector for polar representation')
    return arr_c

def cart_to_polar(arr_c):
    ''' Array of vectors arr_p corresponding to polar representation 
    of array of cartesian vectors arr_c.
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    if arr_c.ndim != 2:
        raise Exception('Require 2d array for conversion')
    dim = arr_c.shape[-1]
    if dim == 1: 
        arr_p = arr_c.copy()
    elif dim == 2:
        arr_p = np.zeros_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    elif dim == 3: 
        arr_p = np.zeros_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
        arr_p[..., 2] = np.arccos(arr_c[..., 2] / arr_p[..., 0])
    else:
        raise Exception('Invalid vector for polar representation') 
    return arr_p

def point_pick_polar(dim, n=1):
    a = np.zeros([n, dim], dtype=np.float)
    if dim == 1:
        a[..., 0] = np.sign(np.random.uniform(-1.0, +1.0, (n, dim)))
    elif dim == 2:
        a[..., 0] = 1.0
        a[..., 1] = np.random.uniform(-np.pi, +np.pi, n)
    elif dim == 3:
        u, v = np.random.uniform(0.0, 1.0, (2, n))
        a[..., 0] = 1.0
        a[..., 1] = np.arccos(2.0 * v - 1.0)
        a[..., 2] = 2.0 * np.pi * u
    else:
        raise Exception('Invalid vector for polar representation')
    return a

def point_pick_cart(dim, n=1):
    return polar_to_cart(point_pick_polar(dim, n))

# Rotations

def rotate_1d(a, p):
    if p < 0.0 or p > 1.0:
        raise Exception('Invalid switching probability for rotation in 1d')
    a_rot = a.copy()
    i_switchers = np.where(np.random.uniform(0.0, 1.0, a.shape[0]) < p)[0]
    a_rot[i_switchers] *= -1
    return a_rot

def rotate_2d(a, theta):
    s, c = np.sin(theta), np.cos(theta)
    if a.shape[-1] != 2:
        raise Exception('Input array not 2d')
    a_rot = np.zeros_like(a)
    a_rot[..., 0] = c * a[..., 0] - s * a[..., 1] 
    a_rot[..., 1] = s * a[..., 0] + c * a[..., 1]
    return a_rot

def rotate_3d(a, theta, ax_raw):
    ax = vector_unit_nonull(ax_raw)
    a_rot = a.copy()
    ax_x, ax_y, ax_z = ax[..., 0], ax[..., 1], ax[..., 2]

    s, c = np.sin(theta), np.cos(theta)
    omc = 1.0 - c
    a_rot[..., 0] = (a[..., 0] * (c + np.square(ax_x) * omc) + 
                     a[..., 1] * (ax_x * ax_y * omc - ax_z * s) + 
                     a[..., 2] * (ax_x * ax_z * omc + ax_y * s))
    a_rot[..., 1] = (a[..., 0] * (ax_y * ax_x * omc + ax_z * s) + 
                     a[..., 1] * (c + np.square(ax_y) * omc) + 
                     a[..., 2] * (ax_y * ax_z * omc - ax_x * s))
    a_rot[..., 2] = (a[..., 0] * (ax_z * ax_x * omc - ax_y * s) + 
                     a[..., 1] * (ax_z * ax_y * omc + ax_x * s) + 
                     a[..., 2] * (c + np.square(ax_z) * omc))
    return a_rot

# Numpy arrays

def extend_array(a, n):
    if n < 1: raise Exception('Extend factor >= 1')
    M_new = a.shape[0] * n
    a_new = np.zeros(a.ndim * [M_new], dtype=a.dtype)
    for x_new in range(M_new):
        for y_new in range(M_new):
            a_new[x_new, y_new] = a[x_new // n, y_new // n]
    return a_new

def field_subset(f, inds, rank=0):
    f_dim_space = f.ndim - rank
    if inds.ndim > 2: 
        raise Exception('Too many dimensions in indices array')
    if inds.ndim == 1:
        if f_dim_space == 1:
            return f[inds]
        else: 
            raise Exception('Indices array is 1d but field is not')
    if inds.shape[1] != f_dim_space: 
        raise Exception('Indices and field dimensions do not match')

    # That's magic, don't touch it fuckers! -- Paul Daniels (not really)
    return f[tuple([inds[:, i_dim] for i_dim in range(inds.shape[1])])]

def empty_lists(n, d):
    if n < 1:
        raise Exception('Require list length > 0')
    if d < 1:
        raise Exception('Require list dimension > 0')
    return empty_lists_core(n, d)

def empty_lists_core(n, d):
    if d == 0: return []
    else: return [empty_lists_core(n, d - 1) for i in range(n)]
