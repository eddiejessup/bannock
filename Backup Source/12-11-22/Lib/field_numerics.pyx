import numpy as np
cimport numpy as np
from utils_cy cimport wrap_inc, wrap_dec

def div(field, dx):
    assert dx > 0.0
    div = np.empty(field.shape[:-1], dtype=field.dtype)    
    if field.ndim == 2:
        div_1d(field, div, dx)
    elif field.ndim == 3:
        div_2d(field, div, dx)
    elif field.ndim == 4:
        div_3d(field, div, dx)
    else:
        raise Exception('Divergence not implemented in this dimension')
    return div

def div_1d(np.ndarray[np.float_t, ndim=2] field, 
           np.ndarray[np.float_t, ndim=1] div,
           double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx, diff

    for i_x in range(M_x):
        diff = field[wrap_inc(M_x, i_x), 0] - field[wrap_dec(M_x, i_x), 0]
        div[i_x] = diff / dx_double

def div_2d(np.ndarray[np.float_t, ndim=3] field, 
           np.ndarray[np.float_t, ndim=2] div,
           double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            diff = (field[wrap_inc(M_x, i_x), i_y, 0] - 
                    field[wrap_dec(M_x, i_x), i_y, 0])
            diff += (field[i_x, wrap_inc(M_y, i_y), 1] - 
                     field[i_x, wrap_dec(M_y, i_y), 1])
            div[i_x, i_y] = diff / dx_double

def div_3d(np.ndarray[np.float_t, ndim=4] field, 
           np.ndarray[np.float_t, ndim=3] div,
           double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                diff = (field[wrap_inc(M_x, i_x), i_y, i_z, 0] - 
                        field[wrap_dec(M_x, i_x), i_y, i_z, 0])
                diff += (field[i_x, wrap_inc(M_y, i_y), i_z, 1] - 
                         field[i_x, wrap_dec(M_y, i_y), i_z, 1])
                diff += (field[i_x, i_y, wrap_inc(M_z, i_z), 2] - 
                         field[i_x, i_y, wrap_dec(M_z, i_z), 2])
                div[i_x, i_y, i_z] = diff / dx_double

def grad(field, dx):
    assert dx > 0.0
    grad = np.empty(field.shape + (field.ndim,), dtype=field.dtype)
    if field.ndim == 1: grad_1d(field, grad, dx)
    elif field.ndim == 2: grad_2d(field, grad, dx)
    elif field.ndim == 3: grad_3d(field, grad, dx)
    else:
        raise Exception("Grad not implemented in this dimension")
    return grad

def grad_1d(np.ndarray[np.float_t, ndim=1] field, 
            np.ndarray[np.float_t, ndim=2] grad,
            double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        grad[i_x, 0] = (field[wrap_inc(M_x, i_x)] - 
                        field[wrap_dec(M_x, i_x)]) / dx_double

def grad_2d(np.ndarray[np.float_t, ndim=2] field, 
            np.ndarray[np.float_t, ndim=3] grad,
            double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            grad[i_x, i_y, 0] = (field[wrap_inc(M_x, i_x), i_y] - 
                                 field[wrap_dec(M_x, i_x), i_y]) / dx_double
            grad[i_x, i_y, 1] = (field[i_x, wrap_inc(M_y, i_y)] - 
                                 field[i_x, wrap_dec(M_y, i_y)]) / dx_double

def grad_3d(np.ndarray[np.float_t, ndim=3] field, 
            np.ndarray[np.float_t, ndim=4] grad,
            double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                grad[i_x, i_y, i_z, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - 
                                          field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
                grad[i_x, i_y, i_z, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - 
                                          field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
                grad[i_x, i_y, i_z, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - 
                                          field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double

def grad_i(field, inds, dx):
    assert dx > 0.0
    assert inds.ndim == 2
    assert field.ndim == inds.shape[1]
    grad_i = np.empty(inds.shape, dtype=field.dtype)
    if field.ndim == 1:
        grad_i_1d(field, inds, grad_i, dx)
    elif field.ndim == 2:
        grad_i_2d(field, inds, grad_i, dx)
    elif field.ndim == 3:
        grad_i_3d(field, grad_i, dx)
    else:
        raise Exception("Grad_i not implemented in this dimension")
    return grad_i

def grad_i_1d(np.ndarray[np.float_t, ndim=1] field, 
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef unsigned int i, i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x = inds[i, 0]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x)] - 
                        field[wrap_dec(M_x, i_x)]) / dx_double

def grad_i_2d(np.ndarray[np.float_t, ndim=2] field, 
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef unsigned int i, i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y = inds[i, 0], inds[i, 1]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y] - 
                        field[wrap_dec(M_x, i_x), i_y]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y)] - 
                        field[i_x, wrap_dec(M_y, i_y)]) / dx_double

def grad_i_3d(np.ndarray[np.float_t, ndim=3] field, 
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef unsigned int i, i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y, i_z = inds[i, 0], inds[i, 1], inds[i, 2]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - 
                        field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - 
                        field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
        grad_i[i, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - 
                        field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double

def laplace(field, dx):
    assert dx > 0.0
    laplace = np.empty_like(field)
    if field.ndim == 1:
        laplace_1d(field, laplace, dx)
    elif field.ndim == 2:
        laplace_2d(field, laplace, dx)
    elif field.ndim == 3:
        laplace_3d(field, laplace, dx)
    else:
        raise Exception('Laplacian not implemented in this dimension')
    return laplace 

def laplace_1d(np.ndarray[np.float_t, ndim=1] field, 
               np.ndarray[np.float_t, ndim=1] laplace,
               double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_sq = dx * dx, diff

    for i_x in range(M_x):
        diff = (field[wrap_inc(M_x, i_x)] + 
                field[wrap_dec(M_x, i_x)] - 
                2.0 * field[i_x])

        laplace[i_x] = diff / dx_sq

def laplace_2d(np.ndarray[np.float_t, ndim=2] field, 
               np.ndarray[np.float_t, ndim=2] laplace,
               double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_sq = dx * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            diff = (field[wrap_inc(M_x, i_x), i_y] + 
                    field[wrap_dec(M_x, i_x), i_y] - 
                     2.0 * field[i_x, i_y])
            diff += (field[i_x, wrap_inc(M_y, i_y)] + 
                     field[i_x, wrap_dec(M_y, i_y)] - 
                     2.0 * field[i_x, i_y])

            laplace[i_x, i_y] = diff / dx_sq

def laplace_3d(np.ndarray[np.float_t, ndim=3] field, 
               np.ndarray[np.float_t, ndim=3] laplace,
               double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_sq = dx * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                diff = (field[wrap_inc(M_x, i_x), i_y, i_z] + 
                        field[wrap_dec(M_x, i_x), i_y, i_z] - 
                        2.0 * field[i_x, i_y, i_z])
                diff += (field[i_x, wrap_inc(M_y, i_y), i_z] + 
                         field[i_x, wrap_dec(M_y, i_y), i_z] - 
                         2.0 * field[i_x, i_y, i_z])
                diff += (field[i_x, i_y, wrap_inc(M_z, i_z)] + 
                         field[i_x, i_y, wrap_dec(M_z, i_z)] - 
                         2.0 * field[i_x, i_y, i_z])

                laplace[i_x, i_y, i_z] = diff / dx_sq

def drift_diffusion(u, V, D, z, dx):
    assert u.shape == V.shape == D.shape == z.shape
    assert dx > 0.0
    result = np.empty_like(u)
    if u.ndim == 1:
        drift_diffusion_1d(u, V, D, z, result, dx)
    else:
        raise Exception("Drift diffusion not implemented in this dimension")
    return result

def drift_diffusion_1d(np.ndarray[np.float_t, ndim=1] u, 
                       np.ndarray[np.float_t, ndim=1] V,  
                       np.ndarray[np.float_t, ndim=1] D,
                       np.ndarray[np.float_t, ndim=1] z_,  
                       np.ndarray[np.float_t, ndim=1] result,  
                       double dx):
    ''' Iterate drift diffusion (aka Smoluchowski) equation in 1d by finite
    differencing, interpolating values with central averageing. D and z_ can be 
    variable, result is RHS of equation,  
        du/dt = D laplace(u) + z_ div(u grad(V)) ,  or 
              = div(D * grad(u) z_ * u * grad(V))    '''

    cdef unsigned int i
    cdef unsigned int inc, dec
    cdef unsigned int I = u.shape[0]
    cdef double dx_sq_double = 2.0 * dx * dx, diff

    for i in range(I):
        inc = wrap_inc(I, i)
        dec = wrap_dec(I, i)

        diff = (((D[i] + D[inc]) * (u[inc] - u[i]) +  
                 0.5 * (z_[i] + z_[inc]) * (u[i] + u[inc]) * (V[inc] - V[i])) - 
                ((D[i] + D[dec]) * (u[i] - u[dec]) +  
                 0.5 * (z_[i] + z_[dec]) * (u[i] + u[dec]) * (V[i] - V[dec])))

        result[i] = diff / dx_sq_double

def density(inds, f, inc):
    assert inds.ndim == 2
    assert inds.shape[-1] == f.ndim
    f[...] = 0.0
    if f.ndim == 1:
        density_1d(inds, f, inc)
    elif f.ndim == 2:
        density_2d(inds, f, inc)
    elif f.ndim == 3:
        density_3d(inds, f, inc)
    else:
        raise Exception('Density calculation not implemented in this '
                        'dimension')

def density_1d(np.ndarray[np.int_t, ndim=2] inds, 
               np.ndarray[np.float_t, ndim=1] f, 
               np.float_t inc):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0]] += inc

def density_2d(np.ndarray[np.int_t, ndim=2] inds, 
               np.ndarray[np.float_t, ndim=2] f, 
               np.float_t inc):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1]] += inc

def density_3d(np.ndarray[np.int_t, ndim=2] inds, 
               np.ndarray[np.float_t, ndim=3] f, 
               np.float_t inc):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1], inds[i_part, 2]] += inc
