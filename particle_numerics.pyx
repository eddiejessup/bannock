import numpy as np
from ciabatta import vector
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def vicsek_inters(np.ndarray[np.float_t, ndim=2] v,
                  np.ndarray[int, ndim=2] inters,
                  np.ndarray[int, ndim=1] intersi):
    cdef unsigned int i_1, i_i_2, i_dim
    cdef np.ndarray[np.float_t, ndim=2] v_vic = v.copy()

    for i_1 in range(v.shape[0]):
        for i_i_2 in range(intersi[i_1]):
            for i_dim in range(v.shape[1]):
                v_vic[i_1, i_dim] += v[inters[i_1, i_i_2] - 1, i_dim]
    return vector.vector_unit_nullnull(v_vic) * vector.vector_mag(v)[:, np.newaxis]
