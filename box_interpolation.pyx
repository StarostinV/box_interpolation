from libc.math cimport floor, ceil

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def box_interpolation(
        np.ndarray[np.float_t, ndim=1] inten, 
        np.ndarray[np.float_t, ndim=1] qx, 
        np.ndarray[np.float_t, ndim=1] qy,
        int nx, int ny, 
        double xs, double xdel, double xhw,
        double ys, double ydel, double yhw):
    
    cdef int ximin, ximax, yimin, yimax
    cdef int i, k, l, ind
    cdef int size = nx * ny
    cdef double xmsdel
    
    cdef np.ndarray[np.float_t, ndim=1] ninten = np.zeros(size)
    cdef np.ndarray[np.float_t, ndim=1] ginten = np.zeros(size)

    cdef double xhwdel = xhw / 2. / xdel
    cdef double yhwdel = yhw / 2./ ydel

    for i in range(inten.size):
        xmsdel = (qx[i] - xs) / xdel
        ximin, ximax = getind(xmsdel, xhwdel, nx)
        xmsdel = (qy[i] - ys) / ydel
        yimin, yimax = getind(xmsdel, yhwdel, ny)

        for l in range(ximin, ximax + 1):
            for k in range(yimin, yimax + 1):
                ind = k * nx + l
                ninten[ind] += 1
                ginten[ind] += inten[i]

    for i in range(size):
        if ninten[i] != 0:
            ginten[i] = ginten[i] / ninten[i]
    return ginten.reshape((ny, nx))


cdef inline (int, int) getind(double xmsdel, double hwdel, int n):
    cdef int imin = int(ceil(xmsdel - hwdel))
    if (imin < 0):
        imin = 0
    elif (imin > (n - 1)):
        imin = n
    cdef int imax = int(floor(xmsdel + hwdel))
    if (imax < 0):
        imax = -1
    elif (imax > (n - 1)):
        imax = n - 1
    return imin, imax