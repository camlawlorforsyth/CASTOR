
import ctypes
import numpy as np

def adaptive_smoothing(xx, yy, hsml, xcenters, ycenters, nRe, weights) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # create_images.py#L159-L250
    
    # prepare the shared object library for use
    sphlib = np.ctypeslib.load_library('adaptive_smoothing', './')
    sphlib.add.restype = None
    sphlib.add.argtypes = [ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double]
    
    XX, YY = np.meshgrid(xcenters, ycenters)
    ny, nx = XX.shape
    Y_flat, X_flat = YY.ravel(), XX.ravel()
    Z_flat = np.zeros_like(X_flat)
    sphlib.add(
        X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Z_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nx),
        ctypes.c_int(ny),
        xx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        yy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hsml.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(xx.size),
        ctypes.c_double(nRe))
    
    HH = Z_flat.reshape(XX.shape)
    
    return HH

def adaptive_smoothing_py(xx, yy, hsml, xcenters, ycenters, nRe, weights) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # create_images.py#L159-L250
    
    XX, YY = np.meshgrid(xcenters, ycenters)
    ny, nx = XX.shape
    Y_flat, X_flat = YY.ravel(), XX.ravel()
    Z_flat = np.zeros_like(X_flat)
    
    Z_flat = add(X_flat, Y_flat, Z_flat, nx, ny, xx, yy, weights, hsml, len(xx), nRe)
    
    return Z_flat.reshape(XX.shape)

def sph_kernel_2d(rr, hh) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # adaptive_smoothing.c#L8-L22
    
    # 2D kernel from Monaghan (1992), re-defined over the interval [0, h] as in
    # Springel et al. (2001) GADGET description, equation (A.1)
    
    xx = rr/hh
    if xx <= 1.0 :
        uu = 1.0 - xx
        norm = 4.0*10.0/(7.0*np.pi*hh*hh)
        if xx <= 0.5 :
            return norm*(1.0 - 6.0*xx*xx*uu)
        else :
            return norm*(2.0*uu*uu*uu)
    return 0.0

def rhalfs_to_pixels(rr, npix, num_rhalfs) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # adaptive_smoothing.c#L24-L26
    return int(np.floor(npix/2.0 + rr*npix/(2.0*num_rhalfs)))

def add(XX, YY, ZZ, nx, ny, x0, y0, weights, hsml, npoints, nRe) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # adaptive_smoothing.c#L28-L48
    
    # ZZ should already exist and be zeros
    # Distances are given in units of the stellar half-mass radius.
    for k in range(npoints) :
        hh = hsml[k]
        
        imin = int(max(0, rhalfs_to_pixels(y0[k] - hh, ny, nRe)))
        imax = int(min(ny - 1, rhalfs_to_pixels(y0[k] + hh, ny, nRe)))
        jmin = int(max(0, rhalfs_to_pixels(x0[k] - hh, nx, nRe)))
        jmax = int(min(nx - 1, rhalfs_to_pixels(x0[k] + hh, nx, nRe)))
        
        for i in range(imin, imax + 1) :
            for j in range(jmin, jmax + 1) :
                nn = i*ny + j
                rr = np.sqrt(np.square(XX[nn] - x0[k]) + np.square(YY[nn] - y0[k]))
                ZZ[nn] += weights[k]*sph_kernel_2d(rr, hh)
    
    return ZZ
