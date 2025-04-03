
from os.path import exists
import numpy as np
import ctypes

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
from scipy.interpolate import RectBivariateSpline

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def create_idealized_image(model_redshift=0.5, fov=10) :
    
    Re = 1.0857552289962769
    center = [26116.61132812, 13629.70800781, 1335.57373047]
    redshift = 1.49551216649556
    
    infile = '/mnt/s/Cam/University/GitHub/TNG/mpb_cutouts_099/cutout_40_299910.hdf5'
    
    # define the FoV, and number of pixels for the redshift of interest
    plate_scale = 0.05*u.arcsec/u.pix
    nPix_raw = fov*Re*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale
    nPix = np.ceil(nPix_raw).astype(int).value
    if nPix % 2 == 0 : # ensure all images have an odd number of pixels,
        nPix += 1      # so that a central pixel exists
    
    with h5py.File(infile, 'r') as hf :
        star_coords = hf['PartType4/Coordinates'][:]
        stellarHsml = hf['PartType4/StellarHsml'][:] # [ckpc/h]
        Mstar = hf['PartType4/GFM_InitialMass'][:]*1e10/cosmo.h # solMass
        Zstar = hf['PartType4/GFM_Metallicity'][:]
        
        # formation times in units of scalefactor
        formation_scalefactors = hf['PartType4/GFM_StellarFormationTime'][:]
    
    # formation times in units of age of the universe (ie. cosmic time)
    formation_times = cosmo.age(1/formation_scalefactors - 1).value
    
    # don't project the galaxy face-on
    dx, dy, dz = (star_coords - center).T # [ckpc/h]
    
    # limit star particles to those that have positive formation times
    mask = (formation_scalefactors > 0)
    star_coords = star_coords[mask]
    stellarHsml = stellarHsml[mask]
    Mstar = Mstar[mask]
    Zstar = Zstar[mask]
    formation_times = formation_times[mask]
    dx = dx[mask]
    dy = dy[mask]
    # dz = dz[mask]
    
    # convert the formation times to actual ages at the time of observation,
    # while also imposing a lower age limit of 1 Myr
    ages = (cosmo.age(redshift).value - formation_times)*1e9 # [Gyr]
    # ages[ages < 1e6] = 1e6
    
    castor_uv = get_fluxes(Mstar, Zstar, ages, 'castor_uv')
    
    # normalize by Re
    dx, dy, hsml = dx/Re, dy/Re, stellarHsml/Re
    
    # define 2D bins (in units of Re)
    edges = np.linspace(-5, 5, nPix + 1) # Re
    xcenters = 0.5*(edges[:-1] + edges[1:])
    ycenters = 0.5*(edges[:-1] + edges[1:])
    
    # store image into an array
    image = np.zeros((nPix, nPix))
    
    if not exists('test_image_new.npy') :
        image[:, :] = adaptive_smoothing(dx, dy, hsml, xcenters, ycenters,
                                         5, castor_uv)
        np.save('test_image_new.npy', image)
    
    return

def adaptive_smoothing(xx, yy, hsml, xcenters, ycenters, nRe, weights) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # create_images.py#L159-L250
    
    # make everything a double
    xx = np.float64(xx)
    yy = np.float64(yy)
    hsml = np.float64(hsml)
    weights = np.float64(weights)
    
    # ignore out-of-range particles
    locs_withinrange = (np.abs(xx) <= nRe) | (np.abs(yy) <= nRe)
    xx = xx[locs_withinrange]
    yy = yy[locs_withinrange]
    hsml = hsml[locs_withinrange]
    weights = weights[locs_withinrange]
    
    # prepare the shared object library for use
    sphlib = np.ctypeslib.load_library('adaptive_smoothing', './')
    sphlib.add.restype = None
    sphlib.add.argtypes = [ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double]
    
    # create arrays to describe the position of the stellar particles
    XX, YY = np.meshgrid(xcenters, ycenters)
    ny, nx = XX.shape
    Y_flat, X_flat = YY.ravel(), XX.ravel()
    Z_flat = np.zeros_like(X_flat)
    
    # perform adaptive smoothing
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

def get_fluxes(initial_masses_Msol, metallicities, stellar_ages_yr, filt, zz=0.5) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # create_images.py#L26-L58
    
    with h5py.File('bc03_2016_magnitudes_z_{:03}.hdf5'.format(
        str(zz).replace('.', '')), 'r') as hf :
        bc03_metallicities = hf['metallicities'][:]
        bc03_stellar_ages = hf['stellar_ages'][:]
        bc03_magnitudes = hf[filt][:]
    
    # setup up a 2D interpolation over the metallicities and ages
    spline = RectBivariateSpline(bc03_metallicities, bc03_stellar_ages,
                                 bc03_magnitudes, kx=1, ky=1, s=0)
    
    # BC03 fluxes are normalized to a mass of 1 Msol
    magnitudes = spline.ev(metallicities, stellar_ages_yr) # [m_AB]
    
    # account for the initial mass of the stellar particles
    magnitudes -= 2.5*np.log10(initial_masses_Msol) # [m_AB]
    
    # convert apparent magnitude to fluxes in Jy
    fluxes = np.power(10, -0.4*magnitudes)*3631 # [Jy]
    
    return fluxes
