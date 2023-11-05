
import numpy as np

import astropy.constants as c
import astropy.units as u

def fnu_to_spatial_electron_flux(fnu, lam_pivot, delta_lam, throughput,
                                 gain=1*u.electron/u.photon) :
    
    lam_pivot = lam_pivot.to(u.m) # convert from um to m
    delta_lam = delta_lam.to(u.m) # convert from um to m
    
    # difference in wavelength to difference in frequency
    delta_nu = (c.c*delta_lam/np.square(lam_pivot)).to(u.Hz)
    
    # calculate the spatial photon flux in photons/s/cm^2/Hz/arcsec^2
    photnu = fnu.to(u.photon/np.square(u.cm*u.arcsec)/u.s/u.Hz,
                    equivalencies=u.spectral_density(lam_pivot))
    
    # calculate the electron flux in electons/s/cm^2/arcsec^2
    spatial_electron_flux = photnu*throughput*delta_nu*gain
    
    return spatial_electron_flux

def spatial_electron_flux_to_fnu(spatial_electron_flux, lam_pivot, delta_lam,
                                 throughput, gain=1*u.electron/u.photon) :
    
    lam_pivot = lam_pivot.to(u.m) # convert from um to m
    delta_lam = delta_lam.to(u.m) # convert from um to m
    
    # difference in wavelength to difference in frequency
    delta_nu = (c.c*delta_lam/np.square(lam_pivot)).to(u.Hz)
    
    # calculate the photon flux in photons/s/cm^2/Hz
    photnu = spatial_electron_flux/throughput/delta_nu/gain
    
    # calculate the flux density in janskys
    fnu = photnu.to(u.Jy, equivalencies=u.spectral_density(lam_pivot))
    
    return fnu
