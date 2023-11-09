
import numpy as np

from astropy.io import fits

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def open_cutout(infile, shape=False, simple=False) :
    
    with fits.open(infile) as hdu :
        if simple :
            data = hdu[0].data
            shape = data.shape
        else :
            data = hdu[0].data
            shape = data.shape
            hdr = hdu[0].header
            redshift = hdr['Z']
            exptime = hdr['EXPTIME']
            area = hdr['AREA']
            photfnu = hdr['PHOTFNU']
            scale = hdr['SCALE']
    
    if simple :
        return data, shape
    else :
        return data, shape, redshift, exptime, area, photfnu, scale
