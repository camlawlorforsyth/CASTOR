
import numpy as np

from astropy.table import Table
import astropy.units as u

import binning
import noise
import photometry

# take the raw SKIRT output and add noise and a PSF
noise.process_everything(population='quenched')

# now bin the cutouts for each galaxy and save the resulting files
binning.bin_all(population='quenched')

# then determine the flux for every bin for each galaxy, saving the
# photometry for each galaxy into a separate file
filters = ['castor_uv', 'castor_u', 'castor_g', 'roman_f106', 'roman_f129',
           'roman_f158', 'roman_f184']
photometry.all_fluxes(filters, population='quenched')


