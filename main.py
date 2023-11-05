
import numpy as np

from astropy.table import Table
import astropy.units as u

import binning
import photometry


# now bin the cutouts for each galaxy and save the resulting files
binning.bin_all('TNG')

# then determine the flux for every bin for each galaxy, saving the
# photometry for each galaxy into a separate file
photometry.determine_fluxes('TNG', filters)


