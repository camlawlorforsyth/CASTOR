
# import numpy as np

# from astropy.table import Table
# import astropy.units as u

import cutouts
import filters
import galaxev
import noise
import photometry
# import skirt

# save the necessary passbands from downloaded files
filters.throughputs_castor()
filters.throughputs_euclid()
filters.throughputs_hst()
filters.throughputs_jwst()
filters.throughputs_roman()
filters.prepare_throughputs_for_fastpp()
# filters.prepare_throughputs_for_skirt()

# save the noise components to file
noise.get_noise()

'''
# save all the SKIRT input for use on CANFAR
# skirt.save_all_skirt_input()

# transfer SKIRT input to CANFAR and run SKIRT on all galaxies, and
# transfer SKIRT output back to local machine
'''

# create synthetic images using GALAXEV/BC03 and a Calzetti dust law
galaxev.read_bc03()
galaxev.determine_magnitudes()
galaxev.create_all_idealized_images()

'''
# take the raw SKIRT output and add noise and a PSF
cutouts.create_all_mocks(population='quenched', psf=0.15*u.arcsec)

# now bin the cutouts for each galaxy and save the resulting files
# binning.bin_all(population='quenched')
'''

# take the idealized (and extincted) output and add noise and a PSF
cutouts.create_all_mocks()

'''
# then determine the flux for every bin for each galaxy, saving the photometry
# for each galaxy into a separate file, using all CASTOR+Roman, HST+JWST filters
photometry.all_fluxes(survey='all', population='quenched')
'''

# determine the flux for every pixel for each galaxy using the CASTOR+Roman
# filters, and save the resulting photometric tables
photometry.determine_all_photometry()

# save all the photometry from the CASTOR+Roman filters and for every pixel
# to be fit into a single table that will be processed with FAST++
photometry.join_all_photometry()


