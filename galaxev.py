
from os.path import exists
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import astropy.units as u
import h5py
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline

from core import calculate_distance_to_center, load_massive_galaxy_sample
from fastpy import calzetti2000

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def create_all_idealized_images(model_redshift=0.5, fov=10) :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # select only the quenched galaxies at the first snapshot >=75% of the way
    # through their quenching episodes
    mask = (((sample['mechanism'] == 1) | (sample['mechanism'] == 3)) &
        (sample['episode_progress'] >= 0.75))
    sample = sample[mask]
    
    # use the first snapshot >=75% of the way through the quenching episode,
    # but not any additional snapshots, for testing purposes
    mask = np.full(len(sample), False)
    idx = 0
    for subIDfinal in np.unique(sample['subIDfinal']) :
        mask[idx] = True
        idx += len(np.where(sample['subIDfinal'] == subIDfinal)[0])
    sample = sample[mask]
    
    # process every galaxy/snapshot pair
    for subID, snap, logM, SFR, Re, center, redshift in zip(sample['subID'],
        sample['snapshot'], sample['logM'], sample['SFR'], sample['Re'],
        sample['center'], sample['redshift']) :
        outfile_e = 'GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(
            snap, subID, str(model_redshift).replace('.', ''))
        if not exists(outfile_e) :
            create_idealized_image(snap, subID, logM, SFR, Re, center, redshift,
                                   model_redshift=model_redshift)
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def create_idealized_image(snap, subID, logM, SFR, Re, center, sim_redshift,
                           model_redshift=0.5, fov=10) :
    
    # define the FoV, and number of pixels for the redshift of interest
    model_plate_scale = 0.05*u.arcsec/u.pix
    fov_arcsec = fov*Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)
    nPix_raw = fov_arcsec/model_plate_scale
    nPix = np.ceil(nPix_raw).astype(int).value
    if nPix % 2 == 0 : # ensure all images have an odd number of pixels,
        nPix += 1      # so that a central pixel exists
    plate_scale = fov_arcsec/(nPix*u.pix)
    
    # open the TNG cutout and get the stellar particle properties
    infile = 'S:/Cam/University/GitHub/TNG/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(
        snap, subID)
    with h5py.File(infile, 'r') as hf :
        star_coords = hf['PartType4/Coordinates'][:]
        # stellarHsml = hf['PartType4/StellarHsml'][:] # [ckpc/h]
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
    # stellarHsml = stellarHsml[mask]
    Mstar = Mstar[mask]
    Zstar = Zstar[mask]
    formation_times = formation_times[mask]
    dx = dx[mask]
    dy = dy[mask]
    # dz = dz[mask]
    
    # normalize by Re
    dx, dy = dx/Re, dy/Re
    # hsml = stellarHsml/Re
    
    # ignore out-of-range particles
    locs_withinrange = (np.abs(dx) <= 5) | (np.abs(dy) <= 5)
    dx = dx[locs_withinrange]
    dy = dy[locs_withinrange]
    # hsml = hsml[locs_withinrange]
    
    # define 2D bins (in units of Re)
    edges = np.linspace(-5, 5, nPix + 1) # Re
    # xcenters = 0.5*(edges[:-1] + edges[1:])
    # ycenters = 0.5*(edges[:-1] + edges[1:])
    
    # convert the formation times to actual ages at the time of observation,
    # while also imposing a lower age limit of 1 Myr
    ages = (cosmo.age(sim_redshift).value - formation_times)*1e9 # [Gyr]
    # ages[ages < 1e6] = 1e6
    
    # define the filters that we want to create images for
    filters = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g',
               'roman_f106', 'roman_f129', 'roman_f158', 'roman_f184']
    num_filters = len(filters)
    
    # populate some attributes of the fits datacube
    header = fits.Header()
    header['ORIGIN'] = ('GALAXEV pipeline', 'adapted by CLF')
    header['BUNIT'] = ('Jy/pixel', 'Physical unit of the array values')
    header['CDELT1'] = (plate_scale.value, 'Coordinate increment along X-axis')
    header['CDELT2'] = (plate_scale.value, 'Coordinate increment along Y-axis')
    header['REDSHIFT'] = (model_redshift, 'redshift')
    for i in range(num_filters) :
        header['FILTER{}'.format(i)] = (filters[i],
                                        'Broadband filter index = {}'.format(i))
    
    # determine the distance/offset from the SFMS at the given snapshot
    dist_from_SFMS = determine_distance_from_SFMS(snap, logM, np.log10(SFR))
    
    # determine the distance to every pixel in units of Re
    dists = calculate_distance_to_center((nPix, nPix))/(nPix/fov)
    
    # determine the dust map as a function of spatial position, stellar mass,
    # and distance from the star forming main sequence
    dust_map = determine_dust_radial_profile(logM, dist_from_SFMS, dists) # [Av]
    
    # determine the flux decrease for every pixel based on the Calzetti dust law
    with h5py.File('tools/bc03_2016.hdf5', 'r') as hf :
        wavelengths = (hf['wavelengths'][:]*u.AA).to(u.m) # [m]
    
    # get the filter responses interpolated onto the BC03 wavelength grid
    response = np.zeros((num_filters, len(wavelengths)))
    for i, filt in enumerate(filters) :
        waves, trans = np.loadtxt('passbands/passbands_micron/{}.txt'.format(
            filt), unpack=True)
        waves = (waves*u.um).to(u.m) # [m]
        response[i] = np.interp(wavelengths*(1 + model_redshift),
                                waves, trans) # [], interpolate to wavelengths
    
    wavelengths_z = wavelengths*(1 + model_redshift) # redshift the wavelengths
    
    # determine the extinction as seen through the filters for unique Av values
    # (for computational efficiency)
    Avs = np.unique(dust_map)
    num_Avs = len(Avs)
    lookup_table = np.zeros((num_Avs, 1 + num_filters))
    for i, Av in enumerate(Avs) :
        lookup_table[i, 0] = Av
        calz = np.power(10, -0.4*Av*calzetti2000(wavelengths.to(u.um).value))
        for j, filt in enumerate(filters) :
            lookup_table[i, j+1] = trapezoid(calz*response[j], x=wavelengths_z)/trapezoid(
                response[j], x=wavelengths_z)
    
    # get the locations of each Av value in the lookup table for each pixel in
    # the dust map
    locs = np.array([np.where(lookup_table[:, 0] == val)[0][0] for val in
                     dust_map.flatten()])
    
    # create an extinction cube which describes the coefficients/factors which
    # will produce extincted images when multiplied with the idealized images
    extinction = np.zeros((num_filters, nPix, nPix)) # [], ie. dimensionless
    for i in range(num_filters) :                    # as a number in [0, 1]
        extinction[i, :, :] = lookup_table[:, i+1][locs].reshape((nPix, nPix))
    
    # store the idealized images into an array
    image = np.zeros((num_filters, nPix, nPix))
    for i, filt in enumerate(filters) :
        weight = get_fluxes(Mstar, Zstar, ages, filt)[locs_withinrange]
        image[i] = np.rot90(np.histogram2d(dx, dy, bins=(edges, edges),
                                           weights=weight)[0].T, k=1)
    # outfile_noDust = 'GALAXEV/{}_{}_z_{:03}_idealized.fits'.format(snap,
    #     subID, str(model_redshift).replace('.', ''))
    # hdulist = fits.HDUList([fits.PrimaryHDU(data=image, header=header)])
    # hdulist.writeto(outfile_noDust)
    
    # apply the extinction in the images
    image *= extinction
    
    # write to a fits file
    outfile = 'GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(
        snap, subID, str(model_redshift).replace('.', ''))
    hdulist = fits.HDUList([fits.PrimaryHDU(data=image, header=header)])
    hdulist.writeto(outfile)
    
    return

def determine_distance_from_SFMS(snap, logM, logSFR) :
    
    # open the helper file from TNG which contains fitted slopes and intercepts
    # for the SFMS as a function of time (ie. snapshot)
    with h5py.File('tools/TNG_SFMS_fits.hdf5', 'r') as hf :
        slope = hf['slope'][snap]
        intercept = hf['intercept'][snap]
    
    expected = slope*logM + intercept - logM # expected sSFR
    distance = (logSFR - logM) - expected    # actual sSFR - expected sSFR
    
    return distance

def determine_dust_radial_profile(logM, dist_from_SFMS, pixel_distances) :
    
    # AA = max(0.2, np.log10(np.power(10, logM)/np.power(10, 9.5)))
    AA = max(0.2, logM - 9.5) # logM = 9.5 is the lowest stellar mass for the
                              # z = 0 quenched sample
    
    BB = max(0, 0.2 + 0.1*dist_from_SFMS)
    
    # create the radial law, where pixel_distances is in units of Re
    radial_law = AA*np.exp(-pixel_distances) + BB
    
    return radial_law

def determine_magnitudes(model_redshift=0.5) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # stellar_photometrics.py#L40-L129
    
    outfile = 'tools/bc03_2016_magnitudes_z_{:03}.hdf5'.format(
        str(model_redshift).replace('.', ''))
    
    # read the saved input from the Bruzual & Charlot (2003) models
    datacube, metallicities, stellar_ages, wavelengths, masses = read_bc03()
    num_metallicities = len(metallicities)
    num_stellar_ages = len(stellar_ages)
    
    # attach units to the relevant quantities
    datacube *= u.solLum/u.AA
    stellar_ages *= u.yr
    wavelengths *= u.AA
    
    # shift restframe wavelengths to observer frame and convert to meters
    wavelengths *= (1 + model_redshift) # [AA]
    wavelengths = wavelengths.to(u.m) # [m]
    
    # define the AB magnitude system in wavelength units
    FAB_nu = 3631e-26*u.W/u.Hz/np.square(u.m) # [W Hz^-1 m^-2]
    FAB_lambda = (FAB_nu*c.c/np.square(wavelengths)).to(
        u.W/np.power(u.m, 3)) # [W m^-2 m^-1]
    
    # convert the restframe spectra from Lsol AA^-1 to W m^-1
    datacube = datacube.to(u.W/u.m) # [W m^-1]
    
    # convert luminosity to flux in observer frame; the factor of (1 + z) comes
    # from the stretching of dlambda (spreading of photons in wavelength)
    d_l = cosmo.luminosity_distance(model_redshift).to(u.m) # [m]
    datacube /= 4*np.pi*(1 + model_redshift)*np.square(d_l) # [W m^-2 m^-1]
    
    # open an HDF5 file for writing outputs
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            hf.create_dataset('metallicities', data=metallicities)
            hf.create_dataset('stellar_ages', data=stellar_ages)
    
    # define the filters that we're interested in creating mock observations for
    filters = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g',
               'roman_f106', 'roman_f129', 'roman_f158', 'roman_f184']
    
    # iterate over the filters
    for filt in filters :
        # store the magnitudes into an array
        magnitudes = np.zeros((num_metallicities, num_stellar_ages))
        
        # read the filter response function
        waves, trans = np.loadtxt('passbands/passbands_micron/{}.txt'.format(
            filt), unpack=True)
        waves = (waves*u.um).to(u.m) # [m]
        RR = np.interp(wavelengths, waves, trans) # [], interpolate to wavelengths
        
        # apply equation (8) from the BC03 manual
        # (https://www.bruzual.org/bc03/doc/bc03.pdf) to calculate the apparent
        # magnitude of the integrated photon flux collected by a detector with
        # filter response R(lambda)
        denominator = trapezoid(FAB_lambda*wavelengths*RR, x=wavelengths)
        
        # iterate for every metallicity+age combination
        for k in range(num_metallicities) :
            for j in range(num_stellar_ages) :
                F_lambda = datacube[k, j, :] # [W m^-2 m^-1], observer frame SED 
                numerator = trapezoid(F_lambda*wavelengths*RR, x=wavelengths)
                magnitudes[k, j] = -2.5*np.log10(numerator/denominator) # [m_AB]
        
        # create a dataset for the fluxes
        with h5py.File(outfile, 'a') as hf :
            if filt not in hf.keys() :
                hf.create_dataset(filt, data=magnitudes)
    
    return

def get_fluxes(initial_masses_Msol, metallicities, stellar_ages_yr, filt,
               model_redshift=0.5) :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # create_images.py#L26-L58
    
    with h5py.File('tools/bc03_2016_magnitudes_z_{:03}.hdf5'.format(
        str(model_redshift).replace('.', '')), 'r') as hf :
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

def read_bc03() :
    # from galaxev_pipeline
    # https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/
    # read_model_data.py#L4-L75
    
    datacube_file = 'tools/bc03_2016.hdf5'
    
    if not exists(datacube_file) :
        inDir = 'bc03/models/Padova1994/chabrier2003_2016update/'
        
        metallicity_ids = [22, 32, 42, 52, 62, 72, 82]
        metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])
        
        num_metallicities = len(metallicities) # 7 for 2016 update (and 2013)
        num_stellar_ages = 221
        num_wavelengths = 2023 # 1221 for 2003 original, 1238 for 2013 update,
                               # and 2023 for 2016 update
        
        # create a datacube which will hold all the spectra for every
        # metallicity+age combination
        datacube = np.zeros((num_metallicities, num_stellar_ages, num_wavelengths))
        
        # create a mass cube which will hold the remaining stellar mass for
        # every metallicity+age combination
        masses = np.zeros((num_metallicities, num_stellar_ages))
        
        for k in range(num_metallicities) :
            infile = inDir + 'bc2003_lr_BaSeL_m{}_chab_ssp.ised_ASCII'.format(
                metallicity_ids[k])
            with open(infile, 'r') as f :
                # the first line has the stellar ages
                stellar_ages = np.array(f.readline().split()[1:], dtype=np.float64)
                
                # the next 5 lines are not needed
                for i in range(5) :
                    f.readline()
                
                # the next line has the wavelengths
                wavelengths = np.array(f.readline().split()[1:], dtype=np.float64)
                
                # the next 221 lines are spectra
                for j in range(num_stellar_ages) :
                    line = f.readline()
                    cur_sed = np.array(line.split()[1:], dtype=np.float64)
                    
                    # there are some extra data points after each spectrum,
                    # so we truncate to the expected number of wavelengths
                    datacube[k, j, :] = cur_sed[:num_wavelengths]
                
                # there are 12 more lines after this (with 221 values per line)
                # that we don't need, though the remaining stellar mass is on
                # the second (index 1) line of these 12 lines
                f.readline()
                masses[k] = np.array(f.readline().split()[1:], dtype=np.float64)
        
        # save the file for faster loading in future
        with h5py.File(datacube_file, 'w') as hf :
            hf.create_dataset('datacube', data=datacube)
            hf.create_dataset('metallicities', data=metallicities)
            hf.create_dataset('stellar_ages', data=stellar_ages)
            hf.create_dataset('wavelengths', data=wavelengths)
            hf.create_dataset('masses', data=masses)
    else :
        # load the saved arrays
        with h5py.File(datacube_file, 'r') as hf :
            datacube = hf['datacube'][:]
            metallicities = hf['metallicities'][:]
            stellar_ages = hf['stellar_ages'][:]
            wavelengths = hf['wavelengths'][:]
            masses = hf['masses'][:]
    
    return datacube, metallicities, stellar_ages, wavelengths, masses
