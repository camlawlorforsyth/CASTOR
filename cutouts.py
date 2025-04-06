
import os
import numpy as np

import astropy.constants as c
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u

from core import load_massive_galaxy_sample
from noise import get_noise

def create_all_mocks(model_redshift=0.5, psf=0.15*u.arcsec) :
    
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
    for subID, snap in zip(sample['subID'], sample['snapshot']) :
        outfile = 'cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))
        if not os.path.exists(outfile) :
            create_mock_observations(snap, subID, psf=psf,
                model_redshift=model_redshift)
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def create_mock_observations(snap, subID, model_redshift=0.5, psf=0.15*u.arcsec,
                             display=False, save=True) :
    
    # open the input GALAXEV file, and get the plate scale for the images
    infile = 'GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))
    with fits.open(infile) as hdu :
        hdr = hdu[0].header
        data = hdu[0].data*u.Jy # Jy [per pixel]
    plate_scale = hdr['CDELT1']*u.arcsec/u.pix # the plate scale of the images
    assert hdr['REDSHIFT'] == model_redshift
    
    # get all the parameters for every telescope
    dictionary = get_noise()
    
    # define the telescope diameters and collecting areas for CASTOR and Roman
    diameters = np.array([100, 100, 100, 100, 100, 236, 236, 236, 236])*u.cm
    areas = np.pi*np.square(diameters/2)
    
    # define the exposure durations for the CASTOR ultradeep survey and the
    # Roman HLWAS
    exposures =  np.array([180000, 180000, 180000, 180000, 360000,
                           146, 146, 146, 146])*u.s
    
    '''
    # telescope diameters from
    # https://www.castormission.org/mission
    # https://www.euclid-ec.org/public/mission/telescope/
    # https://www.jwst.nasa.gov/content/forScientists/
    # faqScientists.html#collectingarea
    # https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-telescope
    # https://roman.ipac.caltech.edu/sims/Param_db.html
    
    observatory, depth = telescope.split('_')
    if observatory == 'castor' :
        area = np.pi*np.square(100*u.cm/2)
        if depth == 'wide' :
            exposures = np.array([1000, 1000, 1000, 1000, 2000])*u.s
        elif depth == 'deep' :
            exposures =  np.array([18000, 18000, 18000, 18000, 36000])*u.s
        elif depth == 'ultradeep' :
            exposures =  np.array([180000, 180000, 180000, 180000, 360000])*u.s
    elif observatory == 'euclid' :
        area = np.pi*np.square(120*u.cm/2)
        if depth == 'wide' : # see Euclid Collab.+Mellier+2024, Table 3 for wide
            exposures = np.array([2454, 448, 448, 448])*u.s # survey details
        elif depth == 'deep' : # assume deep survey repeats wide survey 40 times
            exposures = np.array([98160, 17920, 17920, 17920])*u.s
    elif observatory == 'hst' :
        area = np.pi*np.square(240*u.cm/2)
        if depth == 'hff' :
            # get the maximum exposure time for each HFF filter
            with h5py.File('background/HFF_exposure_times.hdf5', 'r') as hf :
                exps = hf['exposures'][:]
            exps = np.nanmax(exps, axis=1)
            exposures = np.concatenate([exps[:1], exps[:5], exps[4:]])*u.s
        elif depth == 'deep' : # assume 30 hrs per filter for now
            exposures = 108000*np.ones(19)*u.s
    elif observatory == 'jwst' :
        area = np.pi*np.square(578.673*u.cm/2)
        if depth == 'deep' : # assume 10 hrs per filter for now
            exposures = 36000*np.ones(18)*u.s
    elif observatory == 'roman' :
        area = np.pi*np.square(236*u.cm/2)
        if depth == 'hlwas' :
            exposures = 146*np.ones(8)*u.s
    '''
    
    # get the filters
    filters = [hdr['FILTER{}'.format(i)] for i in range(len(hdr['FILTER*']))]
    # filters = [key for key in dictionary.keys() if telescope.split('_')[0] in key]
    num_filters = len(filters)
    
    # get certain attributes of the filters
    pivots = np.full(num_filters, np.nan)*u.um
    widths = np.full(num_filters, np.nan)*u.um
    backgrounds = np.full(num_filters, np.nan)*u.mag/np.square(u.arcsec)
    dark_currents = np.full(num_filters, np.nan)*u.electron/u.s/u.pix
    read_noises = np.full(num_filters, np.nan)*u.electron/u.pix
    for i, filt in enumerate(filters) :
        pivots[i] = dictionary[filt]['pivot']
        widths[i] = dictionary[filt]['fwhm']
        backgrounds[i] = dictionary[filt]['background']
        dark_currents[i] = dictionary[filt]['dark_current']
        read_noises[i] = dictionary[filt]['read_noise']
    
    # get the sky background levels in Jy/arcsec^2
    bkg_Jy = mag_to_Jy(backgrounds)
    
    # get the throughputs at the pivot wavelengths
    throughputs = np.full(num_filters, np.nan)
    for i, (filt, pivot) in enumerate(zip(filters, pivots)) :
        array = np.genfromtxt('passbands/passbands_micron/' + filt + '.txt')
        waves, response = array[:, 0]*u.um, array[:, 1]
        throughputs[i] = np.interp(pivot, waves, response)
    
    # get the area of a pixel on the sky, in arcsec^2
    pixel_area = np.square(plate_scale*u.pix)
    
    # calculate the conversion factor PHOTFNU to get janskys [per pixel]
    # from spatial electron flux electron/s/cm^2 [per pixel]
    photfnus = calculate_photfnu(1*u.electron/u.s/np.square(u.cm), pivots,
        widths, throughputs)
    
    # get the background electrons per second per pixel
    Bsky = bkg_Jy*areas*pixel_area/photfnus
    
    # get the background electrons per pixel over the entire exposure
    background_electrons = Bsky*exposures
    
    # get the dark current electrons per second per pixel
    Bdet = dark_currents*u.pix
    
    # get the dark current electrons per pixel over the entire exposure
    detector_electrons = Bdet*exposures
    
    # get the number of reads, limiting a given exposure to 1000 s, as longer
    # exposures than 1000 s will be dominated by cosmic rays
    single_exposure = 1000*u.s
    Nreads = np.ceil(exposures/single_exposure)
    
    # get the read noise electrons per pixel
    read_electrons = read_noises*u.pix
    
    # get the total non-source noise per pixel over the entire exposure
    nonsource_level = background_electrons + detector_electrons
    
    # check the brightness of the galaxy in the given bands
    # mags = []
    # for frame in data :
    #     print(np.sum(frame*pixel_area).to(u.Jy))
    #     m_AB = -2.5*np.log10(np.sum(frame*pixel_area).to(u.Jy)/(3631*u.Jy))*u.mag
    #     mags.append(m_AB.value)
    # print(mags)
    
    # convert the PSF FWHM (that we'll use to convolve the images) into pixels
    sigma = psf/(2*np.sqrt(2*np.log(2))) # arcseconds
    sigma_pix = sigma/plate_scale # pixels
    
    output = np.zeros((2*data.shape[0], data.shape[1], data.shape[2]))
    for i, (filt, frame, pivot, width, throughput, exposure, area, level, Nread,
        RR, photfnu) in enumerate(zip(filters, data, pivots, widths, throughputs,
        exposures, areas, nonsource_level, Nreads, read_electrons, photfnus)) :
        
        # the noiseless synthetic GALAXEV images are already in convenient units
        # frame = frame.to(u.Jy) # Jy [per pixel]
        
        # get the noiseless synthetic GALAXEV image
        image = frame*exposure*area/photfnu # electron [per pixel]
        
        # define the convolution kernel and convolve the image
        kernel = Gaussian2DKernel(sigma_pix.value)
        convolved = convolve(image.value, kernel)*u.electron # electron [per pixel]
        
        # add the non-source level to the convolved image
        noisey = convolved + level # electron [per pixel]
        
        # sample from a Poisson distribution with the noisey data
        sampled = np.random.poisson(noisey.value)*u.electron # electron [per pixel]
        
        # add the RMS noise value
        sampled = np.random.normal(sampled.value,
            scale=np.sqrt(Nread)*RR.value)*u.electron # electron [per pixel]
        
        # subtract the background from the sampled image
        subtracted = sampled - level # electron [per pixel]
        
        # determine the final noise
        noise = np.sqrt(noisey.value +
            Nread*np.square(RR.value))*u.electron # electron [per pixel]
        
        # convert back to janskys [per pixel]
        subtracted = subtracted/exposure/area*photfnu # Jy [per pixel]
        noise = noise/exposure/area*photfnu # Jy [per pixel]
        
        # place the constructed images into the final array
        output[2*i] = subtracted
        output[2*i+1] = noise
    
    # save the output to file
    if save :
        outDir = 'cutouts/'
        os.makedirs(outDir, exist_ok=True)
        
        # set the output filename
        outfile = 'cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))
        
        # add entries tracking what each frame displays
        data_list = []
        for i in range(num_filters) :
            data_list.append(filters[i])
            data_list.append(filters[i] + '_noise')
        for i, val in enumerate(data_list) :
            hdr['FRAME{}'.format(i)] = (val,
                'Frame index = {}'.format(i))
        
        # add entries for the exposure durations, areas, and photfnu factors
        for i, exposure in enumerate(exposures) :
            hdr['EXPTIME{}'.format(i)] = (exposure.value,
                'exposure duration (seconds)--calculated')
        for i, area in enumerate(areas) :
            hdr['AREA{}'.format(i)] = (area.value,
                'detector area (cm2)--calculated')
        for i, photfnu in enumerate(photfnus) :
            hdr['PHOTFNU{}'.format(i)] = (photfnu.value,
                'inverse sensitivity, Jy*sec*cm2/electron')
        
        hdulist = fits.HDUList([fits.PrimaryHDU(data=output, header=hdr)])
        hdulist.writeto(outfile)
        
        '''
        # outfile = '{}_{}.fits'.format(filt, telescope.split('_')[1])
        # save_cutout(subtracted.value, outDir + outfile, exposure.value,
        #             area.value, photfnu.value, plate_scale.value, redshift)
        # noise_outfile = '{}_{}_noise.fits'.format(filt, telescope.split('_')[1])
        # save_cutout(noise.value, outDir + noise_outfile, exposure.value,
        #             area.value, photfnu.value, plate_scale.value, redshift)
        
        # os.makedirs(outDir + '/snr_maps/', exist_ok=True)
        # snr_outfile = '{}_{}_snr.png'.format(filt, telescope.split('_')[1])
        # plt.display_image_simple(subtracted.value/noise.value,
        #     lognorm=False, vmin=0.5, vmax=10, save=True,
        #     outfile=outDir + '/snr_maps/' + snr_outfile)
        
        # save noiseless images in the correct units for testing
        # save_cutout(image.value, outDir + outfile, exposure.value,
        #             area.value, photfnu.value, plate_scale.value, redshift)
        # noise = np.full(image.shape, 0.0)*u.electron
        # save_cutout(noise.value, outDir + noise_outfile, exposure.value,
        #             area.value, photfnu.value, plate_scale.value, redshift)
        '''
    
    return

def calculate_photfnu(electron_flux, lam_pivot, delta_lam,
                      throughput, gain=1*u.electron/u.photon) :
    
    lam_pivot = lam_pivot.to(u.m) # convert from um to m
    delta_lam = delta_lam.to(u.m) # convert from um to m
    
    # difference in wavelength to difference in frequency
    delta_nu = (c.c*delta_lam/np.square(lam_pivot)).to(u.Hz)
    
    # calculate the photon flux in photons/s/cm^2/Hz
    photnu = electron_flux/throughput/delta_nu/gain
    
    # calculate the flux density in janskys
    photfnu = photnu.to(u.Jy, equivalencies=u.spectral_density(lam_pivot))
    
    return photfnu*u.s/u.electron*np.square(u.cm)

def mag_to_Jy(mag) :
    # convert AB mag/arcsec^2 to Jy/arcsec^2
    mag = mag.to(u.mag/np.square(u.arcsec))
    return np.power(10, -0.4*(mag.value - 8.9))*u.Jy/np.square(u.arcsec)
