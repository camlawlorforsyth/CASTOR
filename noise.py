
import os
from os.path import exists
import numpy as np
import pickle

import astropy.constants as c
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import h5py

import filters
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def add_noise_and_psf(subID, telescope, population='quenched',
                      psf=0.15*u.arcsec, display=False, save=True) :
    
    # telescope diameters from
    # https://www.castormission.org/mission
    # https://www.jwst.nasa.gov/content/forScientists/
    # faqScientists.html#collectingarea
    # https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-telescope
    # https://roman.ipac.caltech.edu/sims/Param_db.html
    
    inDir = 'SKIRT/SKIRT_output_quenched/'
    outDir = 'cutouts/{}/{}/'.format(population, subID)
    passbandsDir = 'passbands/passbands_micron/'
    
    # open the SKIRT output file, and get the plate scale for the SKIRT images
    infile = '{}{}/{}_{}_total.fits'.format(inDir, subID, subID,
        telescope.split('_')[0].upper())
    with fits.open(infile) as hdu :
        plate_scale = (hdu[0].header)['CDELT1']*u.arcsec/u.pix
        data = hdu[0].data*u.MJy/u.sr
    
    # get all the parameters for every telescope
    dictionary = get_noise()
    
    if telescope == 'castor_wide' :
        exposures = np.array([1000, 1000, 2000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'castor_deep' :
        exposures =  np.array([18000, 18000, 36000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'castor_ultradeep' :
        exposures =  np.array([180000, 180000, 360000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'hst_hff' :
        # get the maximum exposure time for each HFF filter
        with h5py.File('background/HFF_exposure_times.hdf5', 'r') as hf :
            exps = hf['exposures'][:]
        exps = np.nanmax(exps, axis=1)
        exposures = np.concatenate([exps[:1], exps[:5], exps[4:]])*u.s
        area = np.pi*np.square(240*u.cm/2)
    elif telescope == 'hst_deep' : # assume 30 hrs per filter for now
        exposures = 108000*np.ones(19)*u.s
        area = np.pi*np.square(240*u.cm/2)
    elif telescope == 'jwst_deep' : # assume 10 hrs per filter for now
        exposures = 36000*np.ones(18)*u.s
        area = np.pi*np.square(578.673*u.cm/2)
    elif telescope == 'roman_hlwas' :
        exposures = 146*np.ones(8)*u.s
        area = np.pi*np.square(236*u.cm/2)
    
    # get the filter names for the telescope
    filters = [key for key in dictionary.keys() if telescope.split('_')[0] in key]
    length = len(filters)
    
    # get certain attributes of the filters
    pivots = np.full(length, np.nan)*u.um
    widths = np.full(length, np.nan)*u.um
    backgrounds = np.full(length, np.nan)*u.mag/np.square(u.arcsec)
    dark_currents = np.full(length, np.nan)*u.electron/u.s/u.pix
    read_noises = np.full(length, np.nan)*u.electron/u.pix
    for i, filt in enumerate(filters) :
        pivots[i] = dictionary[filt]['pivot']
        widths[i] = dictionary[filt]['fwhm']
        backgrounds[i] = dictionary[filt]['background']
        dark_currents[i] = dictionary[filt]['dark_current']
        read_noises[i] = dictionary[filt]['read_noise']
    
    # get the sky background levels in Jy/arcsec^2
    bkg_Jy = mag_to_Jy(backgrounds)
    
    # get the throughputs at the pivot wavelengths
    throughputs = np.full(length, np.nan)
    for i, (filt, pivot) in enumerate(zip(filters, pivots)) :
        array = np.genfromtxt(passbandsDir + filt + '.txt')
        waves, response = array[:, 0]*u.um, array[:, 1]
        throughputs[i] = np.interp(pivot, waves, response)
    
    # get the area of a pixel on the sky, in arcsec^2
    pixel_area = np.square(plate_scale*u.pix)
    
    # calculate the conversion factor PHOTFNU to get janskys [per pixel]
    # from spatial electron flux electron/s/cm^2 [per pixel]
    photfnus = calculate_photfnu(1*u.electron/u.s/np.square(u.cm), pivots,
        widths, throughputs)
    
    # get the background electrons per second per pixel
    Bsky = bkg_Jy*area*pixel_area/photfnus
    
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
    
    for (filt, frame, pivot, width, throughput, exposure, level, Nread, RR,
         photfnu) in zip(filters, data, pivots, widths, throughputs, exposures,
                         nonsource_level, Nreads, read_electrons, photfnus) :
                         
        # convert the noiseless synthetic SKIRT image to convenient units
        frame = frame.to(u.Jy/np.square(u.arcsec)) # Jy/arcsec^2 [per pixel]
        
        # get the noiseless synthetic SKIRT image
        image = frame*exposure*area*pixel_area/photfnu # electron [per pixel]
        # plt.display_image_simple(image.value, vmin=None, vmax=None)
        
        # define the convolution kernel and convolve the image
        kernel = Gaussian2DKernel(sigma_pix.value)
        convolved = convolve(image.value, kernel)*u.electron # electron [per pixel]
        # plt.display_image_simple(convolved.value, vmin=None, vmax=None)
        
        # add the non-source level to the convolved image
        noisey = convolved + level # electron [per pixel]
        # plt.display_image_simple(noisey.value, vmin=None, vmax=None)
        
        # sample from a Poisson distribution with the noisey data
        sampled = np.random.poisson(noisey.value)*u.electron # electron [per pixel]
        # plt.display_image_simple(sampled.value, vmin=None, vmax=None)
        
        # add the RMS noise value
        sampled = np.random.normal(sampled.value,
            scale=np.sqrt(Nread)*RR.value)*u.electron # electron [per pixel]
        # plt.display_image_simple(sampled.value, vmin=None, vmax=None)
        
        # subtract the background from the sampled image
        subtracted = sampled - level # electron [per pixel]
        # plt.display_image_simple(subtracted.value, vmin=None, vmax=None)
        
        # determine the final noise
        noise = np.sqrt(noisey.value +
            Nread*np.square(RR.value))*u.electron # electron [per pixel]
        
        # convert back to janskys [per pixel]
        # subtracted = subtracted/exposure/area*photfnu
        # noise = noise/exposure/area*photfnu
        # if display :
        #     plt.display_image_simple(final.value, vmin=1e-10, vmax=1e-6)
        #     plt.display_image_simple(noise.value, vmin=1e-9, vmax=1e-8)
        
        # save the output to file
        if save :
            os.makedirs(outDir, exist_ok=True)
            
            outfile = '{}_{}.fits'.format(filt, telescope.split('_')[1])
            save_cutout(subtracted.value, outDir + outfile, exposure.value,
                        area.value, photfnu.value, plate_scale.value, 0.5)
            
            noise_outfile = '{}_{}_noise.fits'.format(filt, telescope.split('_')[1])
            save_cutout(noise.value, outDir + noise_outfile, exposure.value,
                        area.value, photfnu.value, plate_scale.value, 0.5)
            
            os.makedirs(outDir + '/snr_maps/', exist_ok=True)
            snr_outfile = '{}_{}_snr.png'.format(filt, telescope.split('_')[1])
            plt.display_image_simple(subtracted.value/noise.value,
                lognorm=False, vmin=0.5, vmax=10, save=True,
                outfile=outDir + '/snr_maps/' + snr_outfile)
    
    return

def background_castor() :
    
    # adapted heavily from
    # https://github.com/CASTOR-telescope/ETC/blob/master/castor_etc/background.py
    
    # with additional discussion about renormalizing the zodiacal file from
    # https://github.com/CASTOR-telescope/ETC/tree/master/castor_etc/data/sky_background
    
    # from CASTOR ETC:
    # etc = np.array([27.72748, 24.24196, 22.58821])
    
    inDir = 'passbands/'
    filters = ['castor_uv', 'castor_u', 'castor_g']
    
    return determine_leo_background(inDir, filters)

def background_hst() :
    
    inDir = 'passbands/'
    filters = ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w',  'hst_f390w',
               'hst_f438w', 'hst_f435w', 'hst_f475w', 'hst_f555w',  'hst_f606w',
               'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp', 'hst_f105w',
               'hst_f110w', 'hst_f125w', 'hst_f140w', 'hst_f160w']
    
    return determine_leo_background(inDir, filters)

def background_jwst() :
    
    inDir = 'passbands/'
    filters = ['jwst_f070w',  'jwst_f090w',  'jwst_f115w',  'jwst_f150w',
               'jwst_f200w',  'jwst_f277w',  'jwst_f356w',  'jwst_f410m',
               'jwst_f444w',  'jwst_f560w',  'jwst_f770w',  'jwst_f1000w',
               'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w',
               'jwst_f2100w', 'jwst_f2550w']
    
    return determine_l2_background(inDir, filters)

def background_roman() :
    
    inDir = 'passbands/'
    filters = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
               'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213']
    
    return determine_l2_background(inDir, filters)

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

def components_l2_background(waves) :
    # Sun-Earth L_2 Lagrange point
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-calculations-page-overview/jwst-etc-backgrounds
    
    # https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model
    
    # https://jwst-docs.stsci.edu/jwst-other-tools/jwst-backgrounds-tool
    
    # https://github.com/spacetelescope/jwst_backgrounds/blob/master/
    # jwst_backgrounds/jbt.py
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-pandeia-engine-tutorial/pandeia-backgrounds
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-pandeia-engine-tutorial/pandeia-configuration-dictionaries
    
    if exists('background/minzodi_median.txt') :
        
        # total background, sum of in-field zodiacal light, in-field galactic
        # light, stray light, and thermal self-emission
        tab = Table.read('background/minzodi_median.txt', format='ascii')
        bkg_waves = tab['WAVELENGTH(micron)'].value*u.um
        bkg_fnu = tab['TOTALBACKGROUND(MJy/sr)'].value*u.MJy/u.sr
        
        # convert units in preparation of computing flam values
        bkg_waves = bkg_waves.to(u.AA)
        bkg_fnu = bkg_fnu.to(u.erg/u.s/u.Hz/np.square(u.cm*u.arcsec))
        
        # determine flam values
        bkg_flam = (c.c.to(u.AA/u.s))/np.square(bkg_waves)*bkg_fnu
        
        # interpolate those values at the specified wavelengths
        sky_background_flam = np.interp(waves, bkg_waves, bkg_flam)
    else :
        inDir = 'background/bathtubs/'
        
        means = np.full(2961, 0.0)   # in MJy/sr
        medians = np.full(2961, 0.0) # in MJy/sr
        
        waves = np.linspace(0.5, 30.1, 2961)
        
        # loop over every wavelength
        for i, wave in enumerate(waves) :
            # get the background values at that wavelength
            bkgs_at_wave = np.genfromtxt(
                inDir + 'bathtub_{:.2f}_micron.txt'.format(wave))[:, 1]
            
            # add the mean and median into the master arrays
            means[i] = np.mean(bkgs_at_wave)
            medians[i] = np.median(bkgs_at_wave)
        
        # jwst_backgrounds.jbt doesn't support wavelengths below 0.5 micron,
        # so we'll fit a linear function to the linear data, and sample at
        # the wavelengths we want, to account for Roman's F062 filter
        means_fit = np.polyfit(waves[:11], means[:11], 1)
        medians_fit = np.polyfit(waves[:11], medians[:11], 1)
        
        waves_front = np.linspace(0.4, 0.49, 10)
        means_front = means_fit[0]*waves_front + means_fit[1]
        medians_front = medians_fit[0]*waves_front + medians_fit[1]
        
        waves = np.concatenate([waves_front, waves])
        means = np.concatenate([means_front, means])
        medians = np.concatenate([medians_front, medians])
        
        means_array = np.array([waves, means]).T
        medians_array = np.array([waves, medians]).T
        
        np.savetxt('background/minzodi_mean.txt', means_array,
                   header='WAVELENGTH(micron) TOTALBACKGROUND(MJy/sr)')
        np.savetxt('background/minzodi_median.txt', medians_array,
                   header='WAVELENGTH(micron) TOTALBACKGROUND(MJy/sr)')
    
    return sky_background_flam

def components_leo_background(waves) :
    # low Earth orbit
    
    # https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
    
    # https://hst-docs.stsci.edu/acsihb/chapter-9-exposure-time-calculations/
    # 9-4-detector-and-sky-backgrounds
    
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-9-wfc3-exposure-time-calculation/
    # 9-7-sky-background
    
    # earthshine_model_001.fits from
    # https://ssb.stsci.edu/cdbs/work/etc/etc-cdbs/background/
    
    # earthshine
    es_tab = Table.read('background/earthshine_model_001.fits')
    es_waves = es_tab['Wavelength'].value*u.AA
    es_flam = es_tab['FLUX'].value*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA
    earthshine = np.interp(waves, es_waves, es_flam)
    
    # zodiacal_model_001.fits from
    # https://ssb.stsci.edu/cdbs/work/etc/etc-cdbs/background/
    
    # zodiacal renormalization? ->
    # https://github.com/gbrammer/wfc3/blob/master/etc_zodi.py#L61-L67
    
    # zodiacal
    zodi_tab = Table.read('background/zodiacal_model_001.fits')
    zodi_waves = zodi_tab['WAVELENGTH'].value*u.AA
    zodi_flam = zodi_tab['FLUX'].value*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA
    zodiacal = np.interp(waves, zodi_waves, zodi_flam)
    
    # geocoronal due to the [O II] 2471 A line
    geo_waves = np.linspace(2470, 2472, 201)*u.AA
    central = 2471*u.AA
    fwhm = 0.023*u.AA
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    flam = 1.5e-17*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA # low value
    gaussian = flam*np.exp(-0.5*np.square((geo_waves - central)/sigma))
    geocoronal = np.interp(waves, geo_waves, gaussian)
    
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/
    # 7-9-other-considerations-for-ir-imaging
    
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/
    # wfc3/documentation/instrument-science-reports-isrs/_documents/2014/WFC3-2014-03.pdf
    
    # airglow due to the He I 10830 A line
    airglow_waves = np.linspace(10795, 10865, 6001)*u.AA
    central = 10830*u.AA
    fwhm = 2*u.AA # from ETC User Manual, "Specifying the Appropriate Background"
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    flam = 0.1*1500*3.7e-14/10830*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA # avg value
    gaussian = flam*np.exp(-0.5*np.square((airglow_waves - central)/sigma))
    airglow = np.interp(waves, airglow_waves, gaussian)
    
    return earthshine, zodiacal, geocoronal, airglow

def determine_l2_background(inDir, filters) :
    
    bkg = []
    for filt in filters :
        # get the wavelengths and response curves for a given filter
        array = np.genfromtxt(inDir + filt + '.txt')
        waves, response = (array[:, 0]*u.um).to(u.AA), array[:, 1]
        
        # super sample the wavelengths and response at 1 angstrom intervals
        lam = np.arange(waves[0].value, waves[-1].value + 1, 1)*u.AA
        response = np.interp(lam.value, waves.value, response)
        
        # get the summed components of the background
        sky_background_flam = components_l2_background(lam)
        
        # ensure the units are correct
        sky_background_flam = sky_background_flam.to(
            u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA)
        
        # integrate the background with the bandpass, and append the value
        mags_per_sq_arcsec = flam_to_mag(lam, sky_background_flam, response)
        bkg.append(mags_per_sq_arcsec)
    
    return np.array(bkg)

def determine_leo_background(inDir, filters) :
    
    bkg = []
    for filt in filters :
        # get the wavelengths and response curves for a given filter
        array = np.genfromtxt(inDir + filt + '.txt')
        waves, response = (array[:, 0]*u.um).to(u.AA), array[:, 1]
        
        # super sample the wavelengths and response at 1 angstrom intervals
        lam = np.arange(waves[0].value, waves[-1].value + 1, 1)*u.AA
        response = np.interp(lam.value, waves.value, response)
        
        # get the components of the background
        earthshine, zodiacal, geocoronal, airglow = components_leo_background(lam)
        
        # sum those components
        sky_background_flam = earthshine + zodiacal + geocoronal + airglow
        
        # integrate the background with the bandpass, and append the value
        mags_per_sq_arcsec = flam_to_mag(lam, sky_background_flam, response)
        bkg.append(mags_per_sq_arcsec)
    
    return np.array(bkg)

def determine_noise_components() :
    
    dictionary = filters.calculate_psfs()
    
    # CASTOR plate scales
    castor_filts = [key for key in dictionary.keys() if 'castor' in key]
    castor_scales = 0.1*np.ones(3)*u.arcsec/u.pix
    castor_bkg = background_castor()*u.mag/np.square(u.arcsec)
    castor_dark = np.array([0.00042, 0.00042, 0.002])*u.electron/u.s/u.pix
    castor_read = np.array([3.15, 3.15, 4.45])*u.electron/u.pix
    castor_scalings = np.square(castor_scales)/np.square(0.05*u.arcsec/u.pix)
    castor_dark = castor_dark/castor_scalings
    castor_read = castor_read/castor_scalings
    for filt, bkg, dark, read in zip(castor_filts, castor_bkg, castor_dark,
                                     castor_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # HST plate scales
    hst_filts = [key for key in dictionary.keys() if 'hst' in key]
    hst_scales = np.array([0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.128, 0.128, 0.128, 0.128, 0.128])*u.arcsec/u.pix
    hst_bkg = background_hst()*u.mag/np.square(u.arcsec)
    hst_dark = np.array([0.00306, 0.00306, 0.00306, 0.00306, 0.00306, 0.00306,
                         0.0153, 0.0153, 0.0153, 0.0153, 0.0153, 0.0153, 0.0153,
                         0.0153, 0.048, 0.048, 0.048, 0.048,
                         0.048])*u.electron/u.s/u.pix
    hst_read = np.array([3.15, 3.15, 3.15, 3.15, 3.15, 3.15, 4.45, 4.45, 4.45,
                         4.45, 4.45, 4.45, 4.45, 4.45, 12.0, 12.0, 12.0, 12.0,
                         12.0])*u.electron/u.pix
    hst_scalings = np.square(hst_scales)/np.square(0.05*u.arcsec/u.pix)
    hst_dark = hst_dark/hst_scalings
    hst_read = hst_read/hst_scalings
    for filt, bkg, dark, read in zip(hst_filts, hst_bkg, hst_dark, hst_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # JWST plate scales
    jwst_filts = [key for key in dictionary.keys() if 'jwst' in key]
    jwst_scales = np.array([0.031, 0.031, 0.031, 0.031, 0.031, 0.063, 0.063,
                            0.063, 0.063, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
                            0.11, 0.11, 0.11])*u.arcsec/u.pix
    jwst_bkg = background_jwst()*u.mag/np.square(u.arcsec)
    jwst_dark = np.array([0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0342,
                          0.0342, 0.0342, 0.0342, 0.2, 0.2, 0.2, 0.2, 0.2, 
                          0.2, 0.2, 0.2, 0.2])*u.electron/u.s/u.pix
    jwst_read = np.array([15.77, 15.77, 15.77, 15.77, 15.77, 13.25, 13.25,
                          13.25, 13.25, 14, 14, 14, 14, 14, 14, 14, 14,
                          14])*u.electron/u.pix
    jwst_scalings = np.square(jwst_scales)/np.square(0.05*u.arcsec/u.pix)
    jwst_dark = jwst_dark/jwst_scalings
    jwst_read = jwst_read/jwst_scalings
    for filt, bkg, dark, read in zip(jwst_filts, jwst_bkg, jwst_dark, jwst_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # Roman plate scales
    roman_filts = [key for key in dictionary.keys() if 'roman' in key]
    roman_scales = 0.11*np.ones(8)*u.arcsec/u.pix
    roman_bkg = background_roman()*u.mag/np.square(u.arcsec)
    roman_dark = 0.005*np.ones(8)*u.electron/u.s/u.pix
    roman_read = 15.5*np.ones(8)*u.electron/u.pix
    roman_scalings = np.square(roman_scales)/np.square(0.05*u.arcsec/u.pix)
    roman_dark = roman_dark/roman_scalings
    roman_read = roman_read/roman_scalings
    for filt, bkg, dark, read in zip(roman_filts, roman_bkg, roman_dark,
                                     roman_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    return dictionary

def flam_to_mag(waves, flam, response) :
    
    # from Eq. 2 of Bessell & Murphy 2012
    numer = np.trapz(flam*response*waves*np.square(u.arcsec), x=waves)
    denom = np.trapz(response/waves, x=waves)
    const = c.c.to(u.AA/u.s)
    
    fnu = (numer/const/denom).value
    
    return -2.5*np.log10(fnu) - 48.6

def get_noise(pretty_print=False) :
    
    infile = 'noise/noise_components.txt'
    
    if exists(infile) :
        with open(infile, 'rb') as file :
            dictionary = pickle.load(file)
    else :
        with open(infile, 'wb') as file :
            pickle.dump(determine_noise_components(), file)
    
    if pretty_print :
        import pprint
        pprint.pprint(dictionary)
    
    return dictionary

def hff_exposure_time_vs_lam() :
    
    # get the average exposure time for each HFF filter
    with h5py.File('background/HFF_exposure_times.hdf5', 'r') as hf :
        exposures = hf['exposures'][:]
    # means = np.nanmean(exposures, axis=1)
    # medians = np.nanmedian(exposures, axis=1)
    exposures[np.isnan(exposures)] = 0 # ignore NaNs for plotting
    
    # means is larger than np.mean(exposures, axis=1)
    # medians is larger than np.median(exposures, axis=1)
    
    # get the pivot wavelengths
    dd = filters.calculate_psfs()
    pivs = [dd[key]['pivot'].value for key in dd.keys() if 'hst' in key]
    pivs = np.concatenate([pivs[1:5], pivs[6:]])
    
    # prepare values for plotting
    xs = [pivs, pivs, pivs, pivs, pivs, pivs]
    ys = np.array([exposures[:, 0], exposures[:, 1], exposures[:, 2],
                   exposures[:, 3], exposures[:, 4], exposures[:, 5]])/3600
    labels = ['a370', 'a1063', 'a2744', 'm416', 'm717', 'm1149']
    colors = ['k', 'r', 'b', 'm', 'g', 'c']
    markers = ['', '', '', '', '', '']
    styles = ['-', '--', ':', '-.', '-', '--']
    alphas = np.ones(6)
    
    # plot the exposure times for each HFF cluster
    plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
        xlabel='Wavelength (um)', ylabel='Average Exposure Time (hr)', loc=2,
        scale='linear')
    
    # create a table for visual inspection
    table = Table([column for column in exposures.T], names=labels)
    table.pprint(max_width=-1)
    
    times = np.nanmax(exposures, axis=1)
    
    return

def mag_to_Jy(mag) :
    # convert AB mag/arcsec^2 to Jy/arcsec^2
    mag = mag.to(u.mag/np.square(u.arcsec))
    return np.power(10, -0.4*(mag.value - 8.9))*u.Jy/np.square(u.arcsec)

def process_everything(population='quenched') :
    
    inDir = 'SKIRT/SKIRT_output_{}/'.format(population)
    subIDs = np.sort(np.int_(os.listdir(inDir)))
    
    for subID in subIDs :
        if subID != 14 :
            add_noise_and_psf(subID, 'castor_ultradeep')
            add_noise_and_psf(subID, 'roman_hlwas')
            # add_noise_and_psf(subID, 'hst_deep')
            # add_noise_and_psf(subID, 'jwst_deep')
    
    return

def save_cutout(data, outfile, exposure, det_area, photfnu, scale, redshift) :
    
    hdu = fits.PrimaryHDU(data)
    
    hdr = hdu.header
    hdr['Z'] = redshift
    hdr.comments['Z'] = 'object spectroscopic redshift--by definition'
    hdr['EXPTIME'] = exposure
    hdr.comments['EXPTIME'] = 'exposure duration (seconds)--calculated'
    hdr['AREA'] = det_area
    hdr.comments['AREA'] = 'detector area (cm2)--calculated'
    hdr['PHOTFNU'] = photfnu
    hdr.comments['PHOTFNU'] = 'inverse sensitivity, Jy*sec*cm2/electron'
    hdr['SCALE'] = scale
    hdr.comments['SCALE'] = 'Pixel size (arcsec) of output image'
    hdr['BUNIT'] = 'electron'
    hdr.comments['BUNIT'] = 'Physical unit of the array values'
    
    hdu.writeto(outfile)
    
    return

def get_largest_psf(telescopes, not_miri=True) :
    
    # get the dictionary of all the filter parameters
    telescope_params = get_noise(pretty_print=0)
    all_filters = telescope_params.keys()
    
    # get all the filters from the specified telescopes
    filters = []
    for telescope in telescopes :
        telescope = telescope.split('_')[0]
        filters.append([key for key in all_filters if telescope in key])
    filters = np.concatenate(filters)
    
    # get the pivot wavelengths and PSFs for the relevant filters
    pivots = np.full(len(filters), np.nan)*u.um
    psfs = np.full(len(filters), np.nan)*u.arcsec
    for i, filt in enumerate(filters) :
        pivots[i] = telescope_params[filt]['pivot']
        psfs[i] = telescope_params[filt]['psf']
    
    if not_miri :
        mask = (pivots < 5*u.um)
        pivots = pivots[mask]
        psfs = psfs[mask]
    
    print(np.max(psfs)) # the Roman F213 filter (which is not part of the HLWAS)
                        # is breaking this
    
    return
