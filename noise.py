
from os.path import exists
import numpy as np
import pickle

import astropy.constants as c
from astropy.table import Table
import astropy.units as u
import h5py

import filters
import plotting as plt

def background_castor() :
    
    # adapted heavily from
    # https://github.com/CASTOR-telescope/ETC/blob/master/castor_etc/background.py
    
    # with additional discussion about renormalizing the zodiacal file from
    # https://github.com/CASTOR-telescope/ETC/tree/master/castor_etc/data/sky_background
    
    # from CASTOR ETC:
    # etc = np.array([27.72748, 24.24196, 22.58821])
    
    inDir = 'passbands/passbands_micron/'
    filters = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g']
    
    return determine_leo_background(inDir, filters)

def background_euclid() :
    
    inDir = 'passbands/passbands_micron/'
    filters = ['euclid_ie', 'euclid_ye', 'euclid_je', 'euclid_he']
    
    return determine_l2_background(inDir, filters)

def background_hst() :
    
    inDir = 'passbands/passbands_micron/'
    filters = ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w',  'hst_f390w',
               'hst_f438w', 'hst_f435w', 'hst_f475w', 'hst_f555w',  'hst_f606w',
               'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp', 'hst_f105w',
               'hst_f110w', 'hst_f125w', 'hst_f140w', 'hst_f160w']
    
    return determine_leo_background(inDir, filters)

def background_jwst() :
    
    inDir = 'passbands/passbands_micron/'
    filters = ['jwst_f070w',  'jwst_f090w',  'jwst_f115w',  'jwst_f150w',
               'jwst_f200w',  'jwst_f277w',  'jwst_f356w',  'jwst_f410m',
               'jwst_f444w',  'jwst_f560w',  'jwst_f770w',  'jwst_f1000w',
               'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w',
               'jwst_f2100w', 'jwst_f2550w']
    
    return determine_l2_background(inDir, filters)

def background_roman() :
    
    inDir = 'passbands/passbands_micron/'
    filters = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
               'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213']
    
    return determine_l2_background(inDir, filters)

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
            bkgs_at_wave = np.loadtxt(
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
        array = np.loadtxt(inDir + filt + '.txt')
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
        array = np.loadtxt(inDir + filt + '.txt')
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
    
    # the saved read noise and dark current values are scaled assuming the
    # images are saved with a final pixel scale of 0.05"/pixel, which is the
    # notional CASTOR pixel scale after dithering
    
    dictionary = filters.calculate_psfs()
    
    # CASTOR
    castor_filts = [key for key in dictionary.keys() if 'castor' in key]
    castor_scales = 0.1*np.ones(5)*u.arcsec/u.pix
    castor_bkg = background_castor()*u.mag/np.square(u.arcsec)
    castor_dark = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])*u.electron/u.s/u.pix
    castor_read = np.array([3.0, 3.0, 3.0, 3.0, 3.0])*u.electron/u.pix
    castor_scalings = np.square(castor_scales)/np.square(0.05*u.arcsec/u.pix)
    castor_dark = castor_dark/castor_scalings
    castor_read = castor_read/castor_scalings
    for filt, bkg, dark, read in zip(castor_filts, castor_bkg, castor_dark,
                                     castor_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # HST
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
    
    # JWST
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
    
    # Roman
    roman_filts = [key for key in dictionary.keys() if 'roman' in key]
    roman_scales = 0.11*np.ones(8)*u.arcsec/u.pix
    roman_bkg = background_roman()*u.mag/np.square(u.arcsec)
    roman_dark = 0.018*np.ones(8)*u.electron/u.s/u.pix
    roman_read = 13.81*np.ones(8)*u.electron/u.pix
    roman_scalings = np.square(roman_scales)/np.square(0.05*u.arcsec/u.pix)
    roman_dark = roman_dark/roman_scalings
    roman_read = roman_read/roman_scalings
    for filt, bkg, dark, read in zip(roman_filts, roman_bkg, roman_dark,
                                     roman_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # Euclid
    euclid_filts = [key for key in dictionary.keys() if 'euclid' in key]
    euclid_scales = np.array([0.1, 0.298, 0.298, 0.298])*u.arcsec/u.pix
    euclid_bkg = background_euclid()*u.mag/np.square(u.arcsec)
    euclid_dark = np.array([0.001, 0.02, 0.02, 0.02])*u.electron/u.s/u.pix
    euclid_read = np.array([4.4, 6.1, 6.1, 6.1])*u.electron/u.pix
    euclid_scalings = np.square(euclid_scales)/np.square(0.05*u.arcsec/u.pix)
    euclid_dark = euclid_dark/euclid_scalings
    euclid_read = euclid_read/euclid_scalings
    for filt, bkg, dark, read in zip(euclid_filts, euclid_bkg, euclid_dark,
                                     euclid_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    return dictionary

def flam_to_mag(waves, flam, response) :
    
    # from Eq. 2 of Bessell & Murphy 2012
    numer = np.trapezoid(flam*response*waves*np.square(u.arcsec), x=waves)
    denom = np.trapezoid(response/waves, x=waves)
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
            dictionary = determine_noise_components()
            pickle.dump(dictionary, file)
    
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

def get_largest_psf(telescopes, not_miri=True) :
    
    # get the dictionary of all the filter parameters
    telescope_params = get_noise(pretty_print=False)
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

'''
# import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)

from core import open_cutout

table = Table.read('tools/subIDs.fits')
subIDs = table['subID'].data
subIDs = subIDs[(subIDs < 14) | (subIDs > 14)]

uv_los, uv_meds, uv_his = np.full(272, -1.), np.full(272, -1.), np.full(272, -1.)
h_los, h_meds, h_his = np.full(272, -1.), np.full(272, -1.), np.full(272, -1.)
integrated_snr = np.full(272, -1.)
for i, subID in enumerate(subIDs) :
    uv_file = 'cutouts/quenched/{}/castor_uv_ultradeep.fits'.format(subID)
    uv_noise_file = 'cutouts/quenched/{}/castor_uv_ultradeep_noise.fits'.format(subID)
    hband_file = 'cutouts/quenched/{}/roman_f158_hlwas.fits'.format(subID)
    hband_noise_file = 'cutouts/quenched/{}/roman_f158_hlwas_noise.fits'.format(subID)
    
    uv, shape = open_cutout(uv_file, simple=True)
    uv_noise, _ = open_cutout(uv_noise_file, simple=True)
    hband, _ = open_cutout(hband_file, simple=True)
    hband_noise, _ = open_cutout(hband_noise_file, simple=True)
    
    # need to calculate the circle that encloses 5 Re, given that the FoV is 20 Re
    # on a side, and also account for even/odd number of pixels
    
    Re = shape[0]/20
    center = int(shape[0]/2) # or np.round() ? not sure how to best select center
    
    YY, XX = np.ogrid[:shape[0], :shape[0]]
    dist_from_center = np.sqrt(np.square(XX - center) + np.square(YY - center))
    mask = (dist_from_center <= 5*Re)
    small = (dist_from_center <= 2*Re)
    # plt.display_image_simple(mask, lognorm=False)
    
    uv_central, uv_noise_central = uv.copy(), uv_noise.copy()
    uv_central[~small], uv_noise_central[~small] = 0, 0
    
    integrated_snr[i] = np.sum(uv_central)/np.sqrt(np.sum(np.square(uv_noise_central)))
    
    uv[~mask], uv_noise[~mask] = 0, 0
    hband[~mask], hband_noise[~mask] = 0, 0
    
    # plt.display_image_simple(uv/uv_noise, lognorm=False, vmin=0.5, vmax=10)
    # plt.display_image_simple(hband/hband_noise, lognorm=False, vmin=0.5, vmax=10)
    
    # then use that circle as a mask to mask out pixels, and calculate the median
    # +/- 1 sigma values for the S/N of the UV image, along with the median
    # +/- 1 sigma values for the S/N of the Hband image, and plot one as a
    # function of the other
    
    uv_snr = (uv/uv_noise).flatten()
    uv_lo, uv_med, uv_hi = np.nanpercentile(uv_snr, [16, 50, 84])
    hband_snr = (hband/hband_noise).flatten()
    h_lo, h_med, h_hi = np.nanpercentile(hband_snr, [16, 50, 84])
    
    uv_los[i] = uv_lo
    uv_meds[i] = uv_med
    uv_his[i] = uv_hi
    h_los[i] = h_lo
    h_meds[i] = h_med
    h_his[i] = h_hi

# print(np.sort(integrated_snr))
# print(np.percentile(integrated_snr, [16, 50, 84]))

plt.plot_scatter_dumb(h_meds, uv_meds, integrated_snr, '', 'o',
    xlabel='median Roman H band S/N per pixel within 5 Re',
    ylabel='median CASTOR UV S/N per pixel within 5 Re',
    cbar_label='integrated CASTOR UV S/N within 2 Re',
    xmin=0.01, xmax=30, ymin=0.01, ymax=100, scale='log', loc=2, vmin=50, vmax=500)

# xs = np.linspace(0, 60, 101)
# plt.plot_simple_multi([xs, h_meds], [xs, uv_meds], ['equality', ''], ['r', 'k'],
#     ['', 'o'], ['-', ''], [1, 1], xlabel='median Roman H band S/N < 5 Re',
#     ylabel='median CASTOR UV S/N < 5 Re', ,
#     scale='log')

# xlo = h_meds - h_los
# xhi = h_his - h_meds
# ylo = uv_meds - uv_los
# yhi = uv_his - uv_meds
# plt.plot_scatter_err_both(h_meds, uv_meds, xlo, xhi, ylo, yhi,
#     xlabel='Roman H band S/N < 5 Re', ylabel='CASTOR UV S/N< 5 Re')
'''





# waves, total, zodi, ism, stray, thermal = np.loadtxt(
#     'background/background_JWST/minzodi_background.txt', unpack=True)
# xs = [waves, waves, waves, waves, waves]
# ys = [ism, stray, thermal, zodi, total]
# labels = ['ISM', 'Stray light', 'Thermal', 'Zodi', 'Total']
# colors = ['b', 'g', 'r', 'orange', 'k']
# markers = ['']*5
# styles = ['-']*5
# alphas = np.ones(5)
# plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
#     xmin=1, xmax=26, ymin=0.008, ymax=1000)

'''
from astropy.modeling import models
bb = models.BlackBody(temperature=2.7255*u.K) # from Fixsen 2009
# https://ui.adsabs.harvard.edu/abs/2009ApJ...707..916F/abstract


files = ['D:/Desktop/CASTOR_paper/IR_background_Spitzer+Herschel/background_minzodi_L2.tbl',
         'D:/Desktop/CASTOR_paper/IR_background_Spitzer+Herschel/background_minzodi_LEO.tbl']
for file in files[:1] :
    data = np.loadtxt(file, dtype=str, skiprows=23).astype(float)
    waves = data[:, 7]*u.um
    nus = (c.c/waves).to(u.Hz)
    zodi = data[:, 2]*u.MJy/u.sr
    ism = data[:, 3]*u.MJy/u.sr
    stars = data[:, 4]*u.MJy/u.sr
    cib = data[:, 5]*u.MJy/u.sr
    # totbg = data[:, 6]*u.MJy/u.sr
    
    # add in the contribution from the CMB
    cmb = bb(waves).to(u.MJy/u.sr)
    totbg = zodi+ism+stars+cib+cmb
    
    xs = [waves, waves, waves, waves, waves, waves]
    ys = [zodi, ism, stars, cib, cmb, totbg]
    labels = ['zodi', 'ism', 'stars', 'CIB', 'cmb', 'total']
    colors = ['orange', 'b', 'gold', 'grey', 'r', 'k']
    markers = ['']*6
    styles = ['-']*6
    alphas = [1]*6
    plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
        xlabel=r'$lambda$ ($mu$m)', ylabel=r'MJy sr$^{-1}$',
        ymin=0.001, ymax=1000, scale='log')
    
    ys = [(zodi*nus).to(u.W/u.m/u.m/u.sr), (ism*nus).to(u.W/u.m/u.m/u.sr),
          (stars*nus).to(u.W/u.m/u.m/u.sr), (cib*nus).to(u.W/u.m/u.m/u.sr),
          (cmb*nus).to(u.W/u.m/u.m/u.sr), (totbg*nus).to(u.W/u.m/u.m/u.sr)]
    # plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
    #     xlabel=r'$lambda$ ($mu$m)', ylabel=r'W m$^{-2}$ sr$^{-1}$',
    #     ymin=1e-09, ymax=1e-05, scale='log')
'''
