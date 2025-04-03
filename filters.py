
import numpy as np

from astropy.table import Table
import astropy.units as u

def calculate_pivots_and_fwhm() :
    
    filters = ['castor_uv',    'castor_uvL',   'castor_uS',    'castor_u',
               'castor_g',
               'euclid_ie',    'euclid_ye',    'euclid_je',    'euclid_he',
               # 'galex_fuv',    'galex_nuv',
               # 'herschel_70',  'herschel_100', 'herschel_160',
               # 'herschel_250', 'herschel_350', 'herschel_500',
               'hst_f218w',    'hst_f225w',    'hst_f275w',    'hst_f336w',
               'hst_f390w',    'hst_f438w',    'hst_f435w',    'hst_f475w',
               'hst_f555w',    'hst_f606w',    'hst_f625w',    'hst_f775w',
               'hst_f814w',    'hst_f850lp',   'hst_f105w',    'hst_f110w',
               'hst_f125w',    'hst_f140w',    'hst_f160w',
               # 'johnson_v',    'johnson_v025', 'johnson_v050',
               'jwst_f070w',   'jwst_f090w',   'jwst_f115w',   'jwst_f150w',
               'jwst_f200w',   'jwst_f277w',   'jwst_f356w',   'jwst_f410m',
               'jwst_f444w',   'jwst_f560w',   'jwst_f770w',   'jwst_f1000w',
               'jwst_f1130w',  'jwst_f1280w',  'jwst_f1500w',  'jwst_f1800w',
               'jwst_f2100w',  'jwst_f2550w',
               'roman_f062',   'roman_f087',   'roman_f106',   'roman_f129',
               'roman_f146',   'roman_f158',   'roman_f184',   'roman_f213',
               # 'spitzer_ch1',  'spitzer_ch2',  'spitzer_ch3',  'spitzer_ch4',
               # 'spitzer_24',   'spitzer_70',   'spitzer_160',
               # 'wise_w1',      'wise_w2',      'wise_w3',      'wise_w4']
               ]
    
    dictionary = {}
    for filt in filters :
        dictionary[filt] = {}
        
        # get the wavelength and transmission array from the file, and populate
        file = 'passbands/passbands_micron/{}.txt'.format(filt)
        array = np.loadtxt(file)
        waves, transmission = array[:, 0], array[:, 1]
        
        # calculate the pivot wavelength, following Eq. A11 of Tokunagea & Vacca 2005
        pivot = np.sqrt(np.trapezoid(transmission*waves, x=waves)/
                        np.trapezoid(transmission/waves, x=waves))
        dictionary[filt]['pivot'] = pivot*u.um
        
        # prepare the transmission array for calculating the FWHM
        trans = transmission/np.max(transmission)
        good = np.where(trans - 0.5 >= 0.0)[0]
        start, end = good[0], good[-1]
        
        # ensure that "start" and "end" are the closest indices
        start = start - 10 + np.abs(trans[start-10:start+10] - 0.5).argmin()
        end = end - 10 + np.abs(trans[end-10:end+10] - 0.5).argmin()
        
        # calculate the FWHM
        fwhm = waves[end] - waves[start]
        dictionary[filt]['fwhm'] = fwhm*u.um
    
    return dictionary

def calculate_psfs() :
    
    dictionary = calculate_pivots_and_fwhm()
    
    for filt in dictionary.keys() :
        # CASTOR PSF FWHMs
        if 'castor' in filt :
            dictionary[filt]['psf'] = 0.15*u.arcsec
        
        # WFC3/UVIS PSF FWHMs
        elif filt in ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w',
                      'hst_f390w', 'hst_f438w'] :
            xp = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000,
                           1100])*u.nm
            fp = np.array([2.069, 1.870, 1.738, 1.675, 1.681, 1.746, 1.844,
                           1.960, 2.091, 2.236])*u.pix
            psf = np.interp(dictionary[filt]['pivot'], xp.to(u.um),
                            fp*0.0395*u.arcsec/u.pix)
            dictionary[filt]['psf'] = psf
        
        # ACS PSF FWHMs -> use UVIS PSFs as no definitive ACS PSFs have been found
        elif filt in ['hst_f435w', 'hst_f475w', 'hst_f555w', 'hst_f606w',
                      'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp'] :
            xp = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000,
                           1100])*u.nm
            fp = np.array([2.069, 1.870, 1.738, 1.675, 1.681, 1.746, 1.844,
                           1.960, 2.091, 2.236])*u.pix
            psf = np.interp(dictionary[filt]['pivot'], xp.to(u.um),
                            fp*0.05*u.arcsec/u.pix)
            dictionary[filt]['psf'] = psf
        
        # WFC3/IR PSF FWHMs
        elif filt in ['hst_f105w', 'hst_f110w', 'hst_f125w', 'hst_f140w',
                      'hst_f160w'] :
            xp = np.array([800, 900, 100, 1100, 1200, 1300, 1400, 1500, 1600,
                           1700])*u.nm
            fp = np.array([0.971, 0.986, 1.001, 1.019, 1.040, 1.067, 1.100,
                           1.136, 1.176, 1.219])*u.pix
            psf = np.interp(dictionary[filt]['pivot'], xp.to(u.um),
                            fp*0.128*u.arcsec/u.pix)
            dictionary[filt]['psf'] = psf
    
    # JWST PSF FWHMs
    jwst_filts = [key for key in dictionary.keys() if 'jwst' in key]
    jwst_scales = np.array([0.031, 0.031, 0.031, 0.031, 0.031, 0.063, 0.063,
                            0.063, 0.063, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
                            0.11, 0.11, 0.11])*u.arcsec/u.pix
    jwst_psfs = np.array([0.935, 1.065, 1.290, 1.613, 2.129, 1.460, 1.841,
                          2.175, 2.302, 1.882, 2.445, 2.982, 3.409, 3.818,
                          4.436, 5.373, 6.127, 7.300])*u.pix
    for filt, psf in zip(jwst_filts, jwst_scales*jwst_psfs) :
        dictionary[filt]['psf'] = psf
    
    # Roman PSF FWHMs
    roman_filts = [key for key in dictionary.keys() if 'roman' in key]
    roman_psfs = np.array([0.058, 0.073, 0.087, 0.106, 0.105, 0.128, 0.146,
                           0.169])*u.arcsec
    for filt, psf in zip(roman_filts, roman_psfs) :
        dictionary[filt]['psf'] = psf
    
    # Euclid PSF FWHMs
    euclid_filts = [key for key in dictionary.keys() if 'euclid' in key]
    euclid_scales = np.array([0.1, 0.298, 0.298, 0.298])*u.arcsec/u.pix
    euclid_psfs = np.array([1.3, 1.10, 1.17, 1.19])*u.pix # from Euclid Collab+Mellier+2024
    for filt, psf in zip(euclid_filts, euclid_scales*euclid_psfs) :
        dictionary[filt]['psf'] = psf
    
    return dictionary

def prepare_throughputs_for_fastpp() :
    
    filters = ['castor_uv',    'castor_uvL',   'castor_uS',    'castor_u',
               'castor_g',
               'euclid_ie',    'euclid_ye',    'euclid_je',    'euclid_he',
               'galex_fuv',    'galex_nuv',
               'herschel_70',  'herschel_100', 'herschel_160',
               'herschel_250', 'herschel_350', 'herschel_500',
               'hst_f218w',    'hst_f225w',    'hst_f275w',    'hst_f336w',
               'hst_f390w',    'hst_f438w',    'hst_f435w',    'hst_f475w',
               'hst_f555w',    'hst_f606w',    'hst_f625w',    'hst_f775w',
               'hst_f814w',    'hst_f850lp',   'hst_f105w',    'hst_f110w',
               'hst_f125w',    'hst_f140w',    'hst_f160w',
               'johnson_v',    'johnson_v025', 'johnson_v050',
               'jwst_f070w',   'jwst_f090w',   'jwst_f115w',   'jwst_f150w',
               'jwst_f200w',   'jwst_f277w',   'jwst_f356w',   'jwst_f410m',
               'jwst_f444w',   'jwst_f560w',   'jwst_f770w',   'jwst_f1000w',
               'jwst_f1130w',  'jwst_f1280w',  'jwst_f1500w',  'jwst_f1800w',
               'jwst_f2100w',  'jwst_f2550w',
               'roman_f062',   'roman_f087',   'roman_f106',   'roman_f129',
               'roman_f146',   'roman_f158',   'roman_f184',   'roman_f213',
               'spitzer_ch1',  'spitzer_ch2',  'spitzer_ch3',  'spitzer_ch4',
               'spitzer_24',   'spitzer_70',   'spitzer_160',
               'wise_w1',      'wise_w2',      'wise_w3',      'wise_w4']
    
    for filt in filters :
        array = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt))
        final = np.array([np.arange(1, len(array) + 1), array[:, 0]*1e4, array[:, 1]]).T
        fmt = ['%-4i', '%12.5e', '%12.5e']
        header = '   {} {}'.format(len(array), filt)
        np.savetxt('passbands/passbands_fastpp_angstrom/{}.txt'.format(filt),
                   final, fmt=fmt, header=header, comments='')
    
    return

def throughputs_castor() :
    
    # https://github.com/CASTOR-telescope/ETC/tree/master/castor_etc/data/passbands
    
    # https://github.com/CASTOR-telescope/ETC_notebooks/tree/master/data
    
    inDir = 'passbands/passbands_CASTOR_Phase-0-ETC-github/'
    files = ['passband_castor.uv', 'passband_castor.u', 'passband_castor.g']
    
    for file in files :
        filt = file.split('.')[1]
        
        final = np.loadtxt(inDir + file)
        
        np.savetxt('passbands/passbands_micron/castor_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    inDir = 'passbands/passbands_CASTOR_Phase-0-ETC-notebooks-github/'
    files = ['passband_castor.uv_split_bb', 'passband_castor.u_split_bb']
    filts = ['uvL', 'uS']
    
    for file, filt in zip(files, filts) :
        final = np.loadtxt(inDir + file)
        np.savetxt('passbands/passbands_micron/castor_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_euclid() :
    
    # http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?id=Euclid/VIS.vis&&
    # mode=browse&gname=Euclid&gname2=VIS#filter
    
    # https://euclid.esac.esa.int/msp/refdata/nisp/NISP-PHOTO-PASSBANDS-V1
    
    inDir = 'passbands/passbands_Euclid_for-2021-11-01-from-ESA/'
    
    vis = np.loadtxt(inDir + 'Euclid_VIS.vis.dat')
    vis = np.array([vis[:, 0]/1e4, vis[:, 1]]).T
    
    np.savetxt('passbands/passbands_micron/euclid_ie.txt', vis,
                header='WAVELENGTH THROUGHPUT')
    
    files = ['NISP-PHOTO-PASSBANDS-V1-Y_throughput.dat',
             'NISP-PHOTO-PASSBANDS-V1-J_throughput.dat',
             'NISP-PHOTO-PASSBANDS-V1-H_throughput.dat']
    filts = ['ye', 'je', 'he']
    
    for file, filt in zip(files, filts) :
        data = np.loadtxt(inDir + file)
        waves = data[:, 0]/1000
        trans = data[:, 1]
        trans[trans < 0] = 0
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/euclid_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_galex() :
    
    # https://asd.gsfc.nasa.gov/archive/galex/instrument.html
    
    # https://asd.gsfc.nasa.gov/archive/galex/Documents/PostLaunchResponseCurveData.html
    
    inDir = 'passbands/passbands_GALEX_for-PostLaunch2006-from-GSFC/'
    filts = ['FUV', 'NUV']
    area = np.pi*np.square(25) # GALEX was 50 cm in diameter
    
    for filt in filts :
        
        waves, eff_area = np.loadtxt(inDir + filt + '.txt', skiprows=1, unpack=True)
        final = np.array([waves/1e4, eff_area/area]).T
        
        np.savetxt('passbands/passbands_micron/galex_{}.txt'.format(filt.lower()),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_herschel() :
    
    # https://nhscsci.ipac.caltech.edu/sc/index.php/Pacs/FilterCurves
    
    inDir = 'passbands/passbands_Herschel_PACS_for-29Mar2010-from-IPAC/'
    
    filts = ['blue', 'green', 'red']
    names = ['70', '100', '160']
    for filt, name in zip(filts, names) :
        waves, trans = np.loadtxt(inDir + 'Herschel_Pacs.{}.dat'.format(filt),
                                  unpack=True)
        final = np.array([waves/1e4, trans]).T
        
        np.savetxt('passbands/passbands_micron/herschel_{}.txt'.format(name),
                   final, header='WAVELENGTH THROUGHPUT')
    
    # https://irsa.ipac.caltech.edu/data/Herschel/docs/nhsc/spire/PhotInstrumentDescription.html
    
    # https://irsa.ipac.caltech.edu/data/Herschel/docs/nhsc/spire/SpirePhotRSRF.fits
    
    # https://archives.esac.esa.int/hsa/legacy/ADP/SPIRE/SPIRE-P_filter_curves/
    
    inDir = 'passbands/passbands_Herschel_SPIRE_for-14Jun2016-from-HSA/'
    
    table = Table.read(inDir + 'SpirePhotRSRF.fits')
    filts = ['psw', 'pmw', 'plw']
    names = ['250', '350', '500']
    
    for filt, name in zip(filts, names) :
        waves, trans = 1/table['wavenumber'].data*1e4, table[filt].data
        trans[trans < 0] = 0
        
        final = np.array([np.flip(waves), np.flip(trans)]).T
        
        np.savetxt('passbands/passbands_micron/herschel_{}.txt'.format(name),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_hst() :
    
    # https://stsynphot.readthedocs.io/en/latest/stsynphot/appendixb_inflight.html
    
    # https://stsynphot.readthedocs.io/en/latest/stsynphot/obsmode.html
    
    # get the throughputs for the ACS WFC filters
    from stsynphot.config import conf
    conf.rootdir = 'D:/Documents/GitHub/CASTOR/noise/trds'
    
    from synphot.config import conf as syn_conf
    syn_conf.vega_file = 'D:/Documents/GitHub/CASTOR/noise/trds/calspec/alpha_lyr_stis_011.fits'
    
    from stsynphot.spectrum import band
    
    # https://core2.gsfc.nasa.gov/time/julian.html, MJD = JD - 2400001
    mjd = 55008 # June 26, 2009, ie. approximately when WFC3 was installed
    
    filts = ['f435w', 'f475w', 'f555w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
    for filt in filts :
        bp1 = band('acs,wfc1,{},mjd#{}'.format(filt, mjd))
        bp2 = band('acs,wfc2,{},mjd#{}'.format(filt, mjd))
        
        waves = bp1.binset.to(u.um) # exactly equal to bp2.binset as well
        trans1, trans2 = bp1(waves), bp2(waves)
        
        final = np.array([waves.data, np.mean([trans1, trans2], axis=0)]).T
        
        np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    # https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs
    
    # get the throughputs for the WFC3 UVIS filters
    inDir = 'passbands/passbands_HST_WFC3-UVIS_for-26Jun2009-from-STScI/'
    filts = ['f218w', 'f225w', 'f275w', 'f336w', 'f390w', 'f438w']
    for filt in filts :
        waves, trans1 = np.loadtxt(inDir + 'wfc3_uvis1_{}.txt'.format(filt),
                                   unpack=True)
        _, trans2 = np.loadtxt(inDir + 'wfc3_uvis2_{}.txt'.format(filt),
                               unpack=True)
        
        final = np.array([waves/1e4, np.mean([trans1, trans2], axis=0)]).T
        
        np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    # get the throughputs for the WFC3 IR filters
    inDir = 'passbands/passbands_HST_WFC3-IR_for-26Jun2009-from-STScI/'
    filts = ['f105w', 'f110w', 'f125w', 'f140w', 'f160w']
    for filt in filts :
        waves, trans = np.loadtxt(inDir + 'wfc3_ir_{}.txt'.format(filt),
                                  unpack=True)
        
        final = np.array([waves/1e4, trans]).T
        
        np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_johnson() :
    
    # https://ui.adsabs.harvard.edu/abs/2006AJ....131.1184M/abstract
    
    # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/131/1184
    
    inDir = 'passbands/passbands_Johnson_for-Feb2006-from-ADS/'
    waves, trans = np.loadtxt(inDir + 'johnson_v.dat', unpack=True)
    
    filts = ['johnson_v', 'johnson_v025', 'johnson_v050']
    coeffs = [1, 1.25, 1.5]
    
    for filt, coeff in zip(filts, coeffs) :
        final = np.array([coeff*waves/1e4, trans]).T
        np.savetxt('passbands/passbands_micron/{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_jwst() :
    
    # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
    
    inDir = ('passbands/passbands_JWST_NIRCam_for-Jan2025-from-JDocs/' +
            'nircam_throughputs_Jan2025_v7/nircam_throughputs/mean_throughputs/')
    
    filts = ['F070W', 'F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W',
             'F410M', 'F444W']
    for filt in filts :
        waves, trans = np.loadtxt(
            inDir + '{}_May2024_mean_system_throughput.txt'.format(filt),
            unpack=True, skiprows=1)
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/jwst_{}.txt'.format(filt.lower()),
                   final, header='WAVELENGTH THROUGHPUT')
    
    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-technical-library
    
    inDir = 'passbands/passbands_JWST_MIRI_for-11Sep2024-from-STScIBox/MIRI_endtoend_throughputs_v4.0/'
    
    filts = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W',
             'F2100W', 'F2550W']
    for filt in filts :
        table = Table.read(inDir +
            'MIRI_{}_endtoend_throughput_perphoton_ETC4.0.fits'.format(filt))
        waves, trans = table['WAVELENGTH'].data, table['THROUGHPUT'].data
        
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/jwst_{}.txt'.format(filt.lower()),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_roman() :
    
    # https://roman.gsfc.nasa.gov/science/WFI_technical.html
    
    # https://vmromanweb1.ipac.caltech.edu/page/param-db
    
    inDir = 'passbands/passbands_Roman_for-27Mar2024-from-GSFC/'
    all_SCAs = np.full((18, 2200, 12), np.nan)
    for i in range(18):
        infile = inDir + 'Roman_effarea_v8_SCA{:02}_20240301.ecsv'.format(i + 1)
        all_SCAs[i] = np.loadtxt(infile, delimiter=',', skiprows=18)
    eff_areas = np.mean(all_SCAs, axis=0)
    area = np.pi*np.square(1.18) # Roman will be 2.36 m in diameter
    
    filters = ['f062', 'f087', 'f106', 'f129', 'f158', 'f184', 'f146', 'f213']
    for i, filt in enumerate(filters) :
        final = np.array([eff_areas[:, 0], eff_areas[:, i+1]/area]).T
        
        np.savetxt('passbands/passbands_micron/roman_{}_new.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_spitzer() :
    
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/
    
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/
    
    inDir = 'passbands/passbands_Spitzer_IRAC_for-Sep2021-from-IRSA/'
    
    filts = ['ch1', 'ch2', 'ch3', 'ch4']
    for filt in filts :
        waves, trans = np.loadtxt(inDir + 'irac_201125{}trans_full.txt'.format(filt),
                                  skiprows=3, unpack=True)
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/spitzer_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/6/
    
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/calibrationfiles/spectralresponse/
    
    inDir = 'passbands/passbands_Spitzer_MIPS_for-Mar2011-from-IPAC/'
    
    filts = ['24', '70', '160']
    cols = [[0, 1], [2, 3], [4, 5]]
    maxis = [128, 111, 400]
    
    for filt, col, maxi in zip(filts, cols, maxis) :
        waves, trans = np.genfromtxt(inDir + 'MIPSfiltsumm.csv', delimiter=',',
                                     usecols=col, unpack=True, max_rows=maxi)
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/spitzer_{}.txt'.format(filt),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_wise() :
    
    # https://www.astro.ucla.edu/~wright/WISE/passbands.html
    
    # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#WISEZMA
    
    inDir = 'passbands/passbands_WISE_for-14Apr2011-from-IPAC/'
    
    filts = ['W1', 'W2', 'W3', 'W4']
    for filt in filts :
        waves, trans, _ = np.loadtxt(inDir + 'RSR-{}.txt'.format(filt),
                                     skiprows=2, unpack=True)
        final = np.array([waves, trans]).T
        
        np.savetxt('passbands/passbands_micron/wise_{}.txt'.format(filt.lower()),
                   final, header='WAVELENGTH THROUGHPUT')
    
    return
