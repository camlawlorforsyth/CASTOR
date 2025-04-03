
import numpy as np

from astropy.table import Table
import astropy.units as u
# from pandeia.engine.instrument_factory import InstrumentFactory

import plotting as plt

def compare_nircam() :

    olddir = 'passbands/passbands_JWST_NIRCam_for-4Nov2022-from-JDocs/'
    newdir = 'passbands/passbands_micron/'
    oldfilts = ['F070W_mean_system_throughput.txt',
                'F090W_mean_system_throughput.txt',
                'F115W_mean_system_throughput.txt',
                'F150W_mean_system_throughput.txt',
                'F200W_mean_system_throughput.txt',
                'F277W_mean_system_throughput.txt',
                'F356W_mean_system_throughput.txt',
                'F410M_mean_system_throughput.txt',
                'F444W_mean_system_throughput.txt']
    newfilts = ['jwst_f070w.txt', 'jwst_f090w.txt', 'jwst_f115w.txt',
                'jwst_f150w.txt', 'jwst_f200w.txt', 'jwst_f277w.txt',
                'jwst_f356w.txt', 'jwst_f410m.txt', 'jwst_f444w.txt']
    
    for oldfilt, newfilt in zip(oldfilts, newfilts) :
        old = np.loadtxt(olddir + oldfilt)
        new = np.loadtxt(newdir + newfilt)
    
        plt.plot_simple_multi([old[:, 0], new[:, 0]], [old[:, 1], new[:, 1]],
            ['old', 'new'], ['k', 'r'], ['', ''], ['-', '-'], [1, 1],
            scale='linear')
    
    return

def check_roman() :
    
    filters = ['f062', 'f087', 'f106', 'f129', 'f146', 'f158', 'f184', 'f213']
    
    waves = []
    throughputs = []
    for filt in filters :
        file = np.loadtxt('passbands/passbands_micron/roman_old/roman_{}.txt'.format(filt))
        waves.append(file[:, 0])
        throughputs.append(file[:, 1])
    
    plt.plot_simple_multi(waves, throughputs, filters,
        ['b', 'g', 'y', 'orange', 'grey', 'orangered', 'm', 'r'],
        ['', '', '', '', '', '', '', '', ''],
        ['-', '-', '-', '-', '--', '-', '-', '-', '-'], np.ones(9),
        xmin=0.4, xmax=2.45, ymin=0, ymax=0.7, scale='linear', loc=7)
    
    return

def check_roman_new() :
    
    filters = ['f062', 'f087', 'f106', 'f129', 'f146', 'f158', 'f184', 'f213']
    
    waves = []
    throughputs = []
    for filt in filters :
        file = np.loadtxt('passbands/passbands_micron/roman_{}.txt'.format(filt))
        waves.append(file[:, 0])
        throughputs.append(file[:, 1])
    
    plt.plot_simple_multi(waves, throughputs, filters,
        ['b', 'g', 'y', 'orange', 'grey', 'orangered', 'm', 'r'],
        ['', '', '', '', '', '', '', '', ''],
        ['-', '-', '-', '-', '--', '-', '-', '-', '-'], np.ones(9),
        xmin=0.4, xmax=2.45, ymin=0, ymax=0.7, scale='linear', loc=7)
    
    return

def prepare_throughputs_for_skirt() :
    
    filters = ['castor_uv',   'castor_uvL',  'castor_uS',   'castor_u',
               'castor_g',
               'euclid_ie',   'euclid_ye',   'euclid_je',   'euclid_he',
               'hst_f218w',   'hst_f225w',   'hst_f275w',   'hst_f336w',
               'hst_f390w',   'hst_f438w',   'hst_f435w',   'hst_f475w',
               'hst_f555w',   'hst_f606w',   'hst_f625w',   'hst_f775w',
               'hst_f814w',   'hst_f850lp',  'hst_f105w',   'hst_f110w',
               'hst_f125w',   'hst_f140w',   'hst_f160w',
               'jwst_f070w',  'jwst_f090w',  'jwst_f115w',  'jwst_f150w',
               'jwst_f200w',  'jwst_f277w',  'jwst_f356w',  'jwst_f410m',
               'jwst_f444w',  'jwst_f560w',  'jwst_f770w',  'jwst_f1000w',
               'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w',
               'jwst_f2100w', 'jwst_f2550w',
               'roman_f062',  'roman_f087',  'roman_f106',  'roman_f129',
               'roman_f146',  'roman_f158',  'roman_f184',  'roman_f213']
    
    for filt in filters :
        array = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt))
        final = np.array([array[:, 0]*1e4, array[:, 1]]).T
        np.savetxt('passbands/passbands_SKIRT_angstrom/{}.txt'.format(filt), final)
    
    return

def throughputs_hst_old() :
    
    from stsynphot.config import conf
    conf.rootdir = 'D:/Documents/GitHub/CASTOR/noise/trds'
    
    from synphot.config import conf as syn_conf
    syn_conf.vega_file = 'D:/Documents/GitHub/CASTOR/noise/trds/calspec/alpha_lyr_stis_011.fits'
    
    from stsynphot.spectrum import band
    
    # HFF_programs = [11108, 11507, 11582, 11591, 13459, 13790, 14038, 14216, # a370
    #                 12458, 13459, 14037, 14209, # a1063
    #                 11689, 13386, 13389, 13495, 14209, # a2744
    #                 12459, 13386, 13496, 14209, # m416
    #                 9722, 10420, 10493, 10793, 12103, 13389, 13459, 13498, 14209, # m717
    #                12068, 13504, 13790, 14041 # m1149
    #                ]
    # unique = [ 9722, 10420, 10493, 10793, 11108, 11507, 11582, 11591, 11689, 12068,
    #           12103, 12458, 12459, 13386, 13389, 13459, 13495, 13496, 13498, 13504,
    #           13790, 14037, 14038, 14041, 14209, 14216]
    # obs = Table.read('noise/HST_HFF_observations.fits')
    # from astropy.time import Time
    # dates = Time(obs['start_TimeISO'], format='iso').mjd
    # median = np.percentile(dates, 50) # Aug 28, 2014
    
    # https://core2.gsfc.nasa.gov/time/julian.html, MJD = JD - 2400001
    median = 55008 # June 26, 2009, ie. approximately when WFC3 was installed
    
    obsmodes = []
    
    wfc3_uv_detectors = ['uvis1', 'uvis2']
    wfc3_uv_filters = ['f218w', 'f225w', 'f275w', 'f336w', 'f390w', 'f438w']
    for filt in wfc3_uv_filters :
        for detector in wfc3_uv_detectors :    
            obsmode = 'wfc3,' + detector + ',' + filt
            obsmodes.append(obsmode)
    
    acs_detectors = ['wfc1', 'wfc2']
    acs_filters = ['f435w', 'f475w', 'f555w', 'f606w', 'f625w', 'f775w',
                   'f814w', 'f850lp']
    for filt in acs_filters :
        for detector in acs_detectors :    
            obsmode = 'acs,' + detector + ',' + filt
            obsmodes.append(obsmode)
    
    wfc3_ir_filters = ['f105w', 'f110w', 'f125w', 'f140w', 'f160w']
    for filt in wfc3_ir_filters :
        obsmode = 'wfc3,ir,' + filt
        obsmodes.append(obsmode)
    
    # get the throughputs for the WFC3 UVIS filters
    for i in range(0, 12, 2) :
        
        filt = obsmodes[i].split(',')[2]
        
        bp1 = band(obsmodes[i] + ',mjd#{}'.format(median))
        bp2 = band(obsmodes[i+1] + ',mjd#{}'.format(median))
        waves1, waves2 = bp1.binset, bp2.binset
        throughput1, throughput2 = bp1(waves1), bp2(waves2)
        
        throughput = np.mean([throughput1, throughput2], axis=0)
        
        final = np.array([waves1.to(u.um).value, throughput]).T
        # np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt), final,
        #            header='WAVELENGTH THROUGHPUT')
        
        # import plotting as plt
        # plt.plot_simple_multi([waves1, waves1, waves1],
        #     [throughput1, throughput2, throughput], ['1', '2', 'avg'],
        #     ['k', 'b', 'r'], ['', '', ''], ['-', '-', '-'], [1, 1, 1],
        #     scale='linear', xmin=1950, xmax=4800)
    
    # get the throughputs for the ACS WFC filters
    for i in range(12, 28, 2) :
        
        filt = obsmodes[i].split(',')[2]
        
        bp1 = band(obsmodes[i] + ',mjd#{}'.format(median))
        bp2 = band(obsmodes[i+1] + ',mjd#{}'.format(median))
        waves1, waves2 = bp1.binset, bp2.binset
        throughput1, throughput2 = bp1(waves1), bp2(waves2)
        
        throughput = np.mean([throughput1, throughput2], axis=0)
        
        final = np.array([waves1.to(u.um).value, throughput]).T
        # np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt), final,
        #            header='WAVELENGTH THROUGHPUT')
        
        # plt.plot_simple_multi([waves1, waves1, waves1],
        #     [throughput1, throughput2, throughput], ['1', '2', 'avg'],
        #     ['k', 'b', 'r'], ['', '', ''], ['-', '-', '-'], [1, 1, 1],
        #     scale='linear', xmin=3500, xmax=11000)
    
    # get the throughputs for the WFC3 IR filters
    for i in range(28, 33) :
        
        filt = obsmodes[i].split(',')[2]
        
        bp = band(obsmodes[i] + ',mjd#{}'.format(median))
        waves = bp.binset
        throughput = bp(waves)
        
        final = np.array([waves.to(u.um).value, throughput]).T
        # np.savetxt('passbands/passbands_micron/hst_{}.txt'.format(filt), final,
        #            header='WAVELENGTH THROUGHPUT')
        
        # plt.plot_simple_multi([waves], [throughput], [''], ['k'], [''], ['-'],
        #     [1], scale='linear', xmin=8500, xmax=17500)
    
    return

def throughputs_jwst_old() :
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-pandeia-engine-tutorial/jwst-etc-instrument-throughputs
    
    nircamDir = 'noise/pandeia_data-2.0/jwst/nircam/filters/'
    miriDir = 'noise/pandeia_data-2.0/jwst/miri/filters/'
    
    sw_filters = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w']
    sw_conf = {#'detector':{'nexp':1, 'ngroup':10, 'nint':1
               #            'readout_pattern':'medium8', 'subarray':'full'},
               'instrument':{'aperture':'sw', 'disperser':'null',
                             'filter':'', 'instrument':'nircam',
                             'mode':'sw_imaging'}}
    
    lw_filters = ['f277w', 'f356w', 'f410m', 'f444w']
    lw_conf = {#'detector':{'nexp':1, 'ngroup':10, 'nint':1
               #            'readout_pattern':'medium8', 'subarray':'full'},
               'instrument':{'aperture':'lw', 'disperser':'null',
                             'filter':'', 'instrument':'nircam',
                             'mode':'lw_imaging'}}
    
    miri_filters = ['f0560w', 'f0770w', 'f1000w', 'f1130w', 'f1280w', 'f1500w',
                    'f1800w', 'f2100w', 'f2550w']
    miri_conf = {#'detector': {'nexp':1, 'ngroup':10, 'nint':1,
                 #             'readout_pattern':'fastr1', 'subarray':'full'},
                 'instrument': {'aperture':'imager', 'filter':'',
                                'instrument':'miri', 'mode':'imaging'}}
    
    for filt in sw_filters :
        file = nircamDir + 'jwst_nircam_{}_trans_20221103152537.fits'.format(filt)
        tab = Table.read(file)
        waves = tab['WAVELENGTH'].value*u.um
        
        # create a configured instrument
        conf = sw_conf
        conf['instrument']['filter'] = filt
        instrument_factory = InstrumentFactory(config=conf)
        
        # get the throughput of the instrument over the desired wavelengths
        throughput = instrument_factory.get_total_eff(waves.value)
        
        final = np.array([waves.to(u.um).value, throughput]).T
        np.savetxt('passbands/passbands_micron/jwst_{}.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    for filt in lw_filters :
        file = nircamDir + 'jwst_nircam_{}_trans_20221103152537.fits'.format(filt)
        tab = Table.read(file)
        waves = tab['WAVELENGTH'].value*u.um
        
        # create a configured instrument
        conf = lw_conf
        conf['instrument']['filter'] = filt
        instrument_factory = InstrumentFactory(config=conf)
        
        # get the throughput of the instrument over the desired wavelengths
        throughput = instrument_factory.get_total_eff(waves.value)
        
        final = np.array([waves.to(u.um).value, throughput]).T
        np.savetxt('passbands/passbands_micron/jwst_{}.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    for filt in miri_filters :
        
        file = miriDir + 'jwst_miri_{}_trans_20221013160429.fits'.format(filt)
        tab = Table.read(file)
        waves = tab['WAVELENGTH'].value*u.um
        
        if filt == 'f0560w' :
            filt = 'f560w'
        if filt == 'f0770w' :
            filt = 'f770w'
        
        # create a configured instrument
        conf = miri_conf
        conf['instrument']['filter'] = filt
        instrument_factory = InstrumentFactory(config=conf)
        
        # get the throughput of the instrument over the desired wavelengths
        throughput = instrument_factory.get_total_eff(waves.value)
        
        final = np.array([waves.to(u.um).value, throughput]).T
        np.savetxt('passbands/passbands_micron/jwst_{}.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    return

def throughputs_roman_old() :
    
    # https://roman.gsfc.nasa.gov/science/WFI_technical.html
    
    # https://roman.gsfc.nasa.gov/science/Roman_Reference_Information.html
    
    inDir = 'passbands/passbands_Roman_for-14Jun2021-from-GSFC/'
    eff_areas = np.loadtxt(inDir + 'Roman_effarea_20210614.csv', delimiter=',')
    area = np.pi*np.square(1.18) # Roman will be 2.36 m in diameter
    
    filters = ['f062', 'f087', 'f106', 'f129', 'f146', 'f158', 'f184', 'f213']
    for i, filt in enumerate(filters) :
        final = np.array([eff_areas[:, 0], eff_areas[:, i+1]/area]).T
        
        # plt.plot_simple_dumb(eff_areas[:, 0], eff_areas[:, i+1]/area, xmin=0.4, xmax=2.5)
        # plt.plot_simple_dumb(eff_areas[:, 0],
        #                      eff_areas[:, i+1]/(area*(1-np.square(0.303))),
        #                      xmin=0.4, xmax=2.5)
        
        np.savetxt('passbands/passbands_micron/roman_{}.txt'.format(filt), final,
                   header='WAVELENGTH THROUGHPUT')
    
    return
