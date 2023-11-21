
import os
import glob
import numpy as np

from astropy.table import Table, vstack
import astropy.units as u

from core import open_cutout

def all_fluxes(survey='all', population='quenched') :
    
    outDir = 'photometry/{}/'.format(population)
    
    os.makedirs(outDir, exist_ok=True) # ensure the output directory for the
        # photometric tables is available
    
    # table = Table.read('tools/subIDs.fits')
    # subIDs = table['subIDs'].data
    subIDs = [63871, 96771, 198186] # for testing
    
    filters = get_filters(survey)
    
    for subID in subIDs :
        determine_fluxes(subID, filters, population=population)
    
    return

def determine_fluxes(subID, filters, population='quenched') :
    
    binsDir = 'bins/{}/'.format(population)
    cutoutDir = 'cutouts/{}/{}/'.format(population, subID)
    outDir = 'photometry/{}/'.format(population)
    seppDir = 'detection/{}/{}/'.format(population, subID)
    
    infile = binsDir + 'subID_{}_annuli.npz'.format(subID)
    outfile = outDir + 'subID_{}_photometry.fits'.format(subID)
    
    # load the elliptical annuli bins data
    bin_data = np.load(infile)
    
    bins_image = bin_data['image']
    sma, smb = bin_data['sma'], bin_data['smb']
    flux, err = bin_data['flux'], bin_data['err']
    nPixels = bin_data['nPixels']
    widths, PAs = bin_data['width'], bin_data['pa']
    
    numBins = int(np.nanmax(bins_image) + 1) # accounts for python 0-index
    
    if not np.isnan(numBins) :
        photometry = Table()
        photometry['bin'] = range(numBins)
        photometry['sma'], photometry['smb'] = sma, smb
        photometry['flux'], photometry['err'] = flux, err
        photometry['SN'], photometry['nPixels'] = flux/err, nPixels
        photometry['width'], photometry['PA'] = widths, PAs
        
        # get the SourceXtractorPlusPlus-derived segmentation map
        segmap_file = seppDir + 'segmap.fits'
        segMap, _ = open_cutout(segmap_file, simple=True)
        
        # open the SourceXtractorPlusPlus-derived catalog file
        catalog = Table.read(seppDir + 'cat.fits')
        detID = catalog['group_id'].data[0]
        r_e = catalog['flux_radius'].data[0] # [pix] the half flux radius
        
        for filt in filters :
            
            # open the science images and the corresponding noise images
            sci_file, noise_file = glob.glob(cutoutDir + filt + '_*.fits')
            
            sci, _, redshift, exptime, area, photfnu, scale = open_cutout(sci_file)
            noise, _ = open_cutout(noise_file, simple=True)
            
            # make a copy of the science image and noise image
            new_sci = sci.copy()
            new_noise = noise.copy()
            
            # mask the copied images based on the segmentation map, but don't
            # mask out the sky -> mask out pixels associated with other galaxies
            new_sci[(segMap > 0) & (segMap != detID)] = 0
            new_noise[(segMap > 0) & (segMap != detID)] = 0
            
            # save extraneous information into the table
            length = len(range(numBins))
            photometry['R_e'] = [r_e]*length*u.pix
            photometry['z'] = [redshift]*length
            photometry['scale'] = [scale]*length
            
            fluxes, uncerts, invalid = [], [], []
            for val in range(numBins) :
                temp_sci, temp_noise = new_sci.copy(), new_noise.copy()
                
                temp_sci[bins_image != val] = np.nan
                flux_sum = np.nansum(temp_sci)
                flux = flux_sum/exptime/area*photfnu
                fluxes.append(flux)
                
                temp_noise[bins_image != val] = np.nan
                noise_sum = np.sqrt(np.nansum(np.square(temp_noise)))
                uncert = noise_sum/exptime/area*photfnu
                uncerts.append(uncert)
                
                pix_sci = temp_sci.copy()
                pix_sci[pix_sci != 0] = np.nan
                pix_sci[pix_sci == 0] = 1
                invalid_pix = np.nansum(pix_sci)
                invalid.append(invalid_pix)
            
            photometry[filt + '_flux'] = fluxes*u.Jy
            photometry[filt + '_err'] = uncerts*u.Jy
            
            valid = nPixels - np.array(invalid)
            photometry[filt + '_nPix'] = np.int_(valid)
        
        photometry.write(outfile)
    
    return

def get_filters(survey) :
    
    castor = ['castor_uv', 'castor_u', 'castor_g']
    hst = ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w', 'hst_f390w',
           'hst_f438w', 'hst_f435w', 'hst_f475w', 'hst_f555w', 'hst_f606w',
           'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp', 'hst_f105w',
           'hst_f110w', 'hst_f125w', 'hst_f140w', 'hst_f160w']
    jwst = ['jwst_f070w', 'jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w',
            'jwst_f277w', 'jwst_f356w', 'jwst_f410m', 'jwst_f444w']
    roman = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
             'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213']
    
    if survey == 'all' :
        filters = castor + hst + jwst + roman
    
    if survey == 'castor_roman' :
        filters = castor + roman
    
    if survey == 'hst_jwst' :
        filters = hst + jwst
    
    return filters

def join_all_photometry(survey='castor_roman', hlwas=True, population='quenched') :
    
    inDir = 'photometry/{}/'.format(population)
    outfile = 'photometry/photometry.cat'
    
    # table = Table.read('tools/subIDs.fits')
    # subIDs = table['subIDs'].data
    subIDs = [96771] #[63871, 96771, 198186] # for testing
    
    filters = get_filters(survey)
    if hlwas :
        filters = filters[:3] + ['roman_f106', 'roman_f129', 'roman_f158',
                                 'roman_f184']
    
    translate = {'castor_uv':'F314', 'castor_u':'F315', 'castor_g':'F316',
                 'hst_f218w':'F317', 'hst_f225w':'F318', 'hst_f275w':'F319', 
                 'hst_f336w':'F320', 'hst_f390w':'F321', 'hst_f438w':'F322',
                 'hst_f435w':'F323', 'hst_f475w':'F324', 'hst_f555w':'F325',
                 'hst_f606w':'F326', 'hst_f625w':'F327', 'hst_f775w':'F328',
                 'hst_f814w':'F329', 'hst_f850lp':'F330', 'hst_f105w':'F331',
                 'hst_f110w':'F332', 'hst_f125w':'F333', 'hst_f140w':'F334',
                 'hst_f160w':'F335',
                 'jwst_f070w':'F336', 'jwst_f090w':'F337', 'jwst_f115w':'F338',
                 'jwst_f150w':'F339', 'jwst_f200w':'F340', 'jwst_f277w':'F341',
                 'jwst_f356w':'F342', 'jwst_f410m':'F343', 'jwst_f444w':'F344',
                 'jwst_f560w':'F345', 'jwst_f770w':'F346', 'jwst_f1000w':'F347',
                 'jwst_f1130w':'F348', 'jwst_f1280w':'F349', 'jwst_f1500w':'F350',
                 'jwst_f1800w':'F351', 'jwst_f2100w':'F352', 'jwst_f2550w':'F353',
                 'roman_f062':'F354', 'roman_f087':'F355', 'roman_f106':'F356',
                 'roman_f129':'F357', 'roman_f146':'F358', 'roman_f158':'F359',
                 'roman_f184':'F360', 'roman_f213':'F361'}
    
    # determine the names that will be used in the final photometric table
    names = ['id']
    for filt in filters :
        names.append(translate[filt])
        names.append(translate[filt].replace('F', 'E'))
    names.append('z_spec')
    
    # loop over all the galaxies in the sample
    tables_to_stack =[]
    for subID in subIDs :
        # get the photometry for an individual galaxy
        infile = inDir + 'subID_{}_photometry.fits'.format(subID)
        table = Table.read(infile)
        
        # get unique identifications for each elliptical annulus
        bins = [str(subID) + '_bin_' + str(binNum) for binNum in table['bin'].data]
        # bins = [str(subID) + '_run_' + str(i) + '_bin_' + str(binNum)
        #         for binNum in table['bin'].data]
        
        # get photometry from the table, and include additional redshift info
        columns = [bins]
        for filt in filters :
            columns.append(table[filt + '_flux'].data)
            columns.append(table[filt + '_err'].data)
        columns.append(table['z'])
        
        # construct a new table
        new = Table(columns, names=names)
        tables_to_stack.append(new)
    
    # stack the photometry, thereby creating a master table of photometry to fit
    final = vstack(tables_to_stack)
    final.write(outfile, format='ascii.commented_header')
    
    return
