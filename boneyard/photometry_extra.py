
import glob
import numpy as np

from astropy.table import Table
import astropy.units as u

from core import open_cutout

def determine_fluxes_elliptical_annuli(subID, filters, population='quenched') :
    
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
    
    castor = ['castor_uv', 'castor_uvl', 'castor_us', 'castor_u', 'castor_g']
    euclid = ['euclid_ie', 'euclid_ye', 'euclid_je', 'euclid_he']
    hst = ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w', 'hst_f390w',
           'hst_f438w', 'hst_f435w', 'hst_f475w', 'hst_f555w', 'hst_f606w',
           'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp', 'hst_f105w',
           'hst_f110w', 'hst_f125w', 'hst_f140w', 'hst_f160w']
    jwst = ['jwst_f070w',  'jwst_f090w',  'jwst_f115w',  'jwst_f150w',  'jwst_f200w',
            'jwst_f277w',  'jwst_f356w',  'jwst_f410m',  'jwst_f444w',  'jwst_f560w',
            'jwst_f770w',  'jwst_f1000w', 'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w',
            'jwst_f1800w', 'jwst_f2100w', 'jwst_f2550w']
    roman = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
             'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213']
    
    if survey == 'all' :
        filters = castor + euclid + hst + jwst + roman
    
    if survey == 'castor_roman' :
        filters = castor + roman
    
    if survey == 'hst_jwst' :
        filters = hst + jwst
    
    return filters
