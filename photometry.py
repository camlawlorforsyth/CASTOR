
import os
import glob
import numpy as np

from astropy.table import Table
import astropy.units as u

from core import open_cutout

def all_fluxes(filters, population='quenched') :
    
    outDir = 'photometry/{}/'.format(population)
    
    os.makedirs(outDir, exist_ok=True) # ensure the output directory for the
        # photometric tables is available
    
    # table = Table.read('subIDs.fits')
    # subIDs = table['subIDs'].data
    subIDs = [96771] # for testing
    
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

def join_all_photometry(population='quenched') :
    
    inDir = 'photometry/{}/'.format(population)
    outfile = 'photometry/photometry.cat'
    
    # table = Table.read('subIDs.fits')
    # subIDs = table['subIDs'].data
    subIDs = [96771] # for testing
    
    translate = {'castor_uv':'F1', 'castor_u':'F2', 'castor_g':'F3',
                 'roman_f106':'F4', 'roman_f129':'F5', 'roman_f158':'F6',
                 'roman_f184':'F7'}
    
    for subID in subIDs :
        infile = inDir + 'subID_{}_photometry.fits'.format(subID)
        table = Table.read(infile)
        bins = [str(subID) + '_' + str(binNum) for binNum in table['bin'].data]
        
        columns = [bins]
        names = ['id']
        for key in translate.keys():
            columns.append(table[key + '_flux'].data)
            names.append(translate[key])
            columns.append(table[key + '_err'].data)
            names.append(translate[key].replace('F', 'E'))
        columns.append(table['z'])
        names.append('z_spec')
        
        new = Table(columns, names=names)
        new.write(outfile, format='ascii.commented_header')
    
    return
