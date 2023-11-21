
import os
import glob
import numpy as np

from astropy.table import Table

from core import open_cutout
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def bin_all(population='quenched', detection_filter='roman_f184') :
    
    outDir = 'bins/{}/'.format(population)
    
    os.makedirs(outDir, exist_ok=True) # ensure the output directory for the
        # bins is available
    
    # table = Table.read('tools/subIDs.fits')
    # subIDs = table['subIDs'].data
    subIDs = [63871, 96771, 198186] # for testing
    
    for subID in subIDs :
        (annuli_map, smas, smbs, fluxes, errs, nPixels, widths,
         pas) = annuli_bins(subID, population=population)
        
        outfile = outDir + 'subID_{}_annuli.npz'.format(subID)
        np.savez(outfile, image=annuli_map, sma=smas, smb=smbs, flux=fluxes,
                 err=errs, nPixels=nPixels, width=widths, pa=pas)
    
    return

def annuli_bins(subID, population='quenched', detection_filter='roman_f184',
                targetSNR=10, rin=0, IR_pixel_scale=0.11) :
    
    cutoutDir = 'cutouts/{}/{}/'.format(population, subID)
    seppDir = 'detection/{}/{}/'.format(population, subID)
    
    # get the reddest NIR band cutout which best traces the stellar mass, in
    # order to compute the elliptical annuli bins
    # open the science images and the corresponding noise images
    sci_file, noise_file = glob.glob(cutoutDir + detection_filter + '_*.fits')
    sci, shape, _, _, _, _, image_pixel_scale = open_cutout(sci_file)
    # plt.display_image_simple(sci, vmin=1, vmax=1000)
    dim = shape[0]
    
    # get the corresponding noise cutout
    noise, _ = open_cutout(noise_file, simple=True)
    # plt.display_image_simple(noise, vmin=4, vmax=30)
    
    # get the SourceXtractorPlusPlus-derived segmentation map
    segmap_file = seppDir + 'segmap.fits'
    segMap, _ = open_cutout(segmap_file, simple=True)
    # plt.display_image_simple(segMap, lognorm=False)
    
    # open the SourceXtractorPlusPlus-derived catalog file for morphological
    # measurements
    catalog = Table.read(seppDir + 'cat.fits')
    detID = catalog['group_id'].data[0]
    # sma = catalog['ellipse_a'].data[0]
    # smb = catalog['ellipse_b'].data[0]
    # eta = 1 - smb/sma # the ellipticity of the ellipse
    pa_deg = catalog['ellipse_theta'].data[0]
    eta = catalog['ellipticity'].data[0] # the ellipticity of the ellipse
    r_e = catalog['flux_radius'].data[0] # [pix] the half flux radius
    
    # determine the center of the ellipse, based on the size of the cutout
    xx, yy = np.indices(shape)
    x0, y0 = np.median(xx), np.median(yy)
    
    # make a copy of the science image and noise image
    new_sci = sci.copy()
    new_noise = noise.copy()
    
    # mask the copied images based on the segmentation map, but don't mask out
    # the sky -> mask out pixels associated with other galaxies
    new_sci[(segMap > 0) & (segMap != detID)] = 0
    new_noise[(segMap > 0) & (segMap != detID)] = 0
    
    # determine the minimum size of the elliptical annuli, where IR_pixel_scale
    # is in units of arcsec/pixel, and image_pixel_scale is also in arcsec/pixel
    dr = max(IR_pixel_scale/image_pixel_scale, 0.1*r_e)
    
    # convert the position angle to radians
    pa = np.pi*pa_deg/180
    
    annulus = np.full(shape, False) # seed an ampty array to ensure that the
        # annuli can start to be found
    rins, fluxes, errs, widths, annuli, nPixels_list = [], [], [], [], [], []
    while continue_growing(annulus, dim) :
        flux, err, rnew, width, annulus, nPixels = compute_annuli(new_sci,
            new_noise, shape, (x0, y0), rin, dr, eta, pa, targetSNR)
        
        if continue_growing(annulus, dim) :
            fluxes.append(flux)
            errs.append(err)
            rins.append(rnew)
            widths.append(width)
            annuli.append(annulus)
            nPixels_list.append(nPixels)
        rin = rnew
    
    # create the annuli map for subsequent determination of photometric fluxes
    annuli_map = np.zeros(shape)
    for i in range(len(annuli)) :
        annuli_map += (i+1)*annuli[i]
    annuli_map[annuli_map == 0] = np.nan
    annuli_map -= 1
    plt.display_image_simple(annuli_map, lognorm=False)
    
    smas = np.array(rins) # the semi-major axes of the inner annuli
    smbs = (1 - eta)*smas # the semi-minor axes of the inner annuli
    widths = np.array(widths)
    pas = np.array([pa_deg]*len(smas))
    
    return annuli_map, smas, smbs, fluxes, errs, nPixels_list, widths, pas

def compute_annuli(sci, noise, dim, xy, rin, dr, eta, pa, targetSN) :
    
    annulus, nPixels = elliptical_annulus(dim, xy, rin, dr, eta, pa)
    flux, err = compute_SNR(sci, noise, annulus)
    if flux/err < targetSN :
        try :
            return compute_annuli(sci, noise, dim, xy, rin, dr+0.01, eta, pa,
                                  targetSN)
        except RecursionError :
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else :
        return flux, err, rin+dr, dr, annulus, nPixels

def compute_SNR(sci, noise, annulus) :
    
    sci, noise = sci.copy(), noise.copy()
    sci[~annulus], noise[~annulus] = 0, 0
    flux, err = np.sum(sci), np.sqrt(np.sum(np.square(noise)))
    
    return flux, err

def continue_growing(annulus, dim) :
    # adapted from https://stackoverflow.com/questions/41200719
    
    # get all the edge values of a given array
    border = np.concatenate([annulus[[0, dim-1], :].ravel(),
                             annulus[1:-1, [0, dim-1]].ravel()])
    if np.sum(border) > 0 :
        return False
    else :
        return True

def elliptical_annulus(dim, xy, rin, dr, eta, pa) :
    
    inner = elliptical_mask(dim, xy, rin, eta, pa)
    outer = elliptical_mask(dim, xy, rin+dr, eta, pa)
    annulus = np.bitwise_xor(inner, outer) # this is the region of overlap
    
    nPixels = np.sum(annulus)
    
    return annulus, nPixels

def elliptical_mask(dim, xy, rin, eta, pa) :
    
    YY, XX = np.ogrid[:dim[0], :dim[1]]
    
    left_num = np.square((XX-xy[0])*np.cos(pa) + (YY-xy[1])*np.sin(pa)) 
    left_denom = np.square(rin)
    
    right_num = np.square(-(XX-xy[0])*np.sin(pa) + (YY-xy[1])*np.cos(pa))
    right_denom = np.square((1-eta)*rin)
    
    ellipse = left_num/left_denom + right_num/right_denom
    mask = ellipse <= 1
    
    return mask
