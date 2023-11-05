
import os
import numpy as np

from astropy.table import Table

from core import open_cutout

def bin_all(cluster) :
    
    os.makedirs('{}/bins'.format(cluster), exist_ok=True) # ensure the output
        # directory for the bins is available
    
    table = Table.read('{}/{}_sample-with-use-cols.fits'.format(cluster, cluster))
    IDs = table['id']
    
    for ID in IDs :
        (annuli_map, smas, smbs, fluxes, errs,
         nPixels, widths, pas) = annuli_bins(cluster, int(ID))
        
        outfile = '{}/bins/{}_ID_{}_annuli.npz'.format(cluster, cluster, ID)
        np.savez(outfile, image=annuli_map, sma=smas, smb=smbs,
                 flux=fluxes, err=errs, nPixels=nPixels, width=widths, pa=pas)
    
    return

def annuli_bins(cluster, ID) :
    
    IR_pixel_scale = 0.128 # arcsec/pixel
    image_pixel_scale = 0.06 # arcsec/pixel
    targetSN = 30 # the target signal-to-noise ratio to use
    rin = 0 # the starting value for the semi-major axis of the ellipse
    
    (sci, dim, photnu, r_e,
     redshift, sma, smb, pa_deg) = open_cutout(
         '{}/cutouts/{}_ID_{}_f160w.fits'.format(cluster, cluster, ID))
    noise = open_cutout(
        '{}/cutouts/{}_ID_{}_f160w_noise.fits'.format(cluster, cluster, ID), 
        simple=True)
    segMap = open_cutout(
        '{}/cutouts/{}_ID_{}_segmap.fits'.format(cluster, cluster, ID),
        simple=True)
    bCGsegMap = open_cutout('{}/cutouts/{}_ID_{}_segmap-bCG.fits'.format(
        cluster, cluster, ID), simple=True)
    
    eta = 1 - smb/sma # the ellipticity of the ellipse
    
    # determine the center of the ellipse, based on the size of the cutout
    xx, yy = np.indices(dim)
    x0, y0 = np.median(xx), np.median(yy)
    
    # make a copy of the science image and noise image
    new_sci = sci.copy()
    new_noise = noise.copy()
    
    # mask the copied images based on the segmentation map, but don't mask out
    # the sky
    if ID >= 20000 : # the bCGs aren't in the segmap, so mask any other galaxy
        new_sci[(segMap > 0) | ((bCGsegMap > 0) & (bCGsegMap != ID))] = 0
        new_noise[(segMap > 0) | ((bCGsegMap > 0) & (bCGsegMap != ID))] = 0
    else : # for the non-bCGs, mask out pixels associated with other galaxies
        new_sci[(segMap > 0) & (segMap != ID)] = 0
        new_noise[(segMap > 0) & (segMap != ID)] = 0
    
    dr = max(IR_pixel_scale/image_pixel_scale, 0.1*r_e)
    pa = np.pi*pa_deg/180
    
    rins, fluxes, errs, widths, annuli, nPixels_list = [], [], [], [], [], []
    while rin < 5*r_e :
        (flux, err, rnew, width,
         annulus, nPixels) = compute_annuli(new_sci, new_noise, dim, (x0, y0),
                                            rin, dr, eta, pa, targetSN)
        if rnew < 5*r_e :
            fluxes.append(flux)
            errs.append(err)
            rins.append(rnew)
            widths.append(width)
            annuli.append(annulus)
            nPixels_list.append(nPixels)
        rin = rnew
    
    # create the annuli map for subsequent determination of photometric fluxes
    annuli_map = np.zeros(dim)
    for i in range(len(annuli)) :
        annuli_map += (i+1)*annuli[i]
    annuli_map[annuli_map == 0] = np.nan
    annuli_map -= 1
    
    smas = np.array(rins) # the semi-major axes of the inner annuli
    smbs = (1-eta)*smas # the semi-minor axes of the inner annuli
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
