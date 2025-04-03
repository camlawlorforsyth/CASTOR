
import os
import numpy as np

from astropy.table import Table
from vorbin.voronoi_2d_binning import voronoi_2d_binning

import plotting as plt
from core import open_cutout

from astropy.nddata import Cutout2D

def vorbin_all(population='quenched', detection_filter='roman_f184'):
    
    inDir = 'cutouts/{}/'.format(population)
    outDir = 'bins/{}/'.format(population)
    
    os.makedirs(outDir, exist_ok=True) # ensure the output directory for the
        # vorbins is available
    
    # table = Table.read('tools/subIDs.fits')
    # subIDs = table['subID'].data
    # subIDs = subIDs[(subIDs < 14) | (subIDs > 14)] # subID 14's SKIRT run failed
    # subIDs = [63871, 96771, 198186] # for testing
    
    for subID in [198186] : #subIDs[:10] :
        sci_file = inDir + '{}/roman_f184_hlwas.fits'.format(subID)
        noise_file = inDir + '{}/roman_f184_hlwas_noise.fits'.format(subID)
        
        # use the CASTOR UV ultradeep images for voronoi binning
        # sci_file = inDir + '{}/castor_uv_ultradeep.fits'.format(subID)
        # noise_file = inDir + '{}/castor_uv_ultradeep_noise.fits'.format(subID)
        sci, shape = open_cutout(sci_file, simple=True)
        noise, _ = open_cutout(noise_file, simple=True)
        
        # find the center of the images and the cropped size
        center = (shape[0] - 1)/2
        size = int(shape[0]/2)
        if size % 2 == 0 :
            size += 1 # ensure even arrays become odd
        else :
            size += 2 # ensureodd array become slightly larger
        
        # crop the images
        sci = Cutout2D(sci, (center, center), (size, size)).data
        noise = Cutout2D(noise, (center, center), (size, size)).data
        plt.display_image_simple(sci/noise, lognorm=False)
        
        # target a signal-to-noise ratio of 30 per voronoi bin
        sci[sci < 0] = 0.0 # vorbin has an issue with 
        xs, ys, binNum, _, _, SN, nPixels = vorbin_data(sci, noise, sci.shape, 10)
        vorbins_image = binNum.reshape(sci.shape) # reshape the bins into an image
        plt.display_image_simple(vorbins_image, lognorm=False)
    
    #     outfile = '{}/vorbins/{}_ID_{}_vorbins.npz'.format(cluster, cluster, ID)
    #     np.savez(outfile, image=bins_image, x=xs, y=ys, binNum=binNum,
    #              xbar=xBar, ybar=yBar, SN=SN, nPixels=nPixels)
    
    # save vorbins as fits image for easier use? check to see what default
    # output type is like
    
    
    return

def vorbin_data(signal, noise, dim, targetSN) :
    '''
    Use the vorbin package to bin the data using Voronoi tesselations.
    
    Parameters
    ----------
    signal : numpy.ndarray
        The data to adaptively bin.
    noise : numpy.ndarray
        The noise corresponding to the signal.
    dim : tuple
        The dimension of the input arrays.
    targetSN : float
        The target signal-to-noise ratio to use for binning.
    display : bool, optional
        Boolean to plot the resulting bin image. The default is False.
    printraw : bool, optional
        Boolean to print the resulting bin image. The default is False.
    
    '''
    
    signal = signal.flatten() # vorbin requires 1D arrays
    noise = noise.flatten()
    
    ys, xs = np.mgrid[:dim[0], :dim[1]] # position arrays for each pixel,
    xs = xs.flatten() - dim[1]/2        # relative to the center of the image
    ys = ys.flatten() - dim[0]/2
    
    (binNum, xNode, yNode, xbar, ybar, sn, npixels,
     scale) = voronoi_2d_binning(xs, ys, signal, noise, targetSN, pixelsize=1,
                                 plot=False, quiet=True)
    
    return xs, ys, binNum, xbar, ybar, sn, npixels

def find_center(dim) :
    return (dim - 1)/2

vorbin_all()

# a = np.outer([0, 1, 2, 3, 2, 1, 0], [0, 1, 2, 3, 2, 1, 0])
# print(a)
# center = find_center(a.shape[0])
# cutout = Cutout2D(a, (center, center), (3, 3)).data
# print(cutout)
# print()
# b = np.outer([0, 1, 2, 2, 1, 0], [0, 1, 2, 2, 1, 0])
# print(b)
# center = find_center(b.shape[0])
# cutout = Cutout2D(b, (center, center), (2, 2)).data
# print(cutout)


