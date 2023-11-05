
# import os
import numpy as np

from astropy.io import fits
from astropy.table import Table
import astropy.units as u

import plotting as plt

def prepare_photometry_for_fitting() :
    
    processedDir = 'SKIRT/SKIRT_processed_images_quenched/'
    outDir = 'fastpp/'
    # subIDs = os.listdir(processedDir)
    
    # dims = []
    # for subID in subIDs :
    #     uv_infile = '{}{}/castor_ultradeep_uv.fits'.format(processedDir, subID)
    #     with fits.open(uv_infile) as hdu :
    #         processed_image = hdu[0].data # janskys [per pixel]
    #         dim = processed_image.shape[0]
    #         dims.append(dim)
    # dims = [441, 316, 79, 100, 357, 172, 99, 126, 226, 108, 107, 218, 225, 73, 138,
    #         198, 152, 161, 98, 153, 107, 90, 138, 126, 183, 87, 57, 366, 415, 88,
    #         142, 63, 90, 101, 81, 137, 137, 219, 57, 132, 82, 236, 204, 300, 81,
    #         336, 34, 311, 91, 131, 45, 61, 178, 131, 62, 291, 126, 267, 181, 107,
    #         120, 179, 166, 178, 416, 152, 128, 121, 61, 80, 37, 172, 67, 55, 166,
    #         82, 170, 100, 69, 96, 137, 222, 319, 233, 170, 174, 188, 242, 126, 255,
    #         69, 63, 155, 237, 71, 116, 92, 229, 178, 260, 100, 150, 39, 196, 98,
    #         283, 193, 66, 116, 62, 182, 265, 240, 187, 275, 108, 209, 130, 77, 438,
    #         139, 238, 137, 253, 60, 123, 264, 162, 191, 58, 157, 73, 180, 87, 147,
    #         341, 43, 226, 101, 59, 52, 54, 75, 368, 161, 67, 124, 55, 99, 251, 162,
    #         162, 93, 172, 50, 68, 130, 193, 52, 97, 282, 212, 88, 56, 68, 171, 77,
    #         57, 62, 140, 44, 131, 48, 139, 117, 188, 132, 65, 43, 271, 263, 69, 67,
    #         86, 185, 105, 307, 46, 65, 69, 70, 41, 152, 75, 176, 107, 151, 66, 171,
    #         263, 349, 324, 157, 58, 65, 278, 56, 140, 134, 349, 203, 412, 239, 436,
    #         243, 241, 204, 111, 252, 213, 94, 249, 187, 293, 203, 83, 119, 171, 97,
    #         275, 126, 64, 119, 88, 157, 238, 109, 242, 78, 55, 55, 293, 383, 336,
    #         256, 208, 164, 235, 237, 191, 132, 119, 53, 88, 270, 109, 74, 191, 94,
    #         129, 346, 471, 80, 149, 244, 187, 356, 76, 294, 247, 248, 269]
    # num_pixs = np.square(np.array(dims)).astype(int)
    # for subID, num_pix in zip(subIDs, num_pixs) :
    #     print(subID, num_pix)
    
    subID = 96771
    filters = ['castor_ultradeep_uv', 'castor_ultradeep_u', 'castor_ultradeep_g',
               'roman_hlwas_f106', 'roman_hlwas_f129', 'roman_hlwas_f158', 'roman_hlwas_f184']
    
    # create empty array which will store all the photometry
    data = np.full((2809, 16), np.nan)
    
    # input id column which FAST++ requires
    data[:, 0] = np.arange(2809)
    
    for i, filt in enumerate(filters) :
        
        infile = '{}{}/{}.fits'.format(processedDir, subID, filt)
        with fits.open(infile) as hdu :
            processed_image = (hdu[0].data*u.Jy).to(u.uJy).value # microjanskys [per pixel]
        data[:, 2*(i + 1) - 1] = processed_image.flatten()
        
        noise_infile = '{}{}/{}_noise.fits'.format(processedDir, subID, filt)
        with fits.open(noise_infile) as hdu :
            noise_image = (hdu[0].data*u.Jy).to(u.uJy).value # microjanskys [per pixel]
        data[:, 2*(i + 1)] = noise_image.flatten()
    
    # input spectroscopic redshift information
    data[:, -1] = 0.5*np.ones(2809)
    
    fmt = ['%3i'] + 14*['%12.5e'] + ['%.1f']
    header = 'id F1 E1 F2 E2 F3 E3 F4 E4 F5 E5 F6 E6 F7 E7 z_spec'
    np.savetxt(outDir + '96771.cat', data, fmt=fmt, header=header)
    
    return

# read the table
data = np.loadtxt('fastpp/96771_1000_sims.fout')
dim = np.sqrt(len(data)).astype(int)

# lmass = data[:, 16]
# plt.histogram(lmass, 'lmass', bins=53)

# plt.display_image_simple(np.rot90(lmass.reshape(dim, dim), k=3), lognorm=False,
#                          vmin=4, vmax=9) # rotation required to match TNG orientation

lsfr = data[:, 19]
# plt.histogram(lsfr[lsfr > -15], 'lsfr', bins=53)

plt.display_image_simple(np.rot90(lsfr.reshape(dim, dim), k=3), lognorm=False,
    vmin=-3.3, vmax=-2.4)

# lssfr = data[:, 22]
# plt.histogram(lssfr[lssfr > -12], 'lssfr', bins=30)

# chi2 = data[:, 28]
# plt.histogram(chi2, 'chi2', bins=20)

