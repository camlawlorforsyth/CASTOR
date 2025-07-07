
import os
import numpy as np

from astropy.convolution import convolve
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip

from core import load_massive_galaxy_sample
import plotting as plt
from photometry import determine_castor_snr_map, determine_roman_snr_map

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def create_all_mass_and_sfr_maps(model_redshift=0.5) :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # select only the quenched galaxies at the first snapshot >=75% of the way
    # through their quenching episodes
    mask = (((sample['mechanism'] == 1) | (sample['mechanism'] == 3)) &
        (sample['episode_progress'] >= 0.75))
    sample = sample[mask]
    
    # use the first snapshot >=75% of the way through the quenching episode,
    # but not any additional snapshots, for testing purposes
    mask = np.full(len(sample), False)
    idx = 0
    for subIDfinal in np.unique(sample['subIDfinal']) :
        mask[idx] = True
        idx += len(np.where(sample['subIDfinal'] == subIDfinal)[0])
    sample = sample[mask]
    
    # process every galaxy/snapshot pair
    for subID, snap in zip(sample['subID'], sample['snapshot']) :
        outfile = 'maps/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))
        if not os.path.exists(outfile) :
            create_mass_and_sfr_maps(snap, subID, model_redshift=model_redshift)
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def create_mass_and_sfr_maps(snap, subID, model_redshift=0.5, save=True) :
    
    # get the fit results
    results = np.loadtxt('fits/fits_2June2025.fout', dtype=str, skiprows=18)
    fit_logM = results[:, 5].astype(float)
    fit_logSFR = results[:, 14].astype(float)
    
    # define which rows to use, based on the 'id' containing the snap and subID
    ids_info = np.stack(np.char.split(results[:, 0], sep='_').ravel())
    isnap = ids_info[:, 0].astype(int)
    isubID = ids_info[:, 1].astype(int)
    pixel = ids_info[:, 3].astype(int)
    use = (isnap == snap) & (isubID == subID)
    
    # open the cutouts
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        hdr = hdu[0].header
        images = hdu[0].data
    uv_image = images[0]
    hband_image = images[14]
    
    # use the CASTOR/Roman images to create average SNR maps for the UV/H-band
    uv_snr = determine_castor_snr_map(images)
    hband_snr = determine_roman_snr_map(images)
    
    # create the maps
    sfr_image = create_map(uv_image, uv_snr, hband_snr, pixel[use],
                           fit_logSFR[use], version='sfr')
    mass_image = create_map(hband_image, uv_snr, hband_snr, pixel[use],
                            fit_logM[use], version='mass')
    
    # save the output to file
    if save :
        os.makedirs('maps/', exist_ok=True) # ensure the output directory for
        # the maps is available
        
        # set the output filename
        outfile = 'maps/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))
        
        # copy relevant header information from the cutouts header
        header = fits.Header()
        header['ORIGIN'] = ('GALAXEV pipeline and FAST++', 'adapted by CLF')
        header['BUNIT'] = ('solMass/pixel, solMass/yr/pixel',
                           'Physical unit of the array values')
        header['CDELT1'] = hdr['CDELT1']
        header.comments['CDELT1'] = hdr.comments['CDELT1']
        header['CDELT2'] = hdr['CDELT2']
        header.comments['CDELT2'] = hdr.comments['CDELT2']
        header['REDSHIFT'] = hdr['REDSHIFT']
        header.comments['REDSHIFT'] = hdr.comments['REDSHIFT']
        
        # write to a fits file
        data = np.array([mass_image, sfr_image])
        hdulist = fits.HDUList([fits.PrimaryHDU(data=data, header=hdr)])
        hdulist.writeto(outfile)
    
    return

def create_map(image, uv_snr_image, hband_snr_image, highSNRpixels,
               fastpp_values, version='mass') :
    
    # use an averaging filter to determine the average for adjacent pixels,
    # including the central pixel; inspired by
    # https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html#the-mean-filter
    mean_kernel = np.full((3, 3), 1/9)
    
    # average the UV/H-band flux image, setting negative pixels to zero
    copied_image = image.copy()
    copied_image[copied_image < 0] = 0.0
    average_image = convolve(copied_image, mean_kernel)
    
    # for pixels with low SNR (ie. SNR < 10), replace those pixels with the
    # values from the averaged map
    best_flux_image = image.copy()
    if version == 'mass' :
        mask = (hband_snr_image < 10)
    if version == 'sfr' :
        mask = (uv_snr_image < 10)
    best_flux_image[mask] = average_image[mask]
    best_flux_image[best_flux_image <= 0] = np.min(
        best_flux_image[best_flux_image > 0]) # replace negatives/zeros
    
    if version == 'mass' :
        expected_image = hband_flux_to_logM(best_flux_image)
    if version == 'sfr' :
        expected_image = uv_flux_to_logsfr(best_flux_image)
    
    # create an image with fitted values for the high-SNR (ie. SNR >= 10) pixels
    all_pixels = np.arange(image.shape[0]*image.shape[1])
    fitted_value_image = np.zeros_like(all_pixels, dtype=float)
    for i, pix in enumerate(all_pixels) :
        if pix in highSNRpixels :
            loc = np.where(highSNRpixels == pix)[0][0]
            fitted_value_image[i] = fastpp_values[loc]
    fitted_value_image = fitted_value_image.reshape(image.shape)
    
    # inject the actual fitted results into the expected image
    log_image = expected_image.copy()
    # log_image[hband_snr_image >= 10] = fitted_value_image[hband_snr_image >= 10]
    
    # convert to physical maps from log images
    final_image = np.power(10, log_image)
    
    # mask out extremely low-SNR pixels
    if version == 'mass' :
        final_image[hband_snr_image < 1] = 0.0
    if version == 'sfr' :
        final_image[uv_snr_image < 1] = 0.0
    
    return final_image

def conversion_factors_pixelbypixel(model_redshift=0.5) :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # select only the quenched galaxies at the first snapshot >=75% of the way
    # through their quenching episodes
    mask = (((sample['mechanism'] == 1) | (sample['mechanism'] == 3)) &
        (sample['episode_progress'] >= 0.75))
    sample = sample[mask]
    
    # use the first snapshot >=75% of the way through the quenching episode,
    # but not any additional snapshots, for testing purposes
    mask = np.full(len(sample), False)
    idx = 0
    for subIDfinal in np.unique(sample['subIDfinal']) :
        mask[idx] = True
        idx += len(np.where(sample['subIDfinal'] == subIDfinal)[0])
    sample = sample[mask]
    
    # create a mask for the UV pixels to use, based on the H-band SNR map
    uv_select = []
    for snap, subID in zip(sample['snapshot'], sample['subID']) :
        with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))) as hdu :
            images = hdu[0].data
        m1 = (determine_castor_snr_map(images) >= 10)
        m2 = (determine_roman_snr_map(images) >= 10)
        use = m1[m2] # use the Roman SNR map as a mask on the CASTOR SNR map
        for val in use :
            uv_select.append(val)
    uv_select = np.array(uv_select)
    
    # get the photometry
    photometry = np.loadtxt('photometry/photometry_2June2025.cat',
        dtype=str, skiprows=1)[:, 1:-1].astype(float) # len = 9425
    uv_flux = photometry[:, 0][uv_select] # mask to the (UV+opt+NIR) high-SNR pixels
    h_flux = photometry[:, 14] # already masked to the NIR high-SNR pixels
    
    # load the fitted results coming out of FAST++
    results = np.loadtxt('fits/fits_2June2025.fout',
        dtype=str, skiprows=18)[:, 2:-1].astype(float)
    fit_logM = results[:, 3]
    fit_logSFR = results[:, -1][uv_select]
    
    # fit a line to the stellar masses based on H-band flux, with sigma clipping
    # adapted from
    # https://docs.astropy.org/en/latest/modeling/example-fitting-line.html
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
        sigma_clip, niter=1, sigma=3) # stable after 1 iteration
    fitted_line, mask = fitter(models.Linear1D(), np.log10(h_flux), fit_logM)
    slope, intercept = fitted_line.parameters # 0.9968073724866119, 14.700207403923189
    x = np.linspace(-7.1, -4.9, 1000); y = slope*x + intercept
    corr = fit_logM[~mask] - (slope*np.log10(h_flux)[~mask] + intercept)
    plt.plot_simple_multi([x, x, x, np.log10(h_flux)[~mask]],
        [y, y - 0.3, y + 0.3, fit_logM[~mask] - 0.2*corr],
        ['fit', r'fit $\pm$ 0.3 dex', '', ''], ['k', 'grey', 'grey', 'b'],
        ['', '', '', 'o'], ['-', '-', '-', ''], [1, 0.5, 0.5, 0.05], [],
        xmin=-7.1, xmax=-4.9, ymin=7.2, ymax=9.7,
        xlabel='Roman log(H-band flux/Jy)', ylabel='FAST++ logM')
    
    # fit a line to the SFRs based on UV flux, with sigma clipping
    realistic = (fit_logSFR >= np.log10(uv_flux) + 5) # mask unphysically low values
    uv_flux = uv_flux[realistic]; fit_logSFR = fit_logSFR[realistic]
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
        sigma_clip, niter=2, sigma=3) # stable after 2 iterations
    fitted_line, mask = fitter(models.Linear1D(), np.log10(uv_flux), fit_logSFR)
    slope, intercept = fitted_line.parameters # 0.9945731443936217, 6.362818780111098
    x = np.linspace(-10, -7, 1000); y = slope*x + intercept
    corr = fit_logSFR[~mask] - (slope*np.log10(uv_flux)[~mask] + intercept)
    plt.plot_simple_multi([x, x, x, np.log10(uv_flux)[~mask]],
        [y, y - 0.5, y + 0.5, fit_logSFR[~mask] - 0.2*corr],
        ['fit', r'fit $\pm$ 0.5 dex', '', ''], ['k', 'grey', 'grey', 'b'],
        ['', '', '', 'o'], ['-', '-', '-', ''], [1, 0.5, 0.5, 0.05], [],
        xmin=-10, xmax=-7, ymin=-4, ymax=0,
        xlabel='CASTOR log(UV-band flux/Jy)', ylabel='FAST++ logSFR')
    
    return

def hband_flux_to_logM(image) :
    return np.log10(image)*0.9968073724866119 + 14.700207403923189

def uv_flux_to_logsfr(image) :
    return np.log10(image)*0.9945731443936217 + 6.362818780111098
