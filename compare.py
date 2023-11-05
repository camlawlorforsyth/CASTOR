
from os.path import exists
import numpy as np

from astropy.io import fits
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter

# from CASTOR_proposal import spatial_plot_info
from noise import add_noise_and_psf
import plotting as plt

def compare_raw_to_SKIRT(subIDfinal, snap, subID, Re, center) :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
    
    # infile = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    skirtDir = 'SKIRT/SKIRT_output_quenched/'
    processedDir = 'SKIRT/SKIRT_processed_images_quenched/'
    outDir = 'TNG50-1/figures/comparison_plots_SFR/'
    
    # test UV flux
    # minAge, maxAge = 0, 100
    # minAge, maxAge = 100, 250
    # minAge, maxAge = 250, 500
    # minAge, maxAge = 500, 1000
    # skirtDir = 'SKIRT/SKIRT_input_UV_flux_test/{}_{}_Myr/'.format(minAge, maxAge)
    # processedDir = 'SKIRT/SKIRT_input_UV_flux_test/{}_{}_Myr/'.format(minAge, maxAge)
    # outDir = 'SKIRT/SKIRT_input_UV_flux_test/'
    
    # use raw output from SKIRT, without adding noise
    uv_infile = '{}{}/{}_CASTOR_total.fits'.format(skirtDir, subIDfinal, subIDfinal)
    with fits.open(uv_infile) as hdu :
        skirt_pixel_area = np.square((hdu[0].header)['CDELT1']*u.arcsec)
        skirt_image = (hdu[0].data*u.MJy/u.sr*skirt_pixel_area).to(u.Jy).value
        skirt_image = skirt_image[0]
        skirt_image = np.rot90(skirt_image, k=3)
    
    nir_infile = '{}{}/{}_Roman_total.fits'.format(skirtDir, subIDfinal, subIDfinal)
    with fits.open(nir_infile) as hdu :
        skirt_pixel_area = np.square((hdu[0].header)['CDELT1']*u.arcsec)
        skirt_contour_image = (hdu[0].data*u.MJy/u.sr*skirt_pixel_area).to(u.Jy).value
        skirt_contour_image = skirt_contour_image[6]
        skirt_contour_image = np.rot90(skirt_contour_image, k=3)
    
    # use processed output, with adding noise
    uv_infile = '{}{}/castor_ultradeep_uv.fits'.format(processedDir, subIDfinal)
    if not exists(uv_infile) :
        add_noise_and_psf(subIDfinal, 'castor_ultradeep')
    with fits.open(uv_infile) as hdu :
        processed_image = hdu[0].data # janskys [per pixel]
        processed_image = np.rot90(processed_image, k=3)
    
    nir_infile = '{}{}/roman_hlwas_f184.fits'.format(processedDir, subIDfinal)
    if not exists(nir_infile) :
        add_noise_and_psf(subIDfinal, 'roman_hlwas')
    with fits.open(nir_infile) as hdu :
        processed_contour_image = hdu[0].data # janskys [per pixel]
        processed_contour_image = np.rot90(processed_contour_image, k=3)
    
    skirt_levels = np.power(10., np.array([-9.5, -8.5, -7.5]))
    # print(skirt_levels)
    # vals = np.sort(skirt_image.flatten())
    # vals = vals[vals > 0]
    # skirt_levels = np.percentile(vals, [50, 84, 95, 99])
    # print(skirt_levels)
    # print()
    
    processed_levels = np.power(10., np.array([-8.5, -7.5]))
    # print(processed_levels)
    # vals = np.sort(skirt_contour_image.flatten())
    # vals = vals[vals > 0]
    # processed_levels = np.percentile(vals, [50, 84, 95, 99])
    # print(processed_levels)
    
    spatial_edges = np.linspace(-10, 10, processed_image.shape[0] + 1)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    hist_centers = spatial_edges[:-1] + np.diff(spatial_edges)/2
    X_cent, Y_cent = np.meshgrid(hist_centers, hist_centers)
    
    # use TNG arrays, and create image similar to comprehensive plots
    tng_image, tng_contour_image, tng_levels = spatial_plot_info(times[snap],
        snap, subID, center, Re, spatial_edges, 100*u.Myr,
        nelson2021version=False, sfr_map=True)
    
    # specify and set new contour levels for the tng stellar mass image
    tng_levels = np.power(10., np.array([5.5, 6.5, 7.5]))
    
    # find vmin, vmax for tng_image
    tng_full = np.array(tng_image).flatten()
    tng_full[tng_full == 0.0] = np.nan
    tng_vmin, tng_vmax = np.nanpercentile(tng_full, [1, 99])
    
    # skirt_full = skirt_image.flatten()
    # skirt_full[skirt_full == 0.0] = np.nan
    # skirt_vmin, skirt_vmax = np.nanpercentile(skirt_full, [1, 99])
    skirt_vmin, skirt_vmax = 1e-10, 1e-8
    
    # pro_full = processed_image.flatten()
    # pro_full[pro_full <= 0.0] = np.nan
    # pro_vmin, pro_vmax = np.nanpercentile(pro_full, [1, 99])
    pro_vmin, pro_vmax = 1e-10, 1e-8
    
    # smooth contour images to match tng smoothed image
    tng_contour_image = gaussian_filter(tng_contour_image, 0.7)
    skirt_contour_image = gaussian_filter(skirt_contour_image, 0.6)
    processed_contour_image = gaussian_filter(processed_contour_image, 1.5)
    
    plt.plot_comparisons(tng_image, tng_contour_image, tng_levels,
        skirt_image, skirt_contour_image, skirt_levels, processed_image,
        processed_contour_image, processed_levels, XX, YY, X_cent, Y_cent,
        tng_vmin=tng_vmin, tng_vmax=tng_vmax, skirt_vmin=skirt_vmin,
        skirt_vmax=skirt_vmax, pro_vmin=pro_vmin, pro_vmax=pro_vmax,
        xlabel=r'$\Delta x$ ($R_{\rm e}$)', ylabel=r'$\Delta y$ ($R_{\rm e}$)',
        mtitle=r'subID$_{z = 0}$' + ' {}'.format(subIDfinal), xmin=-5, xmax=5,
        ymin=-5, ymax=5, save=True, outfile=outDir + '{}.png'.format(subIDfinal))
    
    return

# compare_raw_to_SKIRT(96771, 44, 42759, 0.825415849685669,
#                      [28307.04296875, 7637.23876953125, 4297.02587890625])
