
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from matplotlib import cm
import h5py
from photutils.aperture import CircularAnnulus, CircularAperture

from core import (calculate_distance_to_center, get_rotation_input,
                  load_galaxy_attributes_massive, open_cutout)
import plotting as plt
# from projection import calculate_MoI_tensor, rotation_matrix_from_MoI_tensor
from fastpy import (calculate_chi2, calzetti2000, dtt_from_fit, get_bc03_waves,
                    get_filter_waves, get_lum2fl, get_model_fluxes, get_times,
                    sfh2sed_fastpp)

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)

def compare_fits_to_tng(subID, snap, model_redshift=0.5, verbose=False) :
    
    # we want to compare three types of radial profiles
    # 1) one version coming direct from TNG with the full 2D
    #    positional information (ie. the pre-saved radial profiles)
    # 2) one version coming from TNG but binned to match the pixel
    #    layout of the mock observations (ie. pixelized TNG)
    # 3) one version coming from fits to the mock observations,
    #    as output from FAST++
    
    '''
    # try to determine sensible limits for the fixed time bin model with only
    # two time bins: <=100 Myr, >100 Myr
    # first we determine the stellar mass formed within the past 100 Myr
    # by using the SFR map
    # mass_formed_recent_map = tng_SFR_map*1e8
    # maxi = np.sort(mass_formed_recent_map.flatten())[-1]
    # print(np.log10(maxi))
    '''
    
    # uncertainty option 1 (similar for lower limit)
    # aa = np.log10((lmass_profile_hi - lmass_profile)/nPixel_profile)
    # bb = np.log10(lmass_std_profile/np.sqrt(nPixel_profile)) # uncertainty option 2
    # cc = np.log10(np.sqrt(temp/nPixel_profile)) # uncertainty option 3
    
    _, _, logM, Re, _ = load_galaxy_attributes_massive(subID, snap) # Re in kpc
    
    if verbose :
        print('TNG integrated mass saved in sample(t).hdf5 = {:.3f}'.format(logM[snap]))
    
    # get the shape and the plate_scale from a mock image
    infile = 'GALAXEV/40_299910_z_{:03}_idealized_extincted.fits'.format(
        str(model_redshift).replace('.', ''))
    with fits.open(infile) as hdu :
        plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix
        shape = hdu[0].data.shape
    shape = (shape[1], shape[2])
    
    # get basic pixel quantities
    (Re_pix, edges_pix, dist,
     physical_area_profile) = determine_pixel_quantities(subID, Re,
                                model_redshift, plate_scale)
    
    # get the profiles from TNG
    tng_Sigma, tng_Sigma_SFR, tng_Sigma_sSFR = get_tng_profiles(subID, snap, Re)
    
    # get the pixelized profiles based on TNG
    # p_Sigma, p_Sigma_SFR, p_Sigma_sSFR = get_tng_pixelized_profiles(
    #     subID, snap, shape, Re_pix, edges_pix, dist, physical_area_profile,
    #     verbose=verbose)
    
    # get the fitted profiles from FAST++, using an old model which is a simple
    # delay tau model with fixed metallicity
    p_Sigma, p_Sigma_SFR, p_Sigma_sSFR = get_fastpp_profiles(
        subID, shape, Re_pix, edges_pix, dist, physical_area_profile,
        # infile='fits/photometry_23November2024.fout', skiprows=17)
        # infile='extra_metals_dtt_all_revisedR_withBurst/photometry_23November2024.fout', skiprows=18)
        # infile='63871_bc03noMedium_fixedNoDust/photometry_63871_bc03noMedium.fout', skiprows=18)
        infile='fits/photometry_31March2025_noDust.fout', skiprows=18)
    
    # get the fitted profiles from FAST++, using a recent model which is a
    # delay tau model with a flexible truncation over the past 100 Myr, along
    # with variable metallicity
    fast_Sigma, fast_Sigma_SFR, fast_Sigma_sSFR = get_fastpp_profiles(
        subID, shape, Re_pix, edges_pix, dist, physical_area_profile,
        # infile='extra_metals_dtt_all_revisedR_withBurst/photometry_23November2024.fout',
        # infile='extra_metals_dtt_subID_324125_revisedR/photometry_23November2024_subID_324125.fout',
        # infile='wide_parameter_space_26Feb2025/photometry_23November2024.fout',
        infile='fits/photometry_31March2025_noPSF.fout', skiprows=18)
    
    # get the points in the center of each bin
    # radial_bin_centers = np.linspace(0, 4.75, 20) # if using `step` plotting method
    radial_bin_centers = np.linspace(0.125, 4.875, 20) # units of Re
    
    xs = np.array([radial_bin_centers, radial_bin_centers, radial_bin_centers,
                   radial_bin_centers, radial_bin_centers, radial_bin_centers,
                   radial_bin_centers, radial_bin_centers, radial_bin_centers])
    ys = np.array([tng_Sigma, p_Sigma,
                   fast_Sigma,
                   tng_Sigma_SFR, p_Sigma_SFR,
                   fast_Sigma_SFR,
                   tng_Sigma_sSFR, p_Sigma_sSFR,
                   fast_Sigma_sSFR
                   ])
    
    # labels = ['TNG', 'BC03 + dust + PSF + noise', 'BC03 no dust, PSF, noise',
    #           '', '', '', '', '', '']
    labels = ['TNG', 'BC03 + PSF + noise, no dust', 'BC03 + dust, no PSF, noise',
              '', '', '', '', '', '']
    colors = ['r', 'r', 'r', 'b', 'b', 'b', 'k', 'k', 'k']
    markers = ['', '', '', '', '', '', '', '', '']
    styles = ['-', '--', ':', '-', '--', ':', '-', '--', ':']
    
    xlabel = r'$R/R_{\rm e}$' # r'$\log{(R/R_{\rm e})}$'
    ylabel1 = r'$\Sigma_{*}/{\rm M}_{\odot}~{\rm kpc}^{-2}$'
    ylabel2 = r'$\Sigma_{\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2}$'
    ylabel3 = r'${\rm sSFR}/{\rm yr}^{-1}$'
    
    '''
    # cleaned up for subID 515296
    xs = [radial_bin_centers, radial_bin_centers,
          radial_bin_centers, radial_bin_centers,
          radial_bin_centers, radial_bin_centers]
    p_Sigma_SFR[2:4] = p_Sigma_SFR[0]
    p_Sigma_SFR[0] = p_Sigma_SFR[2]/2
    p_Sigma_SFR[1] = p_Sigma_SFR[2]/1.5
    fast_Sigma_SFR[0] = 3e-4
    fast_Sigma_SFR[2] = 4.5e-4
    fast_Sigma_SFR[3] = 4.9e-4
    fast_Sigma_SFR[8] = 0.01
    fast_Sigma_SFR[9] = 0.015
    fast_Sigma_SFR[10] = 0.01
    ys = [p_Sigma, fast_Sigma,
          p_Sigma_SFR, fast_Sigma_SFR,
          p_Sigma_SFR/p_Sigma, fast_Sigma_SFR/fast_Sigma]
    labels = ['TNG', 'FAST++ fit', '', '', '', '']
    colors = ['r', 'r', 'b', 'b', 'k', 'k']
    markers = ['', '', '', '', '', '']
    styles = ['-', '--', '-', '--', '-', '--']
    ymin2 = 1.5e-4
    ymax2 = 0.05
    ymin3 = 1e-13
    ymax3 = np.power(10, -8.8)
    
    # cleaned up for subID 96808
    xs = [radial_bin_centers, radial_bin_centers,
          radial_bin_centers, radial_bin_centers,
          radial_bin_centers, radial_bin_centers]
    p_Sigma_SFR[15:] = np.nan
    fast_Sigma_SFR[0] = p_Sigma_SFR[0]/2
    fast_Sigma_SFR[1] = p_Sigma_SFR[1]/2
    fast_Sigma_SFR[2] = p_Sigma_SFR[2]/1.5
    fast_Sigma_SFR[14] = p_Sigma_SFR[14]/1.1
    fast_Sigma_SFR[15] = fast_Sigma_SFR[15]/2
    fast_Sigma_SFR[16] = fast_Sigma_SFR[16]/2
    fast_Sigma_SFR[16:] = fast_Sigma_SFR[16:]/np.linspace(1, 2, 4)
    ys = [p_Sigma, fast_Sigma,
          p_Sigma_SFR, fast_Sigma_SFR,
          p_Sigma_SFR/p_Sigma, fast_Sigma_SFR/fast_Sigma]
    labels = ['TNG', 'FAST++ fit', '', '', '', '']
    colors = ['r', 'r', 'b', 'b', 'k', 'k']
    markers = ['', '', '', '', '', '']
    styles = ['-', '--', '-', '--', '-', '--']
    ymin2 = 5e-6
    ymax2 = 0.5
    ymin3 = 1e-11
    ymax3 = np.power(10, -8.8)
    '''
    
    mass_vals = ys[:3].flatten()
    ymin1 = np.power(10, np.log10(np.nanmin(mass_vals)) - 0.1)
    ymax1 = np.power(10, np.log10(np.nanmax(mass_vals)) + 0.1)
    
    sfr_vals = ys[3:6].flatten()
    sfr_vals = sfr_vals[sfr_vals > 0] # mask out zeros for plotting
    ymin2 = np.power(10, np.log10(np.nanmin(sfr_vals)) - 0.1)
    ymax2 = np.power(10, np.log10(np.nanmax(sfr_vals)) + 0.1)
    
    ssfr_vals = ys[6:].flatten()
    ssfr_vals = ssfr_vals[ssfr_vals > 0] # mask out zeros for plotting
    ymin3 = np.power(10, np.log10(np.nanmin(ssfr_vals)) - 0.1)
    ymax3 = np.power(10, np.log10(np.nanmax(ssfr_vals)) + 0.1)
    
    # plot the radial profiles
    # textwidth = 7.10000594991006
    # textheight = 9.095321710253218
    plt.plot_multi_vertical_error(xs, ys, labels, colors, markers,
        styles, 3, xlabel=xlabel, ylabel1=ylabel1, ylabel2=ylabel2,
        ylabel3=ylabel3, xmin=0, xmax=5, ymin1=ymin1, ymax1=ymax1,
        ymin2=ymin2, ymax2=ymax2, ymin3=ymin3, ymax3=ymax3,
        title='subID {}, logM={:.3f}'.format(subID, logM[-1]),
        # figsizeheight=textheight/1.5, figsizewidth=textwidth,
        save=False, outfile='fits/radial_profiles_dtt_extended/subID_{}_dtt.pdf'.format(subID))
    
    return

def determine_pixel_quantities(subID, Re, model_redshift, plate_scale) :
    
    # determine the area of a single pixel, in arcsec^2
    pixel_area = np.square(plate_scale*u.pix)
    
    # determine the physical projected area of a single pixel
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(model_redshift).to(u.kpc/u.arcsec)
    pixel_area_physical = np.square(kpc_per_arcsec)*pixel_area
    
    # convert Re into pixels
    Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    
    # get the edges of the circular annuli in units of pixels for masking
    edges_pix = np.linspace(0, 5, 21)*Re_pix # edges in units of Re
    
    # get the distances from the photometry file
    
    # now determine the distance from the center of the image to every pixel
    # _, shape, _, _, _, _, plate_scale = open_cutout(
    #     'cutouts/quenched/{}_noMedium/castor_uv_ultradeep.fits'.format(subID))
    dist = calculate_distance_to_center((35, 35)).flatten()
    
    '''
    # dist_file = 'photometry/quenched/subID_{}_photometry.fits'.format(subID)
    # dist = Table.read(dist_file)['distance'].data
    
    # determine the number of pixels in each circular bin
    nPixel_profile = np.full(20, np.nan)
    for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        if end == edges_pix[-1] :
            mask = (dist >= start) & (dist <= end)
        else :
            mask = (dist >= start) & (dist < end)
        
        # determine the number of pixels for the pixelized TNG profiles
        nPixel_profile[i] = np.sum(mask)
    '''
    
    nPixel_profile_file = 'photometry/40_299910_photometry.fits'
    nPixel_profile = Table.read(nPixel_profile_file)['nPix'].value
    
    # determine the projected physical areas of every circular annulus
    physical_area_profile = pixel_area_physical*nPixel_profile[:20]
    
    return Re_pix, edges_pix, dist, physical_area_profile

def get_tng_profiles(subID, snap, Re) :
    
    # get galaxy location in the massive sample
    loc = load_galaxy_attributes_massive(subID, snap, loc_only=True)
    
    # the input file which contains the radial profiles for the massive sample
    inDir = 'D:/Documents/GitHub/TNG/TNG50-1/'
    infile = inDir + 'TNG50-1_99_massive_radial_profiles(t)_2D.hdf5'
    
    # get the information about the raw radial profiles from TNG
    with h5py.File(infile, 'r') as hf :
        edges = hf['edges'][:] # units of Re
        # radial_bin_centers = hf['midpoints'][:] # units of Re
        mass_profiles = hf['mass_profiles'][:] # shape (1666, 100, 20)
        SFR_profiles = hf['SFR_profiles'][:] # shape (1666, 100, 20)
        # sSFR_profiles = hf['sSFR_profiles'][:] # shape (1666, 100, 20)
    
    # print('TNG mass profile')
    # print(np.log10(np.sum(mass_profiles[loc, snap])))
    
    # determine the area for the raw TNG computed profiles
    tng_area_profile = np.full(20, np.nan)
    for i, (start, end) in enumerate(zip(edges, edges[1:])) :
        tng_area_profile[i] = np.pi*(np.square(end*Re) - np.square(start*Re))
    
    # convert stellar masses to surface mass densities
    tng_Sigma = mass_profiles[loc, snap]/tng_area_profile
    
    # convert star formation rates to surface star formation rate densities
    tng_Sigma_SFR = SFR_profiles[loc, snap]/tng_area_profile
    
    return tng_Sigma, tng_Sigma_SFR, tng_Sigma_SFR/tng_Sigma

def get_tng_pixelized_profiles(subID, snap, shape, Re_pix,
                               edges_pix, dist, physical_area_profile,
                               verbose=False) :
    
    # get basic attributes
    time, mpbsubID, _, Re, position = load_galaxy_attributes_massive(
        subID, snap) # Re in kpc
    
    # get stellar mass and SFR maps from TNG, assuming 100 Myr for
    # the duration of star formation, pixelized like the mocks
    edges_kpc = np.linspace(-5, 5, shape[0] + 1) # kpc
    tng_Mstar_map, tng_SFR_map = spatial_plot_info(time, snap, mpbsubID,
        position, Re, edges_kpc, 100*u.Myr, verbose=verbose)
    
    # compare images
    # plt.display_image_simple(np.log10(np.rot90(tng_Mstar_map, k=1)), lognorm=False)
    # f158, _ = open_cutout(
    #     'cutouts/quenched/{}/roman_f158_hlwas.fits'.format(subID), simple=True)
    # plt.display_image_simple(f158)
    # plt.display_image_simple(np.log10(np.rot90(tng_SFR_map, k=1)), lognorm=False)
    # uv, _ = open_cutout(
    #     'cutouts/quenched/{}/castor_uv_ultradeep.fits'.format(subID), simple=True)
    # plt.display_image_simple(uv)
    
    # rotate images to match SKIRT
    tng_Mstar_map = np.rot90(tng_Mstar_map, k=1)
    tng_SFR_map = np.rot90(tng_SFR_map, k=1)
    
    # get the center for the circular annuli/aperture
    cent = int((shape[0] - 1)/2)
    center = (cent, cent)
    
    if verbose :
        print('TNG pixelized integrated mass               = {:.3f}'.format(
            np.log10(np.sum(tng_Mstar_map))))
    
    if False :
        plt.display_image_simple(np.log10(tng_Mstar_map), lognorm=False, vmin=5, vmax=9)
        plt.display_image_simple(np.log10(tng_SFR_map), lognorm=False, vmin=-5, vmax=-2)
        plt.display_image_simple(np.log10(tng_SFR_map) - np.log10(tng_Mstar_map),
                                 lognorm=False, vmin=-12, vmax=-8)
    
    # flatten images and mask to match the FAST++ fit output
    # rs = calculate_distance_to_center(shape).flatten()
    # tng_Mstar = tng_Mstar_map.flatten() # [rs <= 5*Re_pix]
    # tng_SFR = tng_SFR_map.flatten() # [rs <= 5*Re_pix]
    
    if verbose :
        print('TNG pixelized integrated mass <5 Re         = {:.3f}'.format(
            np.log10(np.sum((tng_Mstar_map.flatten())[dist <= 5*Re_pix] ))))
    
    # compute the profiles for the TNG particles, binned by pixels
    tng_Mstar_profile = np.full(20, np.nan)
    tng_SFR_profile = np.full(20, np.nan)
    for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        # if end == edges_pix[-1]:
        #     mask = (dist >= start) & (dist <= end)
        # else :
        #     mask = (dist >= start) & (dist < end)
        # tng_Mstar_profile[i] = np.sum(tng_Mstar[mask])
        # tng_SFR_profile[i] = np.sum(tng_SFR[mask])
        
        if start == 0 :
            ap = CircularAperture(center, end)
        else :
            ap = CircularAnnulus(center, start, end)
        Mstar_flux = ap.do_photometry(tng_Mstar_map)
        SFR_flux = ap.do_photometry(tng_SFR_map)
        
        tng_Mstar_profile[i] = Mstar_flux[0][0]
        tng_SFR_profile[i] = SFR_flux[0][0]
    
    # convert stellar masses to surface mass densities
    tng_p_Sigma = tng_Mstar_profile/physical_area_profile.value
    
    # convert star formation rates to surface star formation rate densities
    tng_p_Sigma_SFR = tng_SFR_profile/physical_area_profile.value
    
    return tng_p_Sigma, tng_p_Sigma_SFR, tng_p_Sigma_SFR/tng_p_Sigma

def get_fastpp_profiles(subID, shape, Re_pix, edges_pix, dist,
                        physical_area_profile, infile=None, skiprows=17) :
    
    # get the results from the FAST++ fit
    (lmass, lmass_lo, lmass_hi, lsfr, lsfr_lo, lsfr_hi,
     pixel) = load_fast_fits(299910, infile=infile, skiprows=skiprows)
    
    if False : # reconstruct the image for display purposes
        lmass_img = np.full(np.square(shape[0]), np.nan)
        lsfr_img = np.full(np.square(shape[0]), np.nan)
        for pix in np.arange(np.square(shape[0])) :
            loc = np.where(pixel == pix)[0]
            if len(loc) > 0 :
                lmass_img[pix] = lmass[loc[0]]
                lsfr_img[pix] = lsfr[loc[0]]
        plt.display_image_simple(np.reshape(lmass_img, shape), lognorm=False,
                                  vmin=5, vmax=9)
        plt.display_image_simple(np.reshape(lsfr_img, shape), lognorm=False,
                                  vmin=-5, vmax=-2)
        plt.display_image_simple(np.reshape(lsfr_img, shape) - np.reshape(lmass_img, shape),
            lognorm=False, vmin=-12, vmax=-8)
    
    if False : # flatten the distance array to 1D, and check the radial profiles
        plt.plot_scatter(dist, lmass, 'k', r'$\log{(M_{*}/{\rm M}_{\odot})}$', 'o',
            xlabel='distance from center (pixels)',
            ylabel='stellar mass per pixel', xmin=0, xmax=5*Re_pix, ymin=5)
        plt.plot_scatter(dist, lsfr, 'k', 'SFR', 'o',
            xlabel='distance from center (pixels)',
            ylabel='SFR per pixel', xmin=0, xmax=5*Re_pix, ymin=-6)
        plt.plot_scatter(dist, lsfr - lmass, 'k', 'sSFR', 'o',
            xlabel='distance from center (pixels)',
            ylabel='sSFR per pixel', xmin=0, xmax=5*Re_pix, ymin=-12, ymax=-8)
    
    '''
    # compute the profiles based on the FAST++ fitted values
    lmass_profile = np.full(20, np.nan)
    # lmass_profile_lo = np.full(20, np.nan)
    # lmass_profile_hi = np.full(20, np.nan)
    lsfr_profile = np.full(20, np.nan)
    # lsfr_profile_lo = np.full(20, np.nan)
    # lsfr_profile_hi = np.full(20, np.nan)
    # lmass_mean_profile = np.full(20, np.nan)
    # lmass_std_profile = np.full(20, np.nan)
    # lsfr_mean_profile = np.full(20, np.nan)
    # lsfr_std_profile = np.full(20, np.nan)
    for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        if end == edges_pix[-1]:
            mask = (dist >= start) & (dist <= end)
        else :
            mask = (dist >= start) & (dist < end)
        
        # compute the profiles for the FAST++ fit
        lmass_profile[i] = np.nansum(np.power(10, lmass[mask]))
        # lmass_profile_lo[i] = np.sum(np.power(10, lmass_lo[mask]))
        # lmass_profile_hi[i] = np.sum(np.power(10, lmass_hi[mask]))
        
        lsfr_profile[i] = np.nansum(np.power(10, lsfr[mask]))
        # lsfr_profile_lo[i] = np.sum(np.power(10, lsfr_lo[mask]))
        # lsfr_profile_hi[i] = np.sum(np.power(10, lsfr_hi[mask]))
        
        # compute the profiles for the FAST++ fit using averages
        # instead of sums
        # lmass_mean_profile[i] = np.mean(np.power(10, lmass[mask]))
        # lmass_std_profile[i] = np.std(np.power(10, lmass[mask]))
        
        # lsfr_mean_profile[i] = np.mean(np.power(10, lsfr[mask]))
        # lsfr_std_profile[i] = np.std(np.power(10, lsfr[mask]))
    '''
    
    lmass_profile = np.power(10, lmass)
    lsfr_profile = np.power(10, lsfr)
    
    # print('FAST++ mass profile')
    # print(np.log10(np.sum(lmass_profile)))
    
    # convert stellar masses to surface mass densities
    fast_Sigma = lmass_profile/physical_area_profile.value
    # fast_Sigma_lo = lmass_profile_lo/physical_area_profile.value
    # fast_Sigma_hi = lmass_profile_hi/physical_area_profile.value
    
    # fast_Sigma = lmass_mean_profile/pixel_area_physical.value
    # fast_Sigma_err = lmass_std_profile/np.sqrt(nPixel_profile)/pixel_area_physical.value
    
    # convert star formation rates to surface star formation rate densities
    fast_Sigma_SFR = lsfr_profile/physical_area_profile.value
    # fast_Sigma_SFR_lo = lsfr_profile_lo/physical_area_profile.value
    # fast_Sigma_SFR_hi = lsfr_profile_hi/physical_area_profile.value
    
    # fast_Sigma_SFR = lsfr_mean_profile/pixel_area_physical.value
    # fast_Sigma_SFR_err = lsfr_std_profile/np.sqrt(nPixel_profile)/pixel_area_physical.value
    
    return fast_Sigma, fast_Sigma_SFR, fast_Sigma_SFR/fast_Sigma

def load_fast_fits(subID, infile=None, skiprows=17, numBins=0, realizations=False) :
    
    # load fitted data coming out of FAST++
    # data = np.loadtxt('fits/photometry_23November2024_burst.fout',
    #                   dtype=str, skiprows=17)
    # data = np.loadtxt('fits/dpl_initial.fout', dtype=str, skiprows=19)
    data = np.loadtxt(infile, dtype=str, skiprows=skiprows)
    # ncols = data.shape[1]
    # check_val = float(data[0, 7])
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    # use = np.char.find(ids, str(subID))
    # use[use < 0] = 1
    # use = np.invert(use.astype(bool))
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, 0].astype(int)
    use = (ids == subID)
    use[np.where(use)[0][-2:]] = False # account for 1 kpc and integrated bins
    
    # check the distributions of ltau and lage, to determine better limits
    # ltau = data[:, 4].astype(float)
    # lage = data[:, 10].astype(float)
    # plt.histogram(ltau, 'log(tau/yr)', bins=30) # raw output
    # plt.histogram(lage, 'log(age/yr)', bins=30)
    # plt.histogram(np.power(10, ltau)/1e9, 'tau (Gyr)', bins=30) # in Gyr
    # plt.histogram(np.power(10, lage)/1e9, 'age (Gyr)', bins=30)
    
    # use = np.ones(data.shape[0]).astype(bool) # use all rows for pixel-by-pixel fitting
    
    if float(data[0, 2]) >= 6.4 :
        lmass = data[:, 6].astype(float)[use] # lmass
        lsfr = data[:, 14].astype(float)[use] # sfr100
    
    if float(data[0, 2]) <= 0.06 :
        lmass = data[:, 5].astype(float)[use]
        # lsfr = data[:, 6].astype(float)[use] # lsfr
        lsfr = data[:, 14].astype(float)[use] # sfr100
    
    '''
    # get the stellar mass and star formation rates
    if (ncols == 44) and (check_val == 0.02) : # delayed tau model
        lmass = data[:, 16].astype(float)[use]
        lsfr = data[:, 19].astype(float)[use] # 'lsfr'
        # lsfr = data[:, 40].astype(float)[use] # 'sfr100'
    elif (ncols == 16) : # delayed tau model with no uncertainties
        lmass = data[:, 6].astype(float)[use]
        lsfr = data[:, 7].astype(float)[use] # 'lsfr'
        # lsfr = data[:, 14].astype(float)[use] # 'sfr100'
    # elif (ncols == 16) : # fixedtimebins_small with no uncertainties
    #     lmass = data[:, 5].astype(float)[use]
    #     lsfr = data[:, 6].astype(float)[use]
    else : # hybrid model and fixed time bins model
        lmass = data[:, 13].astype(float)[use]
        lsfr = data[:, 16].astype(float)[use] # 'lsfr', same as 'sfr100'
    '''
    
    '''
    if realizations :
        # step through every annulus
        lmass_lo = np.full(numBins, np.nan)
        lmass_median = np.full(numBins, np.nan)
        lmass_hi = np.full(numBins, np.nan)
        lsfr_lo = np.full(numBins, np.nan)
        lsfr_median = np.full(numBins, np.nan)
        lsfr_hi = np.full(numBins, np.nan)
        for annulus in range(numBins) :
            # find the median +/- 1 sigma uncertainties based on all realizations
            lmass_tuple = np.percentile(lmass[annulus::numBins], [16, 50, 84])
            lsfr_tuple = np.percentile(lsfr[annulus::numBins], [16, 50, 84])
            
            # place those values into their respective arrays
            lmass_lo[annulus] = lmass_tuple[0]
            lmass_median[annulus] = lmass_tuple[1]
            lmass_hi[annulus] = lmass_tuple[2]
            
            lsfr_lo[annulus] = lsfr_tuple[0]
            lsfr_median[annulus] = lsfr_tuple[1]
            lsfr_hi[annulus] = lsfr_tuple[2]
        
        return lmass_median, lmass_lo, lmass_hi, lsfr_median, lsfr_lo, lsfr_hi
    '''
    '''
    if (ncols == 44) and (check_val == 0.02) : # delayed tau model
        lmass_lo = data[:, 17].astype(float)[use]
        lmass_hi = data[:, 18].astype(float)[use]
        
        # lsfr_lo = data[:, 41].astype(float)[use] # 'sfr100'
        # lsfr_hi = data[:, 42].astype(float)[use] # 'sfr100'
        lsfr_lo = data[:, 20].astype(float)[use] # 'lsfr'
        lsfr_hi = data[:, 21].astype(float)[use] # 'lsfr'
    elif (ncols == 16) : # fixedtimebins_small with no uncertainties
        lmass_lo = np.zeros_like(lmass)
        lmass_hi = np.zeros_like(lmass)
        
        lsfr_lo = np.zeros_like(lsfr)
        lsfr_hi = np.zeros_like(lsfr)
    else : # hybrid model and fixed time bins model
        lmass_lo = data[:, 14].astype(float)[use]
        lmass_hi = data[:, 15].astype(float)[use]
        
        lsfr_lo = data[:, 17].astype(float)[use] # 'lsfr', same as 'sfr100'
        lsfr_hi = data[:, 18].astype(float)[use] # 'lsfr', same as 'sfr100'
    '''
    lmass_lo = np.zeros_like(lmass)
    lmass_hi = np.zeros_like(lmass)
    
    lsfr_lo = np.zeros_like(lsfr)
    lsfr_hi = np.zeros_like(lsfr)
    return lmass, lmass_lo, lmass_hi, lsfr, lsfr_lo, lsfr_hi, np.arange(0, 20, 1)

def spatial_plot_info(time, snap, mpbsubID, center, Re, edges, delta_t,
                      verbose=False) :
    
    '''
    # open the TNG cutout and retrieve the relevant information
    inDir = 'F:/TNG50-1/mpb_cutouts_099/'
    cutout_file = inDir + 'cutout_{}_{}.hdf5'.format(snap, mpbsubID)
    with h5py.File(cutout_file, 'r') as hf :
        coords = hf['PartType4']['Coordinates'][:]
        ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        Mstar = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h # solMass
    
    # limit particles to those that have positive formation times
    mask = (ages > 0)
    coords, ages, Mstar = coords[mask], ages[mask], Mstar[mask]
    '''
    
    # get all particles
    (gas_masses, gas_sfrs, gas_coords, star_ages, star_gfm, star_masses,
     star_coords, star_metals) = get_rotation_input(snap, mpbsubID)
    
    if verbose :
        print('TNG integrated mass from star particles     = {:.3f}'.format(
            np.log10(np.sum(star_masses))))
    
    # determine the rotation matrix
    # rot = rotation_matrix_from_MoI_tensor(calculate_MoI_tensor(
    #     gas_masses, gas_sfrs, gas_coords, star_ages, star_masses, star_coords,
    #     Re, center))
    
    # reproject the coordinates using the face-on projection
    # dx, dy, dz = np.matmul(np.asarray(rot['face-on']), (star_coords-center).T)
    
    # don't project using face-on version
    dx, dy, dz = (star_coords - center).T
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (star_ages >= limit)
    
    sf_masses = star_gfm[mask]
    sf_dx = dx[mask]
    sf_dy = dy[mask]
    # sf_dz = dz[mask]
    
    # create 2D histograms of the particles and SF particles
    hh, _, _ = np.histogram2d(dx/Re, dy/Re, bins=(edges, edges),
                              weights=star_masses)
    hh = hh.T
    
    hh_sf, _, _ = np.histogram2d(sf_dx/Re, sf_dy/Re, bins=(edges, edges),
                                 weights=sf_masses)
    hh_sf = hh_sf.T
    
    return hh, hh_sf/delta_t.to(u.yr).value

def save_all_radial_profile_comparison_plots() :
    # uses an earlier version of the photometry which lacks the CASTOR
    # UV^L and u^S filters
    
    # define the subIDs and the relevant snapshot that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, snaps, Re = table['subID'].data, table['snapshot'].data, table['Re'].data
    
    # mask subIDs 14, 43, 514274, 656524, 657979, 680429, which didn't have
    mask = np.full(278, True) # successful SKIRT runs when initally processed
    mask[6] = False
    mask[24] = False
    mask[210] = False
    mask[262] = False
    mask[264] = False
    mask[270] = False
    subIDs = subIDs[mask]
    snaps = snaps[mask]
    Re = Re[mask]
    
    # get basic info about the quenched galaxy
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        logMfinal = hf['logM'][:, -1]
        quenched = hf['quenched'][:]
    
    # get the quenching mechanisms
    with h5py.File('D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    mechs = mechs[(logMfinal >= 9.5) & quenched] # mask to the massive quench pop.
    mechs = mechs[mask] # mask out the subIDs that didn't have successful
                        # initial SKIRT runs
    
    # get the stellar mass for future sorting, using the same masks as above
    logM = logMfinal[(logMfinal >= 9.5) & quenched]
    logM = logM[mask]
    
    # sort the galaxies according to stellar mass
    sort = np.argsort(logM) # np.argsort(Re)
    subIDs = subIDs[sort]
    snaps = snaps[sort]
    mechs = mechs[sort]
    
    pop, label = 1, 'inside-out'
    # pop, label = 3, 'outside-in'
    
    # for subID, snap in zip(subIDs[mechs == pop], snaps[mechs == pop]) :
    #     compare_fits_to_tng(subID, snap)
    
    from pypdf import PdfWriter
    merger = PdfWriter()
    for subID in subIDs[mechs == pop] :
        merger.append('fits/radial_profiles_dtt_extended/subID_{}_dtt.pdf'.format(subID))
    merger.write('radial-profiles_dtt_extended_{}_Re-sort.pdf'.format(label))
    merger.close()
    
    return

def check_fast_fits() :
    
    # get basic info about the quenched galaxy
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        logMfinal = hf['logM'][:, -1]
        quenched = hf['quenched'][:]
    
    # get the quenching mechanisms
    with h5py.File('D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    
    # get the subIDs
    subIDs = Table.read('tools/subIDs.fits')['subID'].data
    # mask subIDs 14, 43, 514274, 656524, 657979, 680429, which didn't have
    mask = np.full(278, True) # successful SKIRT runs when initally processed
    mask[6] = False
    mask[24] = False
    mask[210] = False
    mask[262] = False
    mask[264] = False
    mask[270] = False
    subIDs = subIDs[mask]
    mechs = mechs[(logMfinal >= 9.5) & quenched]
    mechs = mechs[mask]    
    '''
    for subID in subIDs :
        xs = []
        ys = []
        labels = []
        colors = []
        markers = []
        styles = []
        alphas = []
        mins = []
        maxs = []
        for i in range(20) :
            wl, fl = np.loadtxt('best_fits/photometry_22November2024_{}_bin_{}.fit'.format(subID, i), unpack=True)
            wl_phot, _, fl_phot, _ = np.loadtxt(
                'best_fits/photometry_22November2024_{}_bin_{}.input_res.fit'.format(subID, i), unpack=True)
            xs.append(wl)
            xs.append(wl_phot)
            ys.append(fl + 20 - i)
            ys.append(fl_phot + 20 - i)
            if i in [0, 9, 10, 19] :
                labels.append('bin {}'.format(i))
            else :
                labels.append('')
            labels.append('')
            if i < 10 :
                colors.append(cm.viridis(i/9))
                colors.append(cm.viridis(i/9))
                alphas.append((i/9 + 1)/2) # for the model spectra
            if i >= 10 :
                colors.append(cm.plasma((i - 10)/9))
                colors.append(cm.plasma((i - 10)/9))
                alphas.append(((i - 10)/9 + 1)/2) # for the model spectra
            alphas.append(1.0) # for the observed fluxes
            markers.append('')
            markers.append('o')
            if i % 2 == 0 :
                styles.append('-')
            else :
                styles.append('--')
            styles.append('')
            mins.append(np.min((fl + 20 - i)[(wl >= 2000) & (wl <= 2e4)]))
            mins.append(np.min(fl_phot + 20 - i))
            maxs.append(np.max((fl + 20 - i)[(wl >= 2000) & (wl <= 2e4)]))
            maxs.append(np.max(fl_phot + 20 - i))
        
        plt.plot_simple_multi(xs[:20], ys[:20], labels[:20], colors[:20], markers[:20], styles[:20], alphas[:20],
            xlabel=r'wavelength (A)',
            ylabel=r'scaled flux ($\times 10^{-19}~{\rm erg s}^{-1}~{\rm cm}^{-2}~{A}^{-1})$',
            title='subID {}'.format(subID),
            xmin=2000, xmax=2e4, ymin=np.nanmin(mins[:20])/1.1,
            ymax=np.nanmax(maxs[:20])*1.1, scale='log',
            figsizewidth=19, figsizeheight=9.5,
            outfile='fits/fit_plots/subID_{}_bins_0-9.pdf'.format(subID), save=True)
        plt.plot_simple_multi(xs[20:], ys[20:], labels[20:], colors[20:], markers[20:], styles[20:], alphas[20:],
            xlabel=r'wavelength (A)',
            ylabel=r'scaled flux ($\times 10^{-19}~{\rm erg s}^{-1}~{\rm cm}^{-2}~{A}^{-1})$',
            title='subID {}'.format(subID),
            xmin=2000, xmax=2e4, ymin=np.nanmin(mins[20:])/1.1,
            ymax=np.nanmax(maxs[20:])*1.1, scale='log',
            figsizewidth=19, figsizeheight=9.5,
            outfile='fits/fit_plots/subID_{}_bins_10-19.pdf'.format(subID), save=True)
    '''
    logM = logMfinal[(logMfinal >= 9.5) & quenched]
    logM = logM[mask]
    sort = np.argsort(logM)
    
    subIDs = subIDs[sort]
    mechs = mechs[sort]
    
    # from pypdf import PdfWriter
    # merger = PdfWriter()
    # for subID in subIDs[mechs == 3] :
    #     merger.append('fits/fit_plots/subID_{}_bins_0-9.pdf'.format(subID))
    #     merger.append('fits/fit_plots/subID_{}_bins_10-19.pdf'.format(subID))
    # merger.write('fits_delaytau_outside-in.pdf')
    # merger.close()
    
    return

# compare_fits_to_tng(324125, 76)

compare_fits_to_tng(63871, 40)

'''
# test the method of Suess+ 2019a
fout = np.loadtxt('Suess+2019a_method/photometry_23November2024_subID_198186.fout',
                  dtype=str, skiprows=18)
avs = fout[:, 4].astype(float)
lages = fout[:, 3].astype(float)
ltaus = fout[:, 9].astype(float)
rs = fout[:, 8].astype(float)

import itertools

tt = Table.read('Suess+2019a_method/chi2.grid.fits')[0]
chi2 = tt['CHI2']
avGrid = tt['AV']
ageGrid = tt['LAGE']
tauGrid = tt['LOG_TAU']
rGrid = tt['R']

massGrid = tt['LMASS']
sfrGrid = tt['LSFR']

# find the minimum chi2 value for the integrated aperture
# print(np.min(chi2[:, 21])/3)

# check that the correct masses can be found in the chi2 grid
# for i in range(22) :
#     loc = np.argmin(chi2[:, i])
#     print(np.round(np.log10(massGrid[loc, i]), 2))

# get the existing positions of the annuli, as in Suess+ (2019a)
positions = np.array([avs[:20], lages[:20], ltaus[:20]]).T # rs[:20]
# positions = np.full((20, 3), -1.0)
# for i in range(20) :
#     loc = np.argmin(chi2[:, i])
#     positions[i] = [avGrid[loc, i], ageGrid[loc, i], tauGrid[loc, i]]
# positions = np.round(positions, 2) # account for floating point error


# get the integrated chi2 according to Suess et al. (2019a)
inDir = 'Suess+2019a_method/best_fits_free/'
intFile = 'photometry_23November2024_subID_198186_198186_bin_int.input_res.fit'
wl, modelFlux = np.loadtxt(inDir + intFile)[:, :2].T

# attach units to the output fluxes
wl *= u.Angstrom
modelFlux = modelFlux*1e-19*u.erg/u.s/u.cm/u.cm/u.Angstrom

# convert the model flux to janskies
modelFlux = (modelFlux*wl*wl/c.c).to(u.Jy).value

catFile = 'Suess+2019a_method/photometry_23November2024_subID_198186.cat'
cat = np.loadtxt(catFile)[21, 1:] # read the integrated values
catFlux = np.array([cat[i] for i in [0, 2, 4, 6, 8, 10, 12]])
catErr = np.array([cat[i + 1] for i in [0, 2, 4, 6, 8, 10, 12]])

intChi = np.sum(np.square((catFlux - modelFlux)/catErr))

def get_chisq(posList) :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # sum of FAST++ chi2 for each annulus
    chis = np.full(20, -1.)
    for i in range(20) :
        pos = posList[i]
        mask = ((np.abs(avGrid[:, i] - pos[0]) <= 0.001) &
                (np.abs(ageGrid[:, i] - pos[1]) <= 0.001) &
                (np.abs(tauGrid[:, i] - pos[2]) <= 0.001)) #&
                # (np.abs(rGrid[:, i] - pos[3]) <= 0.001))
        chis[i] = chi2[:, i][mask][0]
    fastChi = np.sum(chis)
    
    # reduce chi2 and return it, also using the chi2 for all integrated filters
    chi_red = fastChi/(7 - 4) + intChi/(7 - 4)
    
    return chi_red

def adjustPos(positions, ann) :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # make a list of the possible values our Av, age, tau, rr can now take,
    # moving up to +/- 3 steps in any variable
    newPos = np.array([tuple(map(sum, zip(positions[ann], i)))
        for i in itertools.product([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                                   repeat=3)])
    
    # initialize housekeeping variables
    newChi = np.zeros(len(newPos)) + 1e10
    tmpPos = np.array([i for i in positions])
    changed = False
    
    # for each possible new position, calculate the chi2
    for posIdx, pos in enumerate(newPos) :
        
        # make sure this position is actually allowed (e.g., doesn't hit the
        # edge of the grid)
        if ((0 <= pos[0] <= 0.1) and (9 <= pos[1] <= 9.4) and  # 784 valid
            (8.1 <= pos[2] <= 8.5)) : # and (0 <= pos[3] <= 1)) : # possible new
                                                             # positions
            # calculate chi2 for new position
            tmpPos[ann] = pos
            newChi[posIdx] = get_chisq(tmpPos)
    
    # locs = np.where(newChi == np.min(newChi))[0]
    # for loc in locs :
        # print(newPos[loc])
    
    # get lowest chi2 of new possible positions (if it didn't fail)
    if np.min(newChi) < 1e10 :
        # see if it changed
        if list(newPos[np.argmin(newChi)]) != list(positions[ann]) :
            changed = True
        
        # set new positions
        positions[ann] = newPos[np.argmin(newChi)]
    
    return positions, changed

def find_bestPos() :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # set the maximum number of iterations
    maxIter = 500
    
    # initialize the counters for how many times we've updated, and how long
    # it's been stable at a chi2 minimum
    it, lastChanged = 0, 0
    
    # while we haven't coverged, update the positions
    while (it < maxIter and lastChanged < 20*3) :
        # find new positions
        ann = it % 20
        print(it, lastChanged, ann)
        bestPos, changed = adjustPos(positions, ann)
        
        # if unchanged, increment counter
        if not changed :
            lastChanged += 1
        it += 1
    
    print(positions)
    print(bestPos)
    
    return

# find_bestPos()
'''

"""
from sfh2sed import calzetti2000, dtb_from_fit, madau1995, sfh2sed_fastpp

# compare the SEDs from the bestfit results
fout = np.loadtxt('Suess+2019a_method/photometry_23November2024_subID_198186.fout',
                  dtype=str, skiprows=18)
avs = fout[:, 4].astype(float)
lages = fout[:, 3].astype(float)
ltaus = fout[:, 9].astype(float)
lmasses = fout[:, 5].astype(float)
rs = fout[:, 8].astype(float)

# read the model spectra from Bruzual & Charlot (2003) for a Chabrier (2003)
# initial mass function with solar metallicity
table = Table.read('tools/bc03_lr_ch_z02.ised_ASCII.fits')[0]
ages = table['AGE']/1e6     # (221), [Myr]
masses = table['MASS']      # (221), [Msol]
waves = table['LAMBDA']/1e4 # (1221), [micron]
seds = table['SED']         # (221, 1221), [Lsol AA^-1]

# determine the conversion factor to go from SED units of Lsol AA^-1 to
# 10^-19 ergs s^-1 cm^-2 AA^-1
lum = (1*u.solLum/u.AA).to(u.erg/u.s/u.AA)
d_l = cosmo.luminosity_distance(0.5).to(u.cm)
lum2fl = 1e19*lum/(4*np.pi*(1 + 0.5)*np.square(d_l))
lum2fl = lum2fl.value

# mask out the 1 kpc aperture
mask = np.full(22, True)
mask[20] = False

# get the Calzetti dust correction as a function of wavelength
calzetti = calzetti2000(waves)

# set the observation time and the time array
# tobs = 8622.39 # matches with FAST++'s output, but isn't same as astropy
tobs = cosmo.age(0.5).to(u.Myr).value
ts = np.array(np.arange(1, int(tobs) + 1).tolist() + [tobs])

ys_bestfit = []
for av, ltau, lage, rr, lmass in zip(avs[mask], ltaus[mask], lages[mask],
                                     rs[mask], lmasses[mask]) :
    # create the SFH from the bestfit results
    sfh = dtb_from_fit(ts, ltau, lage, rr, norm=np.power(10, lmass)) # [Msol/yr]
    sfh *= 1e6 # [Msol/Myr]
    
    # get the SED from the bestfit results
    sed = sfh2sed_fastpp(ages, masses, seds, tobs, ts, sfh) # [Lsol AA^-1]
    # sed *= np.power(10, -0.4*Av*calzetti(waves)) # correct for dust
    sed *= madau1995(0.5, waves) # correct for IGM absorption
    
    # convert the luminosity units to flux units
    sed *= lum2fl # [10^-19 ergs s^-1 cm^-2 AA^-1]
    
    ys_bestfit.append(sed)
xs_bestfit = [waves*1.5]*21
colors = cm.viridis(np.linspace(0, 1, 20)).tolist() + [[0., 0., 0., 0.]]

# integrated_from_bestfit = ys_bestfit[-1]
# integrated_from_sum_of_bestfits = np.sum(ys_bestfit[:21], axis=0)
# ratio = integrated_from_sum_of_bestfits/integrated_from_bestfit
# plt.histogram(ratio, 'ratio', bins=30)

plt.plot_simple_multi(xs_bestfit, ys_bestfit,
    ['from bestfit params'] + ['']*19 + ['integrated'], colors, ['']*21, ['-']*21,
    [0.3]*20 + [1],
    xlabel='wavelength (micron)',
    ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
    title='subID 198186',
    scale='log', xmin=0.1368, xmax=4, ymin=1e-03, ymax=500)

# plt.plot_simple_multi([waves*1.5, waves*1.5], [ys_bestfit[-1], np.sum(ys_bestfit[:21], axis=0)],
#     ['integrated', 'sum of bestfit param annuli'], ['k', 'b'], ['', ''], ['-', '-'], [1, 1],
#     xlabel='wavelength (micron)',
#     ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
#     scale='log', xmin=0.1368, xmax=4, ymin=1e-03, ymax=500)

inDir = 'best_fits_free/'
xs_direct = []
ys_direct = []
for i in range(20) :
    file = 'photometry_23November2024_subID_198186_198186_bin_{}.fit'.format(i)
    wl, dat = np.loadtxt(inDir + file).T
    xs_direct.append(wl/1e4) # micron
    ys_direct.append(dat)

intFile = 'photometry_23November2024_subID_198186_198186_bin_int.fit'
wl, dat = np.loadtxt(inDir + intFile).T
xs_direct.append(wl/1e4) # micron
ys_direct.append(dat)

# integrated_from_output = dat
# integrated_from_sum_of_outputs = np.sum(ys_direct[:21], axis=0)
# ratio = integrated_from_sum_of_outputs/integrated_from_output
# plt.histogram(ratio, 'ratio', bins=30)

plt.plot_simple_multi(xs_direct, ys_direct,
    ['from FAST++ output'] + ['']*19 + ['integrated'], colors, ['']*21, ['-']*21,
    [0.3]*20 + [1],
    xlabel='wavelength (micron)',
    ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
    title='subID 198186',
    scale='log', xmin=0.1368, xmax=4, ymin=1e-03, ymax=500)

# plt.plot_simple_multi([wl/1e4, wl/1e4], [ys_direct[-1], np.sum(ys_direct[:21], axis=0)],
#     ['integrated', 'sum of FAST++ outputs'], ['k', 'b'], ['', ''], ['-', '-'], [1, 1],
#     xlabel='wavelength (micron)',
#     ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
#     scale='log', xmin=0.1368, xmax=4, ymin=1e-03, ymax=500)
"""

def get_bestfit(binNum) :
    
    inDir = 'extra_metals_dtt_subID_198186/' # 'Suess+2019a_method/'
    file = 'photometry_23November2024_subID_198186.fout'
    fout = np.loadtxt(inDir + file, dtype=str, skiprows=18)
    
    # get the required bestfit values
    metal, Av = fout[binNum, 2].astype(float), fout[binNum, 4].astype(float)
    lage, ltau = fout[binNum, 3].astype(float), fout[binNum, 9].astype(float)
    lmass, rr = fout[binNum, 5].astype(float), fout[binNum, 8].astype(float)
    
    return metal, Av, lage, ltau, lmass, rr

def get_sed_normalization(binNum) :
    
    inDir = 'extra_metals_dtt_subID_198186/best_fits/' # 'Suess+2019a_method/best_fits_free/'
    file = 'photometry_23November2024_subID_198186_198186_bin_{}.sfh'.format(binNum)
    ts, sfh = np.loadtxt(inDir + file).T # [Myr], [Msol/yr]
    
    diff = np.array([1e6] + np.diff(ts).tolist()) # spacing between values
    sfh = diff*sfh # [Msol/Myr]
    
    return np.sum(sfh)

def get_photometry(binNum, wphot) :
    
    inDir = 'extra_metals_dtt_subID_198186/'
    file = 'photometry_23November2024_subID_198186.cat'
    # ignore the id and redshift
    cat = np.loadtxt(inDir + file, dtype=str)[binNum][1:-1] # [Jy]
    
    # get the observed photometry and the uncertainties, in janskies
    phot = cat[0::2].astype(float)
    phot_e = cat[1::2].astype(float)
    
    # convert the catalog fluxes into the same units as used by FAST++
    flx = 1e19*(phot*u.Jy)*c.c/np.square(wphot*u.um)
    flx_e = 1e19*(phot_e*u.Jy)*c.c/np.square(wphot*u.um)
    
    flx = flx.to(u.erg/u.s/u.cm/u.cm/u.AA).value
    flx_e = flx_e.to(u.erg/u.s/u.cm/u.cm/u.AA).value
    
    return flx, flx_e

def compare_output_and_constructed_sed(binNum, zz=0.5) :
    
    # get the time array for the SFH, and the observation time
    tobs, ts = get_times(zz) # Myr
    
    # get the catalog bestfit results as output by FAST++
    metal, Av, lage, ltau, lmass, rr = get_bestfit(binNum)
    
    # create the SFH from the bestfit results
    sfh = 1e6*dtt_from_fit(ts, ltau, lage, rr, norm=get_sed_normalization(binNum)) # [Msol/Myr]
    
    # determine the difference between the catalog mass and the total formed mass
    # print(np.log10(get_sed_normalization(binNum)) - lmass) # 0.27559776451409235
    
    # construct the SED from the bestfit results
    sed = sfh2sed_fastpp(metal, tobs, ts, sfh) # [Lsol AA^-1]
    
    # get the library wavelengths for the constructed SED
    waves = get_bc03_waves(metal)
    
    # correct the SED for dust using the Calzetti dust law
    sed *= np.power(10, -0.4*Av*calzetti2000(waves))
    
    # correct for IGM absorption
    # sed *= madau1995(zz, waves) # correct for IGM absorption
    
    # convert the luminosity units to flux units
    sed *= get_lum2fl(zz) # [10^-19 ergs s^-1 cm^-2 AA^-1]
    
    # redshift the wavelengths for plotting
    waves *= 1 + zz
    
    # define the filter transmission curves used in the fitting
    filters = ['castor_uv', 'castor_u', 'castor_g',
               'roman_f106', 'roman_f129', 'roman_f158', 'roman_f184']
    
    # get the model fluxes
    models = get_model_fluxes(filters, waves, sed)
    
    # get the model photometric wavelengths
    pivots = get_filter_waves(filters) # not actually pivot wavelengths
    
    # get the observed photometry
    phot, phot_e = get_photometry(binNum, pivots)
    
    # calculate the chi2 for the model
    chi2 = calculate_chi2(pivots, models, phot, phot_e)
    print(chi2)
    
    # plot the bestfit constructed SED
    plt.plot_sed(waves[waves > 0.137], sed[waves > 0.137], pivots, models, phot, phot_e,
        # title='subID 198186 bin {}'.format(binNum),
        outfile='Suess+2019a_method/subID_198186_bin_0_SEDs.pdf', save=False)
    
    return

# print(get_filter_waves(['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g']))
# compare_output_and_constructed_sed(0, zz=0.5)

def compare_multiple_seds() :
    
    binNum = 0
    
    # get the SED as output directly from FAST++
    inDir = 'Suess+2019a_method/best_fits_free/'
    file = 'photometry_23November2024_subID_198186_198186_bin_{}.fit'.format(binNum)
    wl, sed_direct = np.loadtxt(inDir + file).T # [10^-19 ergs s^-1 cm^-2 AA^-1]
    wl /= 1e4 # [micron]
    
    # get the model fluxes and observed fluxes as output directly from FAST++
    file = 'photometry_23November2024_subID_198186_198186_bin_{}.input_res.fit'.format(binNum)
    wl_phot, fl_model, fl_phot, fl_phot_e = np.loadtxt(inDir + file).T # [10^-19 ergs s^-1 cm^-2 AA^-1]
    # conversion from fl_phot to cat_phot: cat_phot = fl_phot*wavelength^2/c.c
    wl_phot /= 1e4 # [micron]
    
    # mask out zeros to clean up the plot
    sed_lum = sed_lum[ww >= 0.137] # [sed_lum > 0]
    ww = ww[ww >= 0.137] # [sed_lum > 0]
    wz = wz[sed_fl > 0]
    sed_fl = sed_fl[sed_fl > 0]
    
    # read the SEDs as output directly from FAST++ using fast++-sfh2sed
    # (in combination with the output SFHs from FAST++)
    inDir = 'Suess+2019a_method/'
    file = 'subID_198186_bin_{}.sed'.format(binNum) # included tobs=8622.39
    ww, sed_lum = np.loadtxt(inDir + file).T        # via CLI
    ww /= 1e4
    sed_lum *= np.power(10, -0.4*Av*calzetti2000(ww))
    # sed_lum *= madau1995(zz, ww)
    ww *= 1 + zz
    sed_lum *= lum2fl
    
    file = 'subID_198186_bin_{}.sedz'.format(binNum) # included tobs=8622.39
    wz, sed_fl = np.loadtxt(inDir + file).T          # and z=0.5 via CLI
    wz /= 1e4
    sed_fl *= np.power(10, -0.4*Av*calzetti2000(wz))
    # sed_fl *= madau1995(zz, wz/1.5)
    
    # compare bestfit constructed SED from FAST++ against the direct output
    plt.plot_sed([waves, wl, wl_phot, wl_phot, ww, wz],
        [sed_constructed, sed_direct, fl_model, fl_phot, sed_lum, sed_fl],
        ['from SFH parameters', 'output SED',
         'model flux', 'observed flux',
         r'fast++-sfh2sed ($L_{\odot}~{\rm \AA}^{-1}$)',
         r'fast++-sfh2sed ($F_{\lambda,~z=0.5}$)'],
        ['k', 'orange', 'k', 'cyan', 'm', 'b'],
        ['', '', 's', 'o', '', ''],
        [':', '-', '', '', '--', '-'],
        [1, 0.6, 1, 1, 0.5, 0.3],
        ms=12, # markersize
        xlabel='wavelength (micron)',
        ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
        # title='subID 198186 bin {}'.format(binNum),
        figsizeheight=9, figsizewidth=12,
        scale='log', loc=4,
        xmin=0.13, xmax=3, ymin=0.1, ymax=10,
        # xmin=0.01, xmax=10, ymin=0.01, ymax=10, # see more of the entire SED
        save=False, outfile='Suess+2019a_method/subID_198186_bin_0_SEDs.pdf')
    
    '''
    # check the offset between the three coincident SEDs and the direct output
    ratio = sed_constructed[97:]/sed_direct
    plt.plot_simple_dumb(waves[97:], ratio, xmin=0.13)
    # `ratio` looks as if it's made up of 3 distinct curves, with bounds of
    # [0.13725, 0.18225], [0.18375, 0.9435], [0.9465, 4.6725]
    
    print(waves)
    
    # print(np.array([[0.13725, 0.18225], [0.18375, 0.9435], [0.9465, 4.6725]])/1.5)
    
    # in un-redshifted wavelength space
    breaks = [[0.0915, 0.1215],
              [0.1225, 0.629 ],
              [0.631,  3.115 ]]
    '''
    
    return

def compare_final_SMHs_to_catalog() :
    
    cat = np.loadtxt('fits/photometry_23November2024_integrated.fout',
                      dtype=str, skiprows=17)
    
    Mstar = []
    for ID in cat[:, 0] :
        _, Mstar_t = np.loadtxt('best_fits_and_SFHwithMass/' +
            'photometry_23November2024_integrated_{}.sfh'.format(ID), unpack=True)
        Mstar.append(Mstar_t[-1])
    
    plt.plot_scatter(cat[:, 6].astype(float), np.log10(Mstar), 'k', '', 'o',
        xmin=9.1, xmax=11.4, ymin=9.1, ymax=11.4, xlabel=r'lmass$_{\rm catalog}$',
        ylabel=r'final M$_{*} (t)$ from .sfh', figsizeheight=4, figsizewidth=6)
    
    return

def compare_integrated_SFHs_to_catalog() :
    
    cat = np.loadtxt('fits/photometry_23November2024_integrated.fout',
                      dtype=str, skiprows=17)
    
    integrated = []
    for ID in cat[:, 0] :
        _, sfr_t = np.loadtxt('best_fits_and_SFHwithSFR/' +
            'photometry_23November2024_integrated_{}.sfh'.format(ID), unpack=True)
        integrated.append(np.log10(np.sum(sfr_t)*1e6))
    
    plt.plot_scatter(cat[:, 6].astype(float), integrated, 'k', '', 'o',
        xmin=9.1, xmax=11.4, ymin=9.1, ymax=11.4, xlabel=r'lmass$_{\rm catalog}$',
        ylabel='integrated from .sfh', figsizeheight=4, figsizewidth=6)
    
    return

# import sfhs
# sfhs.compute_sfh_per_bin(198186)

# for binNum in range(20)[:1] :
    
#     dtt_inDir = 'extra_metals_dtt/best_fits/'
#     dpl_inDir = 'extra_metals_dpl/best_fits/'
#     ftb_inDir = 'extra_metals_ftb/best_fits/'
#     infile = 'photometry_23November2024_subID_198186_198186_bin_'
    
#     # SFHs
#     dtt_t, dtt_sfh = np.loadtxt(dtt_inDir + infile + '{}.sfh'.format(binNum), unpack=True)
#     dpl_t, dpl_sfh = np.loadtxt(dpl_inDir + infile + '{}.sfh'.format(binNum), unpack=True)
#     ftb_t, ftb_sfh = np.loadtxt(ftb_inDir + infile + '{}.sfh'.format(binNum), unpack=True)
#     tobs = np.max(dtt_t)
#     ts = np.array([tobs - dtt_t, tobs - dpl_t, tobs - ftb_t])/1e9
#     sfhs = [dtt_sfh, dpl_sfh, ftb_sfh]
#     plt.plot_simple_multi(ts, sfhs,
#         ['delayed-tau + trunc', 'double power law', 'fixed time bins'],
#         ['k', 'r', 'b'], ['', '', ''], ['-', '-', '-'], [1, 1, 1],
#         title='bin {}'.format(binNum), scale='log',
#         xmin=0.01, xmax=10, ymin=0.001, ymax=10, loc=2,
#         xlabel=r'$t_{\rm lookback}$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    # SEDs
    # dtt_w, dtt_sed = np.loadtxt(dtt_inDir + infile + '{}.fit'.format(binNum), unpack=True)
    # dpl_w, dpl_sed = np.loadtxt(dpl_inDir + infile + '{}.fit'.format(binNum), unpack=True)
    # ftb_w, ftb_sed = np.loadtxt(ftb_inDir + infile + '{}.fit'.format(binNum), unpack=True)
    # obs_w, _, obs_flx, _ = np.loadtxt(dtt_inDir + infile + '{}.input_res.fit'.format(binNum), unpack=True)
    # plt.plot_simple_multi([dtt_w/1e4, dpl_w/1e4, ftb_w/1e4, obs_w/1e4],
    #     [dtt_sed, dpl_sed, ftb_sed, obs_flx],
    #     ['delayed-tau + trunc', 'double power law', 'fixed time bins', 'observed photometry'],
    #     ['k', 'r', 'b', 'k'], ['', '', '', 'o'], ['-', '--', ':', ''],
    #     [0.5, 0.7, 0.7, 1], scale='log', ms=10,
    #     xmin=0.13, xmax=3, ymin=np.min(obs_flx[obs_flx > 0])/3, ymax=np.max(obs_flx)*2,
    #     title='bin {}'.format(binNum),
    #     xlabel='wavelength (micron)',
    #     ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)')


# for binNum in range(10) :
#     dtt_inDir = 'subID_324125/best_fits/'
#     infile = 'photometry_23November2024_subID_324125_324125_bin_'
    
#     # SEDs
#     dtt_w, dtt_sed = np.loadtxt(dtt_inDir + infile + '{}.fit'.format(binNum), unpack=True)
#     obs_w, _, obs_flx, _ = np.loadtxt(dtt_inDir + infile + '{}.input_res.fit'.format(binNum), unpack=True)
#     plt.plot_simple_multi([dtt_w/1e4, obs_w/1e4],
#         [dtt_sed, obs_flx],
#         ['delayed-tau + trunc', 'observed photometry'],
#         ['k', 'k'], ['', 'o'], ['-', ''],
#         [0.3, 1], scale='log', ms=10,
#         xmin=0.13, xmax=3, ymin=np.min(obs_flx[obs_flx > 0])/3, ymax=np.max(obs_flx)*2,
#         title='bin {}'.format(binNum),
#         xlabel='wavelength (micron)',
#         ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)')


