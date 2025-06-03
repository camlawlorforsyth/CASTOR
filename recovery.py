
import os
import numpy as np

# import astropy.constants as c
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip
from astropy.table import Table
import astropy.units as u
# from astropy.utils.exceptions import AstropyUserWarning
import h5py
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.profiles import CurveOfGrowth, RadialProfile
from photutils.segmentation import detect_threshold, detect_sources
# from scipy.optimize import curve_fit # uses non-linear least squares
# import statmorph

from core import load_massive_galaxy_sample#, open_cutout
# from fitting import get_fastpp_profiles, get_tng_profiles
import plotting as plt
from photometry import determine_castor_snr_map, determine_roman_snr_map

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def create_radial_profile(image, Re_pix, center, pixel_area_physical,
                          Nannuli=20, surfacedensity=True) :
    
    # get the edges of the circular annuli in units of pixels for masking,
    # along with the bin centers for plotting
    edges_pix = np.linspace(0, 5, Nannuli+1)*Re_pix # [Re]
    # radial_bin_centers = edges_pix[:-1] + np.diff(edges_pix)/2 # [Re]
    
    profile = np.full(Nannuli, -1.0)
    nPixels = np.full(Nannuli, -1.0)
    for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        if start == 0 :
            ap = CircularAperture(center, end)
        else :
            ap = CircularAnnulus(center, start, end)
        profile[i] = ap.do_photometry(image)[0][0]
        nPixels[i] = ap.area # the pixel areas per annulus
    physical_areas = pixel_area_physical*nPixels # [kpc2]
    
    # plt.plot_simple_dumb(radial_bin_centers/Re_pix,
    #     np.log10(profile/physical_areas.value), xmin=0, xmax=5)
    
    if surfacedensity :
        profile = profile/physical_areas.value
    
    return profile



def mass_to_light_ratio_circular_annuli(model_redshift=0.5) :
    
    outfile = 'tools/comparisons_z_{:03}.hdf5'.format(
       str(model_redshift).replace('.', ''))
    with h5py.File(outfile, 'r') as hf :
        # tng_logM = hf['tng_logM'][:]
        # tng_SFR = hf['tng_SFR'][:]
        # tng_metal = hf['tng_metal'][:]
        # tng_dust = hf['tng_dust'][:]
        fit_logM = hf['fit_logM'][:]
        fit_SFR = hf['fit_SFR'][:]
        # fit_metal = hf['fit_metal'][:]
        # fit_dust = hf['fit_dust'][:]
    
    # get the photometry file that was passed into FAST++
    photometry = np.loadtxt('photometry/circular_annuli/photometry_2April2025.cat',
                            dtype=str, skiprows=1)[:, 1:-1].astype(float)
    mask = np.full(22, True)
    mask[-2], mask[-1] = False, False
    photometry = photometry[np.array(list(mask)*212)] # select the correct rows
    snrs = np.array([photometry[:, 0]/photometry[:, 1],
                     photometry[:, 2]/photometry[:, 3],
                     photometry[:, 4]/photometry[:, 5],
                     photometry[:, 6]/photometry[:, 7],
                     photometry[:, 8]/photometry[:, 9],
                     photometry[:, 10]/photometry[:, 11],
                     photometry[:, 12]/photometry[:, 13],
                     photometry[:, 14]/photometry[:, 15],
                     photometry[:, 16]/photometry[:, 17]]).T
    mask = np.all(snrs > 10, axis=1) # mask to the UV+opt+NIR high SNR population
    fit_logM = fit_logM[mask]
    # y_flux = photometry[:, 10][mask]
    # j_flux = photometry[:, 12][mask]
    h_flux = photometry[:, 14][mask]
    # k_flux = photometry[:, 16][mask]
    
    # check the correlation between CASTOR g-band SNR and average Roman SNR
    # plt.plot_scatter(snrs[:, 4], np.mean(snrs[:, 5:], axis=1), 'k', '', 'o',
    #                  scale='log', xmin=1, xmax=1000, ymin=1, ymax=1000)
    
    
    # determine the NIR color which has the strongest Pearson correlation with
    # fitted stellar mass
    # from scipy.stats import pearsonr
    # p = pearsonr(-2.5*np.log10(y_flux/j_flux), fit_logM) # R = 0.647
    # p = pearsonr(-2.5*np.log10(y_flux/h_flux), fit_logM) # R = 0.696
    # p = pearsonr(-2.5*np.log10(y_flux/k_flux), fit_logM) # R = 0.706, best correlation for colours
    # p = pearsonr(np.log10(y_flux), fit_logM) # R = 0.883
    # p = pearsonr(np.log10(j_flux), fit_logM) # R = 0.919
    # p = pearsonr(np.log10(h_flux), fit_logM) # R = 0.937, second best correlation for bands, R = 0.939 after sigma clipping
    # p = pearsonr(np.log10(k_flux), fit_logM) # R = 0.942, best correlation for band fluxes, R = 0.944 after sigma clipping
    
    # adapted from
    # https://docs.astropy.org/en/latest/modeling/example-fitting-line.html
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
        sigma_clip, niter=1, sigma=3) # stable after 1 iteration
    fitted_line, mask = fitter(models.Linear1D(), np.log10(h_flux), fit_logM)
    # print(pearsonr(np.log10(h_flux)[~mask], fit_logM[~mask]).statistic)
    # print(fitted_line.parameters)
    # plt.plot_simple_dumb(np.log10(h_flux), fit_logM)
    plt.plot_simple_dumb(np.log10(h_flux)[~mask], fit_logM[~mask])
    
    
    # plot fit logM against NIR flux
    # temp = np.linspace(-6.5, -4.5, 1000)
    # temp_x = np.log10(h_flux) #-2.5*np.log10(y_flux/k_flux)
    # temp_y = fit_logM
    # good = np.isfinite(temp_x) & np.isfinite(temp_y)
    # popt, _ = curve_fit(linear, temp_x[good], temp_y[good]); print(popt[0], popt[1])
    # plt.plot_simple_multi([temp, temp_x], [popt[0]*temp + popt[1], temp_y],
    #     ['fit', ''], ['k', 'b'], ['', 'o'], ['-', ''], [1, 0.4], [],
    #     xmin=-6.5, xmax=-4.5, ymin=7.6, ymax=10.5,
    #     xlabel=r'Roman log(F184-band flux/Jy)', ylabel='FAST++ logM')
    # linearfit = fitting.LinearLSQFitter()
    # popt = linearfit(models.Linear1D(), temp_x[good], temp_y[good]).parameters
    # print(popt[0], popt[1])
    
    # convert flux in observer frame to luminosity; the factor of (1 + z) comes
    # from the stretching of dlambda (spreading of photons in wavelength)
    # h_flux_Jy = h_flux*u.Jy
    # d_l = cosmo.luminosity_distance(model_redshift).to(u.m)
    # h_flux_Jy *= 4*np.pi*(1 + model_redshift)*np.square(d_l)
    # lum = (h_flux_Jy*(c.c/(1.8383069835389803*u.um)).to(u.Hz)).to(u.solLum).value
    # plt.plot_simple_dumb(np.log10(lum), fit_logM,
    #     xlabel='Roman log(F158-band luminosity/Lsol)', ylabel='FAST++ logM')
    # plt.plot_simple_dumb(fit_logM, np.log10(lum),
    #     ylabel='Roman log(F158-band luminosity/Lsol)', xlabel='FAST++ logM')
    # this ultimately works as a multipicative factor that is applied universally
    # and so doesn't change the plot of the raw flux versus stellar mass
    
    
    mask = np.all(snrs > 10, axis=1) # mask to the UV+opt+NIR high SNR population
    fit_SFR = fit_SFR[mask]
    uv_flux = photometry[:, 0][mask]
    # uvL_flux = photometry[:, 2][mask]
    # uS_flux = photometry[:, 4][mask]
    # u_flux = photometry[:, 6][mask]
    # g_flux = photometry[:, 8][mask]
    
    # determine the UV/opt color which has the strongest Pearson correlation with
    # fitted SFR
    # from scipy.stats import pearsonr
    # p = pearsonr(-2.5*np.log10(uv_flux/uvL_flux), np.log10(fit_SFR)) # R = -0.619
    # p = pearsonr(-2.5*np.log10(uv_flux/uS_flux), np.log10(fit_SFR)) # R = -0.635
    # p = pearsonr(-2.5*np.log10(uv_flux/u_flux), np.log10(fit_SFR)) # R = -0.633
    # p = pearsonr(-2.5*np.log10(uv_flux/g_flux), np.log10(fit_SFR)) # R = -0.685, 
    # p = pearsonr(-2.5*np.log10(uvL_flux/uS_flux), np.log10(fit_SFR)) # R = -0.612
    # p = pearsonr(-2.5*np.log10(uvL_flux/u_flux), np.log10(fit_SFR)) # R = -0.610
    # p = pearsonr(-2.5*np.log10(uvL_flux/g_flux), np.log10(fit_SFR)) # R = -0.675
    # p = pearsonr(-2.5*np.log10(uS_flux/u_flux), np.log10(fit_SFR)) # R = -0.442
    # p = pearsonr(-2.5*np.log10(uS_flux/g_flux), np.log10(fit_SFR)) # R = -0.677
    # p = pearsonr(-2.5*np.log10(u_flux/g_flux), np.log10(fit_SFR)) # R = -0.685, best correlation for colors
    # p = pearsonr(np.log10(uv_flux), np.log10(fit_SFR)) # R = 0.791, best correlation for band fluxes, R = 0.852 after sigma clipping
    # p = pearsonr(np.log10(uvL_flux), np.log10(fit_SFR)) # R = 0.769
    # p = pearsonr(np.log10(uS_flux), np.log10(fit_SFR)) # R = 0.719
    # p = pearsonr(np.log10(u_flux), np.log10(fit_SFR)) # R = 0.714
    # p = pearsonr(np.log10(g_flux), np.log10(fit_SFR)) # R = 0.531
    
    # adapted from
    # https://docs.astropy.org/en/latest/modeling/example-fitting-line.html
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
        sigma_clip, niter=4, sigma=3) # stable after 4 iterations
    fitted_line, mask = fitter(models.Linear1D(), np.log10(uv_flux), np.log10(fit_SFR))
    # print(pearsonr(np.log10(uv_flux)[~mask], np.log10(fit_SFR)[~mask]).statistic)
    # print(fitted_line.parameters)
    # plt.plot_simple_dumb(np.log10(uv_flux), np.log10(fit_SFR),
    #     xmin=-9.1, xmax=-5.9, ymin=-7.5, ymax=1)
    plt.plot_simple_dumb(np.log10(uv_flux)[~mask], np.log10(fit_SFR)[~mask],
        xmin=-9.1, xmax=-5.9, ymin=-7.5, ymax=1)
    
    # plot TNG SFR against UV flux
    # temp = np.linspace(-9.5, -5.9, 1000)
    # temp_x = np.log10(uv_flux)
    # temp_y = np.log10(fit_SFR)
    # good = np.isfinite(temp_x) & np.isfinite(temp_y)
    # popt, _ = curve_fit(linear, temp_x[good], temp_y[good]); print(popt[0], popt[1])
    # plt.plot_simple_multi([temp, temp_x], [popt[0]*temp + popt[1], temp_y],
    #     ['fit', ''], ['k', 'b'], ['', 'o'], ['-', ''], [1, 0.4], [],
    #     xmin=-9.5, xmax=-5.9, ymin=-3.5, ymax=0.8,
    #     xlabel='CASTOR log(UV flux/Jy)', ylabel='FAST++ log(SFR)')
    # linearfit = fitting.LinearLSQFitter()
    # popt = linearfit(models.Linear1D(), temp_x[good], temp_y[good]).parameters; print(popt[0], popt[1])
    
    
    return

def all_metrics(model_redshift=0.5) :
    
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
    
    outfile = 'tools/comparisons_z_{:03}.hdf5'.format(
       str(model_redshift).replace('.', ''))
    with h5py.File(outfile, 'r') as hf :
        tng_logM = hf['tng_logM'][:].reshape(212, 20)
        tng_SFR = hf['tng_SFR'][:].reshape(212, 20)
        # tng_metal = hf['tng_metal'][:]
        # tng_dust = hf['tng_dust'][:]
        fit_logM = hf['fit_logM'][:].reshape(212, 20)
        fit_SFR = hf['fit_SFR'][:].reshape(212, 20)
        # fit_metal = hf['fit_metal'][:]
        # fit_dust = hf['fit_dust'][:]
    
    # get the photometry file that was passed into FAST++
    photometry = np.loadtxt('photometry/photometry_2April2025.cat',
                            dtype=str, skiprows=1)[:, 1:-1].astype(float)
    mask = np.full(22, True); mask[-2], mask[-1] = False, False
    photometry = photometry[np.array(list(mask)*len(sample))] # select the correct rows
    snrs = np.array([photometry[:, 0]/photometry[:, 1],
                     photometry[:, 2]/photometry[:, 3],
                     photometry[:, 4]/photometry[:, 5],
                     photometry[:, 6]/photometry[:, 7],
                     photometry[:, 8]/photometry[:, 9],
                     photometry[:, 10]/photometry[:, 11],
                     photometry[:, 12]/photometry[:, 13],
                     photometry[:, 14]/photometry[:, 15],
                     photometry[:, 16]/photometry[:, 17]]).T
    roman_avg_snr = np.mean(snrs[:, 4:], axis=1).reshape(212, 20) # include CASTOR g-band
    
    # get the morphological metrics for the quenched galaxies
    metrics_file = 'D:/Documents/GitHub/TNG/TNG50-1/morphological_metrics_-10.5_+-1_2D.fits'
    table = Table.read(metrics_file)
    tmask = ((table['quenched_status'] == True) & (table['episode_progress'] >= 0.75) &
             ((table['mechanism'] == 1) | (table['mechanism'] == 3)))
    table = table[tmask]
    table.remove_columns(['quenched_status', 'sf_status', 'below_sfms_status',
                          'control_subID', 'quenched_comparison_mechanism'])
    tmask = np.full(len(table), False)
    idx = 0
    for quenched_subID in np.unique(table['quenched_subID']) :
        tmask[idx] = True
        idx += len(np.where(table['quenched_subID'] == quenched_subID)[0])
    table = table[tmask]
    
    # mask to the high SNR population
    mask = np.all(roman_avg_snr[:, :12] > 10, axis=1)
    sample = sample[mask]
    table = table[mask]
    tng_logM = tng_logM[mask]
    tng_SFR = tng_SFR[mask]
    fit_logM = fit_logM[mask]
    fit_SFR = fit_SFR[mask]
    
    # bin_edges = np.linspace(0, 5, 21) # units of Re
    # bin_centers = np.linspace(0.125, 4.875, 20) # units of Re
    
    # CSFs = np.zeros(len(sample))
    RSFs = np.zeros(len(sample))
    # Rinners = np.zeros(len(sample))
    # Routers = np.zeros(len(sample))
    for i, (subIDfinal, snap, subID, Re, logM_profile, SFR_profile, SNR_profile,
            tng_logM_profile, tng_SFR_profile) in enumerate(
        zip(sample['subIDfinal'], sample['snapshot'], sample['subID'],
            sample['Re'], fit_logM, fit_SFR, roman_avg_snr, tng_logM, tng_SFR)) :
        
        infile = 'GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(
            snap, subID, str(model_redshift).replace('.', ''))
        with fits.open(infile) as hdu :
            infile_redshift = hdu[0].header['REDSHIFT']
            # shape = (hdu[0].data.shape[1], hdu[0].data.shape[2])
            plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix
        assert infile_redshift == model_redshift
        
        # convert Re, 1 kpc into pixels
        Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
        # kpc_pix = (1*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
        
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(model_redshift).to(u.kpc/u.arcsec)
        pixel_area_physical = np.square(kpc_per_arcsec)*np.square(plate_scale*u.pix)
        
        # CSFs[i] = calculate_CSF1(SFR_profile, kpc_pix, Re_pix, bin_edges) # integrate out to 1 kpc, compared to integrated (5 Re) profile
        # CSFs[i] = calculate_CSF2(subIDfinal, snap, subID) # use bestfit results from 1 kpc aperture, compared to integrated (5 Re) aperture
        # CSFs[i] = calculate_CSF3(SFR_profile) # integrate out to 1 Re, compared to integrated (5 Re) profile
        # CSFs[i] = calculate_CSF4(SFR_profile, kpc_pix, Re_pix, bin_edges) # interpolate out to 1 kpc and use integrated mass, compared to integrated (5 Re) profile
        # CSFs[i] = calculate_CSF5(snap, subID, Re_pix, kpc_pix) # use CASTOR UV image and flux within 1 kpc, compared to flux within integrated (5 Re) aperture
        # CSFs[i] = calculate_CSF6(snap, subID, Re_pix) # use CASTOR UV image and flux within 1 Re, compared to flux within integrated (5 Re) aperture
        # CSFs[i] = calculate_CSF7(snap, subID, Re_pix, bin_edges) # use CASTOR UV image and calculate concentration like Conselice based on curve of growth for UV flux
        # CSFs[i] = calculate_CSF8(snap, subID) # use CASTOR UV image and calculate concentration like Conselice using statmorph
        # fit Sersic profile to photometry?
        # CSFs[i] = calculate_CSF9(snap, subID, Re_pix, kpc_pix)
        
        
        # RSFs[i] = calculate_RSF1(snap, subID, Re_pix) # use CASTOR UV image half-light radius, compared to Roman H-band image half-light radius
        # RSFs[i] = calculate_RSF2(subIDfinal, snap, subID, Re_pix) # use CASTOR UV image Sersic profile, compared to Roman H-band Sersic profile
        # try same thing as with concentration integrating out the profile to the 50th percent, for both SFR and logM
        RSFs[i] = calculate_RSF3(subIDfinal, snap, subID, Re_pix, pixel_area_physical)
        
        # calculate Rinner and Router
        # profile = np.log10(SFR_profile) - logM_profile
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-11, slope=1)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile) # as in TNG: threshold=-10.5, slope=1
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-9.5, slope=1)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-9, slope=1)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-10.5, slope=0.5)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-10.5, slope=0.75)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-10.5, slope=1.25)
        # Rinners[i] = calculate_Rinner1(bin_centers, profile, threshold=-10.5, slope=1.5)
        # Rinners[i] = calculate_Rinner2(snap, subID, Re_pix) # use CASTOR UV image profile, compared to Roman H-band image profile
        # cumulative SFR growth curve, look for breaks in the profile or 90th percentile for Router, and 10th percentile for Rinner
        
        # profile = np.log10(SFR_profile) - logM_profile
        # Routers[i] = calculate_Router(bin_centers, profile)
        
        
    '''
    CSFs = [0.2501604750878618, 0.7079750024364541, 0.4112941977700516,
            0.6235375945497238, 0.4676898740193128, 0.557486797380193,
            0.29220905211190806, 0.7892108249133367, 0.6263019174766814,
            0.48061685410934113, 0.5910950052186251, 0.6629737416446977,
            0.5060474183040443, 0.21552205631299132, 0.4787989487935712,
            0.5030759109622821, 0.5396857092585854, 0.24931659885328425,
            0.46580036606588004, 0.5631419697146938, 0.48066413285991016,
            0.5840006059913643, 0.49160529928550395, 0.37538292474096935,
            0.49550188634766573, 0.7725503185635211, 0.4561614200062797,
            0.4877475854989184, 0.6606788047604635, 0.3978644904865723,
            0.7966387559624265, 0.46984876319052543, 0.6150423598416866,
            0.5152691917706113, 0.4276101670189293, 0.5616831832751847,
            0.455018100451886, 0.4693433459472088, 0.4994097379233958,
            0.5179555099460299, 0.6059237145586653, 0.5272251264453156,
            0.6878539106335281, 0.3016948652520103, 0.25002512780184355,
            0.2241640817255925, 0.5569289460813747, 0.4447809221392399,
            0.5370913714213238, 0.5056962975557614, 0.23975192677643697,
            0.21420381125497098, 0.474635691833171, 0.3686858698546442,
            0.39917283489824473, 0.1754689103073782, 0.39340973009319946,
            0.17658617701076584, 0.5669860559871269, 0.4338732216643278,
            0.18013084463537476, 0.29224729401256944, 0.3858185090892018,
            0.49276069593734473, 0.49173250596915674, 0.4713068224132906,
            0.4540700587887748, 0.2978644900953535, 0.23632112403789357,
            0.4453105306173873, 0.5209412553673308, 0.2626399110951055,
            0.5156831842901113, 0.4540892477086014, 0.3627873541296004,
            0.5324058417585926, 0.21289961248362985, 0.35395368516017195,
            0.40344519633452397, 0.4732518549917245, 0.5937958191749053,
            0.45896928356434097, 0.2154980515853037, 0.7152868660989681,
            0.5728994075750097, 0.3579881187065726, 0.508666778917566,
            0.26940573025956444, 0.5657703748908777, 0.548948728685523,
            0.561830676013757, 0.4466258367574463, 0.47320823450294414,
            0.4420992069952246, 0.30489513887454806, 0.49964261076957384,
            0.380560122464974, 0.368664780401943, 0.5135610020167061,
            0.31068422257190076, 0.4815454944861228, 0.33582214329115484,
            0.8540903770167232, 0.45379993363209686, 0.3736507885000691,
            0.598568764986145, 0.4793541051337497, 0.3705504752593801,
            0.6880107452492313, 0.3195816550527377, 0.532074372766329,
            0.4637747353664722, 0.25211744646975803, 0.46256353065308764,
            0.6057762100898575] # using calculate_CSF8
    CSFs = np.array(CSFs)
    '''
    # io_labels = sample['subIDfinal'][sample['mechanism'] == 1].astype(str).tolist()
    # oi_labels = sample['subIDfinal'][sample['mechanism'] == 3].astype(str).tolist()
    
    # plot the results
    # xx = np.linspace(-0.02, 1.02, 1000)
    # mask = np.isfinite(CSFs)
    # popt, _ = curve_fit(linear, table['C_SF'][mask], CSFs[mask])
    # CSFs = (1 - popt[0])*table['C_SF'] + CSFs - popt[1]
    # CSFs[CSFs < 0.] = 0.
    # CSFs[CSFs > 1.] = 1.
    # plt.plot_simple_multi(
    #     [xx, table['C_SF'][table['mechanism'] == 1],
    #      table['C_SF'][table['mechanism'] == 3]],
    #     [xx, CSFs[sample['mechanism'] == 1], CSFs[sample['mechanism'] == 3]],
    #     ['', 'inside-out', 'outside-in'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     [io_labels, oi_labels],
    #     xlabel=r'$C_{\rm SF}$ from TNG', ylabel=r'$C_{\rm SF}$ fitted from FAST++',
    #     xmin=-0.02, xmax=1.02, ymin=-0.02, ymax=1.02)
    
    # xx = np.linspace(-0.75, 1.54)
    # mask = np.isfinite(RSFs)
    # popt, _ = curve_fit(linear, np.log10(table['R_SF'][mask]), RSFs[mask])
    # RSFs = (1 - popt[0])*np.log10(table['R_SF']) + RSFs - popt[1]
    # plt.plot_simple_multi(
    #     [xx, np.log10(table['R_SF'][table['mechanism'] == 1]),
    #      np.log10(table['R_SF'][table['mechanism'] == 3])],
    #     [xx, RSFs[sample['mechanism'] == 1],
    #      RSFs[sample['mechanism'] == 3]],
    #     ['', 'inside-out', 'outside-in'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     [io_labels, oi_labels],
    #     xlabel=r'$R_{\rm SF}$ from TNG', ylabel=r'$R_{\rm SF}$ fitted from FAST++',
    #     xmin=-0.75, xmax=1.54, ymin=-0.75, ymax=1.54)
    
    # xx = np.linspace(-0.1, 5.1, 1000)
    # plt.plot_simple_multi(
    #     [xx, np.random.normal(table['Rinner'][table['mechanism'] == 1], 0.03),
    #      np.random.normal(table['Rinner'][table['mechanism'] == 3], 0.03)],
    #     [xx, Rinners[sample['mechanism'] == 1], Rinners[sample['mechanism'] == 3]],
    #     ['', 'inside-out', 'outside-in'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     [io_labels, oi_labels],
    #     xlabel=r'$R_{\rm inner}$ from TNG', ylabel=r'$R_{\rm inner}$ fitted from FAST++',
    #     xmin=-0.1, xmax=5.1, ymin=-0.1, ymax=5.1)
    
    # plt.plot_simple_multi(
    #     [xx, np.random.normal(table['Router'][table['mechanism'] == 1], 0.03),
    #      np.random.normal(table['Router'][table['mechanism'] == 3], 0.03)],
    #     [xx, Routers[sample['mechanism'] == 1], Routers[sample['mechanism'] == 3]],
    #     ['', 'inside-out', 'outside-in'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     [io_labels, oi_labels],
    #     xlabel=r'$R_{\rm outer}$ from TNG', ylabel=r'$R_{\rm outer}$ fitted from FAST++',
    #     xmin=-0.1, xmax=5.1, ymin=-0.1, ymax=5.1)
    
    
    return

def calculate_CSF1(SFR_profile, kpc_pix, Re_pix, bin_edges) :
    
    # 1 kpc = ~3.19 pix at redshift z = 0.5
    
    # calculate the mass formed within 100 Myr, based on the SFR from FAST++
    young_mass_profile = SFR_profile*1e8
    
    # define the fraction of the young mass profile that we want to use to
    # calculate C_SF, based on the inner 1 kpc
    mass_frac = np.full(SFR_profile.shape, 0.0)
    
    # get the bin edges, measured in pixels
    bin_edges_pix = Re_pix*bin_edges
    
    # calculate the area of the annuli
    area_per_annuli = np.diff(np.square(bin_edges_pix))
    
    # find which radial bin 1 kpc is within
    inner = np.where(kpc_pix > bin_edges_pix)[0][-1]
    
    # update the fractions array for future use
    mass_frac[:inner] = 1.0
    
    # calculate the fraction of area covered up to 1 kpc in the correct bin
    mass_frac[inner] = (np.square(kpc_pix) -
                        np.square(bin_edges_pix[inner]))/area_per_annuli[inner]
    
    # calculate the total mass within 1 kpc
    C_SF = np.nansum(mass_frac*young_mass_profile)/np.nansum(young_mass_profile)
    
    return C_SF

def calculate_CSF2(subIDfinal, snap, subID, skiprows=18) :
    
    # load fitted data coming out of FAST++
    data = np.loadtxt('fits/fits_2April2025.fout', dtype=str, skiprows=skiprows)
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, :2].astype(int)
    use = (ids[:, 0] == snap) & (ids[:, 1] == subID)
    use[np.where(use)[0][:-2]] = False # select only 1 kpc and integrated bins
    
    sfr_profile = np.power(10, data[:, 14].astype(float)[use]) # sfr100
    
    return sfr_profile[0]/sfr_profile[1]

def calculate_CSF3(SFR_profile) :
    return np.nansum(SFR_profile[:4])/np.nansum(SFR_profile)

def calculate_CSF4(SFR_profile, kpc_pix, Re_pix, bin_edges) :
    
    young_mass_cumulative_profile = np.nancumsum(SFR_profile*1e8)
    
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    young_mass_within_kpc = np.interp(kpc_pix, Re_pix*bin_centers,
                                      young_mass_cumulative_profile)
    
    return young_mass_within_kpc/young_mass_cumulative_profile[-1]

def calculate_CSF5(snap, subID, Re_pix, kpc_pix, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        image = hdu[0].data[0] # Jy [per pixel]
    
    # determine the center of the image
    center = (int((image.shape[0] - 1)/2), int((image.shape[0] - 1)/2))
    
    # calculate the total CASTOR UV flux within 1 kpc and 5 Re apertures
    kpc_flux = CircularAperture(center, kpc_pix).do_photometry(image)[0][0]
    int_flux = CircularAperture(center, 5*Re_pix).do_photometry(image)[0][0]
    
    return kpc_flux/int_flux

def calculate_CSF6(snap, subID, Re_pix, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        image = hdu[0].data[0] # Jy [per pixel]
    
    # determine the center of the image
    center = (int((image.shape[0] - 1)/2), int((image.shape[0] - 1)/2))
    
    # calculate the total CASTOR UV flux within 1 and 5 Re apertures
    one_flux = CircularAperture(center, Re_pix).do_photometry(image)[0][0]
    int_flux = CircularAperture(center, 5*Re_pix).do_photometry(image)[0][0]
    
    return one_flux/int_flux

def calculate_CSF7(snap, subID, Re_pix, bin_edges, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        image = hdu[0].data[0] # Jy [per pixel]
        error = hdu[0].data[1] # Jy [per pixel]
    
    # determine the center of the image
    center = (int((image.shape[0] - 1)/2), int((image.shape[0] - 1)/2))
    
    cog = CurveOfGrowth(image, center, Re_pix*np.linspace(0, 5, 101)[1:],
                        error=error)
    cog.normalize(method='max')
    try :
        r20 = cog.calc_radius_at_ee(0.2)
        r80 = cog.calc_radius_at_ee(0.8)
        concentration = np.log10(r80/r20)
    except ValueError :
        concentration = np.nan
    
    return concentration

def calculate_CSF8(snap, subID, model_redshift=0.5, psf=0.15*u.arcsec) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix # the plate scale of the images
        image = hdu[0].data[0] # Jy [per pixel]
        error = hdu[0].data[1] # Jy [per pixel]
    
    sigma = psf/(2*np.sqrt(2*np.log(2))) # arcseconds
    sigma_pix = sigma/plate_scale # pixels
    kernel = Gaussian2DKernel(sigma_pix.value)
    psf = kernel.array
    
    segmap = detect_sources(convolve(image, kernel), detect_threshold(image, 1.5),
                            npixels=10)
    
    if len(segmap.labels) > 1 :
        best_label = segmap.labels[np.argsort(segmap.areas)][-1]
        # seg_array = segmap.data
        # seg_array[seg_array != best_label] = 0. # mask out small sources
        # seg_array = (seg_array/best_label).astype(int)
        segmap.keep_label(best_label, relabel=True)
    
    morph = statmorph.source_morphology(image, segmap, weightmap=error,
                                        psf=psf)[0]
    
    return morph.concentration/5

def calculate_CSF9(snap, subID, Re_pix, kpc_pix, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        image = hdu[0].data[0] # Jy [per pixel]
    image = uv_flux_to_sfr(image)
    image[~np.isfinite(image)] = 0.0
    
    # determine the center of the image
    center = (int((image.shape[0] - 1)/2), int((image.shape[0] - 1)/2))
    
    # calculate the total CASTOR UV flux within 1 kpc and 5 Re apertures
    kpc_flux = CircularAperture(center, kpc_pix).do_photometry(image)[0][0]
    int_flux = CircularAperture(center, 5*Re_pix).do_photometry(image)[0][0]
    
    cog = CurveOfGrowth(image, center, Re_pix*np.linspace(0, 5, 101)[1:])
    # if snap < 30 :
    #     plt.plot_simple_dumb(cog.radius, cog.profile)
    cog.normalize(method='max')
    # try :
    r10 = cog.calc_radius_at_ee(0.1)
        # r20 = cog.calc_radius_at_ee(0.2)
    #     r80 = cog.calc_radius_at_ee(0.8)
    #     concentration = np.log10(r80/r20)
    # except ValueError :
    #     concentration = np.nan
    
    return np.log10(r10/Re_pix) #kpc_flux/int_flux




def calculate_RSF1(snap, subID, Re_pix, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        uv = hdu[0].data[0] # Jy [per pixel], CASTOR UV band
        uv_error = hdu[0].data[1] # Jy [per pixel]
        nir = hdu[0].data[14] # Jy [per pixel], H band
        nir_error = hdu[0].data[15] # Jy [per pixel]
    
    # determine the center of the image
    center = (int((uv.shape[0] - 1)/2), int((uv.shape[0] - 1)/2))
    
    uv_cog = CurveOfGrowth(uv, center, Re_pix*np.linspace(0, 5, 101)[1:],
                           error=uv_error)
    uv_cog.normalize(method='max')
    
    nir_cog = CurveOfGrowth(nir, center, Re_pix*np.linspace(0, 5, 101)[1:],
                            error=nir_error)
    nir_cog.normalize(method='max')
    
    try :
        Re_uv = uv_cog.calc_radius_at_ee(0.5) # [pixel]
        Re_nir = nir_cog.calc_radius_at_ee(0.5) # [pixel]
        RSF = np.log10(Re_uv/Re_nir)
    except ValueError :
        RSF = np.nan
    
    return RSF

def calculate_RSF2(subIDfinal, snap, subID, Re_pix, model_redshift=0.5) :
    
    # if snap < 24 :
    #     print(subIDfinal)
    #     with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
    #         str(model_redshift).replace('.', ''))) as hdu :
    #         uv = hdu[0].data[0] # Jy [per pixel], CASTOR UV-band
    #         uv_error = hdu[0].data[1] # Jy [per pixel]
    #         nir = hdu[0].data[16] # Jy [per pixel], Roman K-band (F184)
    #         nir_error = hdu[0].data[17] # Jy [per pixel]
    #         ny, nx = uv.shape
        
        # determine the center of the image
        # center = (int((uv.shape[0] - 1)/2), int((uv.shape[0] - 1)/2))
        
        # create example logM maps based on derived scaling
        # logM_map = 1.04401005*np.log10(nir) + 14.88105114
        # sm_cog = CurveOfGrowth(logM_map, center, Re_pix*np.linspace(0, 5, 101)[1:])
        
        
        
        
        # create example SFR maps
        # sfr_map = np.power(10, 1.044486810*np.log10(uv) + 6.29683788)
        # sfr_cog = CurveOfGrowth(sfr_map, center, Re_pix*np.linspace(0, 5, 101)[1:])
        # sfr_prof = RadialProfile(sfr_map, center, Re_pix*np.linspace(0, 5, 21)[1:])
        # plt.plot_simple_dumb(sfr_cog.radius, sfr_cog.profile)
        # plt.plot_simple_dumb(sfr_prof.radius, sfr_prof.profile)
        
        # plt.display_image_simple(nir/nir_error, lognorm=False)
        # nir_cog = CurveOfGrowth(nir, center, Re_pix*np.linspace(0, 5, 21)[1:],
        #                         error=nir_error)
        # nir_cog.normalize(method='max')
        # Re_nir = nir_cog.calc_radius_at_ee(0.5) # [pixel]
        
        # nir_prof = RadialProfile(nir, center, Re_pix*np.linspace(0, 5, 21),
        #                          error=nir_error)
        # plt.plot_simple_dumb(nir_prof.radius, nir_prof.profile, scale='log')
        # mod = models.Sersic1D(amplitude=1, r_eff=Re_nir, n=0.6)
        # plt.plot_simple_dumb(nir_prof.radius, mod(nir_prof.radius), scale='log')
        
        # fitter = fitting.LMLSQFitter()
        # sersic = fitter(models.Sersic1D(r_eff=Re_nir), nir_prof.radius, nir_prof.profile)
        
        # yy, xx = np.mgrid[0:ny, 0:nx].astype(uv.dtype)
        # fitter = fitting.LMLSQFitter()
        # sersic = fitter(models.Sersic2D(r_eff=Re_nir,
        #                 fixed={'x_0':center[0], 'y_0':center[1]}),
        #                 xx, yy, z=nir, weights=1/nir_error)
        # plt.display_image_simple(sersic(xx, yy))
    
    return

def calculate_RSF3(subIDfinal, snap, subID, Re_pix, pixel_area_physical,
                   model_redshift=0.5) :
    
    if subIDfinal == 531320 :
    # from scipy.ndimage import gaussian_filter1d
    
    # print(subIDfinal, snap, subID)
    
        with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', ''))) as hdu :
            uv = hdu[0].data[0] # Jy [per pixel], CASTOR UV band
            uv_error = hdu[0].data[1] # Jy [per pixel]
            # y_band = hdu[0].data[10] # Jy [per pixel], Y band (F106)
            # y_error = hdu[0].data[11] # Jy [per pixel]
            # j_band = hdu[0].data[12] # Jy [per pixel], J band (F129)
            # j_error = hdu[0].data[13] # Jy [per pixel]
            h_band = hdu[0].data[14] # Jy [per pixel], H band (F158)
            h_error = hdu[0].data[15] # Jy [per pixel]
            # k_band = hdu[0].data[16] # Jy [per pixel], K_Roman band (F184)
            # k_error = hdu[0].data[17] # Jy [per pixel]
        '''
        plt.display_image_simple(y_band/y_error, title='H/H error', vmin=0.3, vmax=10)
        plt.display_image_simple(j_band/j_error, title='H/H error', vmin=0.3, vmax=10)
        plt.display_image_simple(h_band/h_error, title='H/H error', vmin=0.3, vmax=10)
        plt.display_image_simple(k_band/k_error, title='H/H error', vmin=0.3, vmax=10)
        a = np.array([y_band/y_error, j_band/j_error, h_band/h_error, k_band/k_error])
        b = np.mean(a, axis=0)
        plt.display_image_simple(b, vmin=0.3, vmax=10)
        d = b.copy()
        d[d < 0] = 0.0
        e = convolve(d, np.full((3, 3), 1/9))
        plt.display_image_simple(e, vmin=0.3, vmax=10)
        print(np.sum(e >= 10)/(e.shape[0]*e.shape[1]))
        '''
        # determine the center of the image
        center = (int((uv.shape[0] - 1)/2), int((uv.shape[0] - 1)/2))
        
        # use an averaging filter to determine the average for adjacent pixels,
        # including the central pixel; inspired by
        # https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html#the-mean-filter
        mean_kernel = np.full((3, 3), 1/9)
        
        # average the UV flux image, setting negative pixels to zero
        temp = uv.copy()
        temp[temp < 0] = 0.0
        uv_avg = convolve(temp, mean_kernel)
        
        # for pixels with low SNR (ie. SNR < 10), replace those pixels with the
        # values from the averaged map
        uv_final = uv.copy()
        mask = (uv/uv_error < 10)
        uv_final[mask] = uv_avg[mask]
        # plt.display_image_simple(uv, title='UV', vmin=1e-11, vmax=1e-8)
        # plt.display_image_simple(uv_avg, title='avg(UV)', vmin=1e-11, vmax=1e-8)
        # plt.display_image_simple(uv_final, title='UV with low SNR pixels replaced', vmin=1e-11, vmax=1e-8)
        # plt.display_image_simple(uv/uv_error, title='SNR = UV/UV error', vmin=0.3, vmax=10)
        # plt.display_image_simple(uv_avg/uv_error, title='avg(UV)/UV error', vmin=0.3, vmax=10)
        # plt.display_image_simple(uv_final/uv_error, title='final UV/UV error', vmin=0.3, vmax=10)
        
        # then use the final UV image and convert to logSFR
        logSFR_map = uv_flux_to_logsfr(uv_final)
        
        # propagate uncertainties to determine the logSFR error map
        # logSFR_e_map = np.sqrt(np.square(1.22004385/np.log(10)*uv_error/uv_final))
        
        # convert to SFR maps from logSFR
        sfr_map = np.power(10, logSFR_map)
        # sfr_map_lo = np.abs(np.power(10, logSFR_map - logSFR_e_map) - sfr_map)
        # sfr_map_hi = np.abs(np.power(10, logSFR_map + logSFR_e_map) - sfr_map)
        
        # investigate a curve of growth for the SFR map
        sfr_cog = CurveOfGrowth(sfr_map, center, Re_pix*np.linspace(0, 5, 101)[1:])
        sfr_cog.normalize(method='max')
        Re_sfr = sfr_cog.calc_radius_at_ee(0.5) # [pixel]
        # plt.plot_simple_dumb(sfr_cog.radius, sfr_cog.profile)
        
        # sfr_prof = RadialProfile(sfr_map, center, Re_pix*np.linspace(0, 5, 21)[1:])
        # plt.plot_simple_dumb(sfr_prof.radius, sfr_prof.profile)
        
        # get the edges of the circular annuli in units of pixels for masking
        # Nannuli = 100
        # edges_pix = np.linspace(0, 5, Nannuli+1)*Re_pix # edges in units of Re
        # radial_bin_centers = np.linspace(0.125, 4.875, Nannuli) # units of Re
        # sfr_prof = np.full(Nannuli, -1.0)
        # nPixel_profile = np.full(Nannuli, -1.0)
        # for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        #     if start == 0 :
        #         ap = CircularAperture(center, end)
        #     else :
        #         ap = CircularAnnulus(center, start, end)
        #     sfr_prof[i] = ap.do_photometry(sfr_map)[0][0]
        #     nPixel_profile[i] = ap.area # the pixel areas per annulus
        # physical_area_profile = pixel_area_physical*nPixel_profile
        # plt.plot_simple_dumb(radial_bin_centers,
        #     gaussian_filter1d(sfr_prof/physical_area_profile.value, 1),
        #     ymin=1e-3, ymax=1e-1)
        
        # average the NIR flux image
        temp = h_band.copy()
        temp[temp < 0] = 0.0
        h_avg = convolve(temp, mean_kernel)
        
        # for pixels with low SNR (ie. SNR < 10), replace those pixels with the
        # values from the averaged map
        h_final = h_band.copy()
        mask = (h_band/h_error < 10)
        h_final[mask] = h_avg[mask]
        # plt.display_image_simple(nir, title='NIR', vmin=1e-10, vmax=1e-6)
        # plt.display_image_simple(nir_avg, title='avg(NIR)', vmin=1e-10, vmax=1e-6)
        # plt.display_image_simple(nir_final, title='NIR with low SNR pixels replaced', vmin=1e-10, vmax=1e-6)
        plt.display_image_simple(h_band/h_error, title='SNR = NIR/NIR error', vmin=0.3, vmax=10)
        plt.display_image_simple(h_avg/h_error, title='avg(NIR)/NIR error', vmin=0.3, vmax=10)
        plt.display_image_simple(h_final/h_error, title='final NIR/NIR error', vmin=0.3, vmax=10)
        # print(np.sum(h_band/h_error >= 10))
        # print(np.sum(h_avg/h_error >= 10))
        # print(np.sum(h_final/h_error >= 10))
        
        # then use the final NIR image and convert to logM
        logM_map = hband_flux_to_logM(h_final)
        
        # propagate uncertainties to determine the logM error map
        # logM_e_map = np.sqrt(np.square(1.02078658/np.log(10)*nir_error/nir_final))
        
        # convert to mass maps from logM
        mass_map = np.power(10, logM_map)
        
        # investigate a curve of growth for the mass map
        mass_cog = CurveOfGrowth(mass_map, center, Re_pix*np.linspace(0, 5, 101)[1:])
        mass_cog.normalize(method='max')
        Re_mass = mass_cog.calc_radius_at_ee(0.5) # [pixel]
        # plt.plot_simple_dumb(mass_cog.radius, mass_cog.profile)
        
        # mass_prof = RadialProfile(mass_map, center, Re_pix*np.linspace(0, 5, 21)[1:])
        # area = mass_prof.area/(kpc_pix*kpc_pix)
        # plt.plot_simple_dumb(mass_prof.radius/Re_pix, np.log10(mass_prof.profile/area))
        
        # get the edges of the circular annuli in units of pixels for masking
        # edges_pix = np.linspace(0, 5, Nannuli+1)*Re_pix # edges in units of Re
        # radial_bin_centers = np.linspace(0.125, 4.875, Nannuli) # units of Re
        # mass_prof = np.full(Nannuli, -1.0)
        # nPixel_profile = np.full(Nannuli, -1.0)
        # for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        #     if start == 0 :
        #         ap = CircularAperture(center, end)
        #     else :
        #         ap = CircularAnnulus(center, start, end)
        #     mass_prof[i] = ap.do_photometry(mass_map)[0][0]
        #     nPixel_profile[i] = ap.area # the pixel areas per annulus
        # physical_area_profile = pixel_area_physical*nPixel_profile
        # plt.plot_simple_dumb(radial_bin_centers,
        #     gaussian_filter1d(mass_prof/physical_area_profile.value, 1),
        #     ymin=1e7, ymax=1e11)
        
        # plt.display_image_simple(logSFR_map - logM_map, lognorm=False, title='sSFR')
        # plt.plot_simple_dumb(radial_bin_centers,
        #     gaussian_filter1d(sfr_prof/mass_prof, 1),
        #     ymin=1e-13, ymax=1e-9)
        
        
        # RSF = np.log10(Re_sfr/Re_mass)
        
        # print(logM_prof.area/kpc_sqr)
        # plt.plot_simple_dumb(logM_prof.radius, logM_prof.profile - np.log10(logM_prof.area*kpc_sqr))
        # plt.plot_simple_dumb(logM_prof.radius/Re_pix, np.log10(sfr_prof.profile) - logM_prof.profile)
    
    return 0


def calculate_Rinner1(bin_centers, profile, threshold=-10.5, slope=1) :
    
    # for NaNs, set the value in each bin to be below the minimum valid
    # value appearing in the profile
    mask = ~np.isfinite(profile)
    profile[mask] = np.min(profile[np.isfinite(profile)]) - 0.5
    
    # take the derivative of the profile, using the bin centers as x values
    deriv = np.gradient(profile, bin_centers)
    
    # find the locations where the derivative is more than our desired slope
    locs = np.where(deriv >= slope)[0]
    
    if len(locs) > 0 :
        # loop through every location, and check to see if the profile is
        for loc in np.flip(locs) : # always less than the threshold value
            if np.all(profile[:loc+1] <= threshold) : # before that location
                Rinner = bin_centers[loc] # if so, return that location
                break
            else :
                Rinner = 0
    else :         # if the derivative is never more than the desired slope,
        Rinner = 0 # simply return the innermost radius
    
    return Rinner

def calculate_Rinner2(snap, subID, Re_pix, model_redshift=0.5) :
    
    with fits.open('cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))) as hdu :
        uv = hdu[0].data[0] # Jy [per pixel], CASTOR UV band
        uv_error = hdu[0].data[1] # Jy [per pixel]
        nir = hdu[0].data[14] # Jy [per pixel], H band
        nir_error = hdu[0].data[15] # Jy [per pixel]
    
    # determine the center of the image
    center = (int((uv.shape[0] - 1)/2), int((uv.shape[0] - 1)/2))
    
    if snap < 30 :
        uv_prof = RadialProfile(uv, center, Re_pix*np.linspace(0, 5, 21),
                                error=uv_error)
        nir_prof = RadialProfile(nir, center, Re_pix*np.linspace(0, 5, 21),
                                 error=nir_error)
        import matplotlib.pyplot as mplplt
        fig, ax = mplplt.subplots()
        uv_prof.plot(ax=ax, label='UV')
        uv_prof.plot_error(ax=ax)
        nir_prof.plot(ax=ax, label='NIR')
        nir_prof.plot_error(ax=ax)
        ax.legend()
    
    return



def calculate_Router(bin_centers, profile, threshold=-10.5, slope=-1) :
    
    # for NaNs, set the value in each bin to be below the minimum valid
    # value appearing in the profile
    mask = ~np.isfinite(profile)
    profile[mask] = np.min(profile[np.isfinite(profile)]) - 0.5
    
    # take the derivative of the profile, using the bin centers as x values
    deriv = np.gradient(profile, bin_centers)
    
    # find the locations where the derivative is less than our desired slope
    locs = np.where(deriv <= slope)[0]
    
    if len(locs) > 0 :
        # loop through every location, and check to see if the profile is
        for loc in locs : # always less than the threshold value beyond
            if np.all(profile[loc:] <= threshold) : # that location
                Router = bin_centers[loc] # if so, return that location
                break
            else :
                Router = 5
    else :         # if the derivative is never less than the desired slope,
        Router = 5 # simply return the outermost radius
    
    return Router

def linear(xx, slope, intercept) :
    return slope*xx + intercept

def quad(xx, aa, bb, cc) :
    return aa*xx*xx + bb*xx + cc

# mass_to_light_ratio_circular_annuli()
# all_metrics()

# import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

# from photutils.aperture import CircularAnnulus, CircularAperture
# from photutils.profiles import CurveOfGrowth, RadialProfile

# plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix

# convert Re into pixels
# Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
# kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(model_redshift).to(u.kpc/u.arcsec)
# pixel_area_physical = np.square(kpc_per_arcsec)*np.square(plate_scale*u.pix)

# determine the center of the image
# center = (int((images.shape[1] - 1)/2), int((images.shape[2] - 1)/2))

# mass_prof = create_radial_profile(mass_image, Re_pix, center, pixel_area_physical)
# sfr_prof = create_radial_profile(sfr_image, Re_pix, center, pixel_area_physical)

