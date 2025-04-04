
from os.path import exists
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d

from core import get_particles
from fastpy import (calculate_chi2, calzetti2000, dtt_from_fit, get_bc03_waves,
                    get_filter_waves, get_lum2fl, get_model_fluxes, get_times,
                    sfh2sed_fastpp)
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def compare_fast_models() :
    
    data = np.loadtxt('fitting/photometry_14July2024.fout', dtype=str)
    lmass = data[:, 16].astype(float)
    lsfr = data[:, 19].astype(float) # 'lsfr

    data = np.loadtxt('fitting/photometry_14July2024_noErr.fout', dtype=str)
    lmass_noErr = data[:, 6].astype(float)
    lsfr_noErr = data[:, 7].astype(float) # 'lsfr'

    data = np.loadtxt('fitting/photometry_14July2024_noErr_fineAgeGrid.fout', dtype=str)
    lmass_noErr_fineAgeGrid = data[:, 6].astype(float)
    lsfr_noErr_fineAgeGrid = data[:, 7].astype(float) # 'lsfr'

    plt.histogram_multi([lmass, lmass_noErr, lmass_noErr_fineAgeGrid],
        'mass', ['k', 'r', 'b'], ['-', '-', '-'],
        ['delayed-tau', 'no error', 'no error + finer age grid'], [50, 50, 50])

    plt.histogram_multi([lsfr[lsfr >= -10], lsfr_noErr[lsfr_noErr >= -10],
        lsfr_noErr_fineAgeGrid[lsfr_noErr_fineAgeGrid >= -10]], 'SFR',
        ['k', 'r', 'b'], ['-', '-', '-'],
        ['delayed-tau', 'no error', 'no error + finer age grid'], [50, 50, 50], loc=2)
    
    return

def compare_fast_output_to_tng(subID, snap, population='quenched',
                               display=False, realizations=False) :
    
    # get galaxy attributes from TNG
    time, mpbsubID, logM, tngRe, center = load_galaxy_attributes(subID, snap)
    
    # get the annuli map, derived using SourceXtractor++ morphological values
    bins_image, dim, numBins = load_annuli_map(population, subID)
    
    # get stellar mass and SFR maps direct from TNG, assuming 100 Myr for the
    # duration of star formation
    # edges = np.linspace(-10*tngRe, 10*tngRe, dim + 1) # kpc
    edges = np.linspace(-5*tngRe, 5*tngRe, dim + 1) # kpc
    tng_Mstar_map, tng_SFR_map = spatial_plot_info(time, snap, mpbsubID,
        center, tngRe, edges, 100*u.Myr)
    tng_Mstar_map = np.rot90(tng_Mstar_map, k=3) # rotate images to match SKIRT
    tng_SFR_map = np.rot90(tng_SFR_map, k=3)
    
    if display :
        plt.display_image_simple(tng_Mstar_map)
        plt.display_image_simple(tng_SFR_map)
        # plt.display_image_simple(bins_image, lognorm=False)
    
    # get the output from FAST++
    lmass, lmass_lo, lmass_hi, lsfr, lsfr_lo, lsfr_hi = load_fast_fits(subID,
        numBins, realizations=realizations)
    
    # get basic information from the photometric table
    nPixels, redshift, scale, rr, Re = load_photometric_information(population,
        subID, realizations=realizations)
    
    # determine the area of a single pixel, in arcsec^2
    pixel_area = np.square(scale*u.pix)
    
    # determine the physical projected area of a single pixel
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc/u.arcsec)
    pixel_area_physical = np.square(kpc_per_arcsec)*pixel_area
    
    # determine the projected physical areas of every annulus
    physical_area = pixel_area_physical*nPixels/u.pixt
    # physical_area_pc2 = physical_area.to(np.square(u.pc))
    
    # convert the pixel values to physical sizes, mostly for plotting purposes
    rr = rr*scale*kpc_per_arcsec # kpc
    Re = Re*scale*kpc_per_arcsec # kpc
    # rs = np.log10(rr/Re)
    
    # loop through the elliptical annuli, binning together valid pixels
    tng_annuli_masses = np.full(numBins, 0.0)
    tng_annuli_sfrs = np.full(numBins, 0.0)
    for val in range(numBins) :
        
        # copy the maps so that subsequent versions aren't erroneously used
        mass_map, sfr_map = tng_Mstar_map.copy(), tng_SFR_map.copy()
        
        # mask out pixels that aren't in the annulus
        mass_map[bins_image != val] = np.nan
        sfr_map[bins_image != val] = np.nan
        
        # sum values and place into array
        tng_annuli_masses[val] = np.nansum(mass_map) # solMass
        tng_annuli_sfrs[val] = np.nansum(sfr_map) # solMass/yr
    
    # check the integrated stellar masses and SFRs
    # tng_integrated_mass = np.log10(np.sum(tng_annuli_masses))
    # fast_integrated_mass = np.log10(np.sum(np.power(10, lmass)))
    # print(tng_integrated_mass, fast_integrated_mass,
    #       np.abs(tng_integrated_mass - fast_integrated_mass))
    # tng_integrated_sfr = np.sum(tng_annuli_sfrs)
    # fast_integrated_sfr = np.sum(np.power(10, lsfr))
    # print(tng_integrated_sfr, fast_integrated_sfr,
    #       np.abs(tng_integrated_sfr - fast_integrated_sfr))
    
    # find the surface mass/SFR densities of the idealized TNG maps
    tng_Sigma = np.log10(tng_annuli_masses/physical_area.value)
    tng_Sigma_SFR = np.log10(tng_annuli_sfrs/physical_area.value)
    
    # set the uncertainty for the TNG values
    zeros = np.zeros_like(tng_Sigma)
    
    # convert stellar masses to surface mass densities
    Sigma = np.log10(np.power(10, lmass)/physical_area.value)
    Sigma_lo = np.log10(np.power(10, lmass_lo)/physical_area.value)
    Sigma_hi = np.log10(np.power(10, lmass_hi)/physical_area.value)
    
    # convert star formation rates to surface star formation rate densities
    Sigma_SFR = np.log10(np.power(10, lsfr)/physical_area.value)
    Sigma_SFR_lo = np.log10(np.power(10, lsfr_lo)/physical_area.value)
    Sigma_SFR_hi = np.log10(np.power(10, lsfr_hi)/physical_area.value)
    
    # set plot attributes
    xlabel = r'$R/R_{\rm e}$' # r'$\log{(R/R_{\rm e})}$'
    ylabel1 = r'$\log{(\Sigma_{*}/{\rm M}_{\odot}~{\rm kpc}^{-2})}$'
    ylabel2 = r'$\log{(\Sigma_{\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})}$'
    
    # plot the radial profiles
    xs = [rr/Re, rr/Re, rr/Re, rr/Re]
    ys = [Sigma, tng_Sigma, Sigma_SFR, tng_Sigma_SFR]
    lo = [Sigma - Sigma_lo, zeros, Sigma_SFR - Sigma_SFR_lo, zeros]
    hi = [Sigma_hi - Sigma, zeros, Sigma_SFR_hi - Sigma_SFR, zeros]
    labels = ['fit', 'TNG', '', '']
    colors = ['k', 'k', 'b', 'b']
    markers = ['', '', '', '']
    styles = ['--', '-', '--', '-']
    
    plt.plot_multi_vertical_error(xs, ys, lo, hi, labels, colors, markers,
        styles, 2, xlabel=xlabel, ylabel1=ylabel1, ylabel2=ylabel2,
        xmin=-0.1, #xmax=8, ymin1=7.0, ymax1=10.6, ymin2=-4.5, ymax2=-0.5,
        save=False, outfile='fitting/c12_updated-expression.png')
    
    return

def compare_sfhs(subID, snap, population='quenched') :
    
    # get the scalefactors and ages for interpolation
    scalefactor, age = get_ages_and_scalefactors()
    
    # get galaxy attributes from TNG
    time, mpbsubID, logM, tngRe, center = load_galaxy_attributes(subID, snap)
    times, SFH = load_galaxy_sfh(subID)
    
    # get the annuli map, derived using SourceXtractor++ morphological values
    bins_image, dim, numBins = load_annuli_map(population, subID)
    # plt.display_image_simple(bins_image, lognorm=False)
    
    # define the edges which match the synthetic images
    edges = np.linspace(-10*tngRe, 10*tngRe, dim + 1) # kpc
    
    # open the TNG cutout and retrieve the relevant information
    cutout_file = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, mpbsubID)
    with h5py.File(cutout_file, 'r') as hf :
        coords = hf['PartType4']['Coordinates'][:]
        ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        Mstar = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h # solMass
    
    # limit particles to those that have positive formation times
    mask = (ages > 0)
    coords, ages, Mstar = coords[mask], ages[mask], Mstar[mask]
    
    # don't project using face-on version
    dx, dy, _ = (coords - center).T # kpc
    
    hh, _, _ = np.histogram2d(dx, dy, bins=(edges, edges))
    hh = hh.T
    hh = np.rot90(hh, k=3)
    # plt.display_image_simple(hh)
    
    if not exists('{}_bin_sfhs.npz'.format(subID)) :
        maxi = int(np.max(hh))
        ages_3d_histogram = np.full((dim, dim, maxi), np.nan)
        masses_3d_histogram = np.full((dim, dim, maxi), np.nan)
        for i in range(dim) :
            for j in range(dim) :
                valid = np.where((edges[i] <= dx) & (dx < edges[i + 1]) &
                                 (edges[j] <= dy) & (dy < edges[j + 1]))[0]
                length = int(len(valid))
                if length > 0 :
                    ages_3d_histogram[i, j, :] = np.concatenate(
                        [ages[valid], np.full(maxi - length, np.nan)])
                    masses_3d_histogram[i, j, :] = np.concatenate(
                        [Mstar[valid], np.full(maxi - length, np.nan)])
        np.savez('{}_bin_sfhs.npz'.format(subID), ages=ages_3d_histogram,
                masses=masses_3d_histogram)
    else :
        with np.load('{}_bin_sfhs.npz'.format(subID)) as data :
            ages_3d_histogram = data['ages']
            masses_3d_histogram = data['masses']
        print('Arrays loaded.')
    
    # define the SFH time bin edges
    bins = times[snap] - np.array([times[snap], 3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, 0.0])
    centers = (bins[:-1] + bins[1:])/2
    delta_t = (bins[1:] - bins[:-1])*1e9 # Gyr
    
    # loop through the elliptical annuli, binning together valid pixels
    SFHs = []
    for val in range(numBins) :
        
        # copy the maps so that subsequent versions aren't erroneously used, and
        # mask out pixels that aren't in the annulus
        ages_2d = ages_3d_histogram.copy()
        ages_2d[bins_image != val] = np.nan
        
        masses_2d = masses_3d_histogram.copy()
        masses_2d[bins_image != val] = np.nan
        
        # flatten the arrays and mask NaNs
        ages_per_bin = ages_2d.flatten()
        ages_per_bin = ages_per_bin[~np.isnan(ages_per_bin)]
        
        masses_per_bin = masses_2d.flatten()
        masses_per_bin = masses_per_bin[~np.isnan(masses_per_bin)]
        
        # convert the scalefactor ages to real ages
        ages_per_bin = np.interp(ages_per_bin, scalefactor, age)
        
        # histogram the data
        hist, _ = np.histogram(ages_per_bin, bins=bins, weights=masses_per_bin)
        SFHs.append(hist/delta_t)
    
    plt.plot_sfhs_comparison(times, gaussian_filter1d(SFH, 2), times[snap],
        SFHs, centers, xmin=0, xmax=14, xlabel=r'$t$ (Gyr)',
        ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)')
    
    return

def determine_bc03_model_units() :
    
    # read the model spectra from Bruzual & Charlot (2003) for a Chabrier (2003)
    # initial mass function with solar metallicity
    table = Table.read('tools/bc03_lr_ch_z02.ised_ASCII.fits')[0]
    # ages = table['AGE']/1e6   # (221), Myr
    # masses = table['MASS']    # (221), Msol
    # waves = table['LAMBDA']   # (1221), angstrom
    seds = table['SED']         # (221, 1221), Lsol AA^-1
    
    gpl = np.loadtxt('tools/bc2003_lr_m62_chab_ssp.Lsol_per_AA', skiprows=6)
    # waves_gpl = gpl[:, 0] # (1221), angstrom
    sed_1_Myr = gpl[:, 1]   # (1221), Lsol AA^-1
    sed_10_Myr = gpl[:, 2]  # (1221), Lsol AA^-1
    sed_100_Myr = gpl[:, 3] # (1221), Lsol AA^-1
    sed_1_Gyr = gpl[:, 4]   # (1221), Lsol AA^-1
    
    # check the ratios for each example age
    plt.histogram(sed_1_Myr/seds[19], '1 Myr ratio', bins=30)
    plt.histogram(sed_10_Myr/seds[70], '10 Myr ratio', bins=30)
    plt.histogram(sed_100_Myr/seds[116], '100 Myr ratio', bins=30)
    plt.histogram(sed_1_Gyr/seds[136], '1 Gyr ratio', bins=30)
    
    # print(seds[19][:221])  # starts on line 5218 in bc2003_lr_m62_chab_ssp.ised_ASCII
    # print(seds[70][221:])  # starts on line 18658
    # print(seds[116][221:]) # starts on line 30994
    # print(seds[136][221:]) # starts on line 36774
    
    return

def get_ages_and_scalefactors() :
    
    # look-up table for converting scalefactors to cosmological ages
    with h5py.File('tools/scalefactor_to_Gyr.hdf5', 'r') as hf :
        scalefactor, age = hf['scalefactor'][:], hf['age'][:]
    
    return scalefactor, age

def load_annuli_map(population, subID) :
    
    # open the annuli map to use for masking
    infile = 'bins/{}/subID_{}_annuli.npz'.format(population, subID)
    bin_data = np.load(infile)
    
    bins_image = bin_data['image']
    numBins = int(np.nanmax(bins_image) + 1) # accounts for python 0-index
    
    return bins_image, bins_image.shape[0], numBins

def load_galaxy_attributes(subID, snap) :
    
    infile = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Re = hf['Re'][:]
        centers = hf['centers'][:]
    
    # find the location of the subID within the entire sample
    loc = np.where(subIDfinals == subID)[0][0]
    
    return (times[snap], subIDs[loc, snap], logM[loc, snap], Re[loc, snap],
            list(centers[loc, snap]))

def load_galaxy_sfh(subID) :
    
    infile = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        SFHs = hf['SFH'][:]
    
    # find the location of the subID within the entire sample
    loc = np.where(subIDfinals == subID)[0][0]
    
    return times, SFHs[loc]

def load_photometric_information(population, subID, realizations=False) :
    
    # get information that was put into FAST++ about the elliptical annuli
    if realizations :
        infile = 'photometry/{}/subID_{}_0_photometry.fits'.format(population, subID)
    else :
        infile = 'photometry/{}/subID_{}_photometry.fits'.format(population, subID)
    table = Table.read(infile)
    
    nPixels = table['nPixels'].data*u.pix
    redshift = table['z'].data[0]
    scale = table['scale'].data[0]*u.arcsec/u.pix # arcsec/pixel
    
    # define the centers of the elliptical annuli for plotting purposes
    # rr = (table['sma'].data - 0.5*table['width'].data)*u.pix # pixels
    rr = (table['sma'].data - table['width'].data)*u.pix # pixels
    
    # get the half light radius
    Re = table['R_e'].data*u.pix
    
    return nPixels.astype(int), redshift, scale, rr, Re

# compare_fast_output_to_tng(96771, 44, display=True)
# compare_fast_output_to_tng(63871, 44, display=True)
# compare_fast_output_to_tng(198186, 53, display=True)

def check_tng_sfrs_using_fast_edges() :
    
    if not exists('tools/TNG_SFRs_with_FAST_edges.hdf5') :
        # get the scalefactors and ages for interpolation
        scalefactor, age = get_ages_and_scalefactors()
        
        table = Table.read('tools/subIDs.fits')
        subIDs = table['subID'].data
        snaps = table['snap'].data
        # subIDs_at_snaps = table['subID_at_snap'].data
        
        SFRs = np.full((len(table), 5), 0.0)
        for i, (subID, snap) in enumerate(zip(subIDs, snaps)) :
            SFRs[i] = get_tng_sfrs_using_fast_edges(subID, snap, scalefactor, age)
            print(i)
        
        with h5py.File('tools/TNG_SFRs_with_FAST_edges.hdf5', 'w') as hf :
            hf['SFRs'] = SFRs
    else :
        with h5py.File('tools/TNG_SFRs_with_FAST_edges.hdf5', 'r') as hf :
            SFRs = hf['SFRs'][:]
        
        SFRs = np.log10(SFRs, out=-4*np.ones_like(SFRs), where=(SFRs != 0.0))
        
        # make a histogram to visually see the distribution of the values
        data = [SFRs[:, 0], SFRs[:, 1], SFRs[:, 2], SFRs[:, 3], SFRs[:, 4]]
        hlabel = r'$\log{({\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1})}$'
        colors = ['k', 'r', 'g', 'b', 'm']
        styles = ['-.']*5
        labels = [r'$t > 3~{\rm Gyr}$', r'$1 < t/{\rm Gyr} < 3$',
                  r'$0.3 < t/{\rm Gyr} < 1$', r'$0.1 < t/{\rm Gyr} < 0.3$',
                  r'$ t < 0.1~{\rm Gyr}$']
        bins = [np.arange(-4, 2 + 1/3, 1/3)]*5
        
        plt.histogram_multi(data[:2], hlabel, colors[:2], styles[:2],
            labels[:2], bins[:2], xmin=-4.2, xmax=2.5, ymin=0, ymax=75, loc=2)
        plt.histogram_multi(data[2:], hlabel, colors[2:], styles[2:],
            labels[2:], bins[2:], xmin=-4.2, xmax=2.5, ymin=0, ymax=75, loc=2)
        
        simple = ['t > 3 Gyr', '1 < t/Gyr < 3', '0.3 < t/Gyr < 1',
                  '0.1 < t/Gyr < 0.3', 't < 0.1 Gyr']
        for i in range(5) :
            percentiles = np.percentile(SFRs[:, i],
                [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
            np.set_printoptions(precision=4)
            print('{:<17} {}'.format(simple[i], percentiles))
    
    return

def get_tng_sfrs_using_fast_edges(subID, snap, scalefactor, age) :
    
    # get galaxy attributes from TNG
    time, mpbsubID, logM, tngRe, center = load_galaxy_attributes(subID, snap)
    
    # formation ages in units of scalefactor
    formation_ages, masses, rs = get_particles(snap, mpbsubID, center)
    
    # convert the scalefactor ages to real ages
    formation_ages = np.interp(formation_ages, scalefactor, age) # Gyr, 0 = Big Bang
    
    # find the maximum time to set the edges of the SFH
    # based on https://stackoverflow.com/questions/58065055
    # max_time = np.true_divide(np.ceil(time*10), 10)
    
    # check the SFH to ensure that times are as expected
    # SFH_edges = np.arange(0, max_time + 0.1, 0.1) # Gyr
    # SFH = np.zeros(len(SFH_edges) - 1)
    # for i, (lo, hi) in enumerate(zip(SFH_edges, SFH_edges[1:])) :
    #     mask = (rs <= 2*tngRe) & (formation_ages > lo) & (formation_ages <= hi)
    #     SFH[i] = np.sum(masses[mask])/((hi - lo)*1e9)
    
    # plt.plot_simple_dumb(SFH_edges[:-1] + 0.05, gaussian_filter1d(SFH, 2),
    #                      xmin=0, xmax=14)
    
    FAST_edges = time - np.array([time, 3, 1, 0.3, 0.1, 0.0]) # Gyr
    FAST_SFH = np.zeros(len(FAST_edges) - 1)
    for i, (lo, hi) in enumerate(zip(FAST_edges, FAST_edges[1:])) :
        mask = (rs <= 10*tngRe) & (formation_ages > lo) & (formation_ages <= hi)
        FAST_SFH[i] = np.sum(masses[mask])/((hi - lo)*1e9)
    
    return FAST_SFH

'''
data = np.loadtxt('fitting/c12_with-better-normalization.fout', dtype=str)
ncols = data.shape[1]
check_val = float(data[0, 7])

# define which rows to use, based on the 'binNum' containing the subID
ids = data[:, 0]
use = np.char.find(ids, str(96771))
use[use < 0] = 1
use = np.invert(use.astype(bool))

lage = data[:, 7].astype(float)[use]
r0 = data[:, 25].astype(float)[use]
ltau = data[:, 28].astype(float)[use]
lscale = data[:, -3].astype(float)[use]

arrays = []
labels = []
max_age = cosmo.age(0.5).value*1e9
times = np.linspace(0, max_age, 10001)
for i in range(len(data)) :
    delayedtau = (times < max_age - 1e8).astype(float)
    burst = (times >= max_age - 1e8).astype(float)
    SFRs = (delayedtau*times/np.power(10, ltau[i])*np.exp(-times/np.power(10, ltau[i])) +
            burst*times*np.power(10, r0[i]))
    arrays.append(SFRs)
    labels.append('bin {}'.format(i))
colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'darkred', 'royalblue',
          'darkgreen', 'deeppink', 'teal']
markers = ['']*12
styles = ['-']*6 + ['--']*6
alphas = np.ones(12)

plt.plot_simple_multi([times/1e9]*12, arrays, labels, colors, markers, styles, alphas)
'''

'''
from astropy.io import fits
from matplotlib import cm

subID = 96771
inDir = 'SKIRT/SKIRT_output_quenched_redshift-0.5_3-CASTOR-bands-no-Euclid/'
withDust_file = inDir + '{}/{}_HST_total.fits'.format(subID, subID)
noDust_file = inDir + '{}/NoMedium/{}_NoMedium_HST_total.fits'.format(subID, subID)

filt = 8 # 8 for F555W

with fits.open(withDust_file) as hdu :
    plate_scale = (hdu[0].header)['CDELT1']*u.arcsec/u.pix
    dusted = (hdu[0].data*u.MJy/u.sr)[filt]

# plt.display_image_simple(f555w.data)

# get the area of a pixel on the sky, in arcsec^2
pixel_area = np.square(plate_scale*u.pix)

# check the brightness of the galaxy
# print(np.sum(f555w*pixel_area).to(u.Jy))
# m_AB = -2.5*np.log10(np.sum(f555w*pixel_area).to(u.Jy)/(3631*u.Jy))*u.mag

sb = -2.5*np.log10((dusted*pixel_area).to(u.Jy)/(3631*u.Jy))

with fits.open(noDust_file) as hdu :
    dustless = (hdu[0].data*u.MJy/u.sr)[filt]

sb_dustless = -2.5*np.log10((dustless*pixel_area).to(u.Jy)/(3631*u.Jy))

# plt.display_image_simple(sb_dustless, lognorm=False, cmap=cm.Greys, vmin=24, vmax=33)
# plt.display_image_simple(sb, lognorm=False, cmap=cm.Greys, vmin=24, vmax=33)

sb_diff = (sb - sb_dustless).value # large positive numbers are dimmer in the extincted image

plt.display_image_simple(sb_diff, lognorm=False, cmap=cm.RdBu_r)

infile = 'bins/quenched/subID_{}_annuli.npz'.format(subID)
bin_data = np.load(infile)
bins_image = bin_data['image']
numBins = int(np.nanmax(bins_image) + 1) # accounts for python 0-index

Av = np.full(numBins, np.nan)
for val in range(numBins) :
    temp_dust = dusted.copy()
    temp_dustless = dustless.copy()
    
    temp_dust[bins_image != val] = np.nan
    temp_dustless[bins_image != val] = np.nan
    
    dust_mag = -2.5*np.log10((np.nansum(temp_dust)*pixel_area).to(u.Jy)/(3631*u.Jy))
    dustless_mag = -2.5*np.log10((np.nansum(temp_dustless)*pixel_area).to(u.Jy)/(3631*u.Jy))
    
    mag_diff = dust_mag - dustless_mag
    
    if mag_diff >= 0.0 :
        Av[val] = mag_diff
    else :
        Av[val] = 0.0
Av = np.around(Av, 2) # round Av to 2 decimals

print(Av)
'''






def prepare_photometry_for_fitting() :
    
    processedDir = 'SKIRT/SKIRT_processed_images_quenched/'
    outDir = 'fastpp/'
    # import os
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

# from https://ned.ipac.caltech.edu/level5/March07/Ellis/Figures/figure16.jpg
# a 10^11 stellar mass galaxy should have a K-band apparent magnitude between
# roughly 16.5 and 17.5 AB mags

'''
# read the table
data = np.loadtxt('fastpp/96771_1000_sims.fout')
dim = np.sqrt(len(data)).astype(int)


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
'''

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
    
    # plot the bestfit constructed SED
    plt.plot_sed(waves[waves > 0.137], sed[waves > 0.137], pivots, models, phot, phot_e,
        # title='subID 198186 bin {}'.format(binNum),
        outfile='Suess+2019a_method/subID_198186_bin_0_SEDs.pdf', save=False)
    
    return

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

# print(get_filter_waves(['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g']))
# compare_output_and_constructed_sed(0, zz=0.5)

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

def check_fast_fits() :
    
    # from matplotlib import cm
    
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
