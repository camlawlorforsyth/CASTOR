
from os.path import exists
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d

from core import get_particles
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
