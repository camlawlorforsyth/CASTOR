
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from photutils.aperture import CircularAnnulus, CircularAperture
from scipy.optimize import curve_fit

from core import get_rotation_input, load_galaxy_attributes_massive, open_cutout
from fastpy import (calculate_chi2, calzetti2000, dtt_from_fit, get_bc03_waves,
                    get_filter_waves, get_lum2fl, get_model_fluxes, get_times,
                    sfh2sed_fastpp)
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)

def check_integrated_mass_and_metallicity(subID) :
    
    # define the subIDs and the relevant snapshot that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, snaps = table['subID'].data, table['snapshot'].data
    
    # find the snapshot for the subID of interest
    loc = np.where(subIDs == subID)[0][0]
    
    # load relevant properties for the galaxy of interest
    _, mpbsubID, _, Re, center = load_galaxy_attributes_massive(subID, snaps[loc])
    
    # get all particles, noting that star ages is in units of scale factor
    _, _, _, _, _, star_masses, star_coords, star_metals = get_rotation_input(
        snaps[loc], mpbsubID)
    
    # get the 2D projected distances to each stellar particle
    rs = np.sqrt(np.square(star_coords[:, 0] - center[0]) +
                 np.square(star_coords[:, 1] - center[1]))
    
    # get the median and average metallicities for star particles within the
    # integrated aperture, along with the integrated (present day) mass
    mask = (rs <= 5*Re)
    plt.histogram(star_metals[mask], 'metallicity', bins=50)
    print(np.median(star_metals[mask]), np.mean(star_metals[mask]))
    print(np.log10(np.sum(star_masses[mask])))
    
    return

def check_bin_results(subID, check_chi2=False, plot=False) :

    # check on bin 4 first
    binNum = 0
    
    # get the bestfit results as directly output by FAST++ from the bestfit table
    # inDir = 'extra_metals_dtt_all_revisedR_withBurst/'
    # inDir = 'wide_parameter_space_26Feb2025_subID_63871/'
    inDir = '63871_default/'
    data = np.loadtxt(inDir + 'photometry_23November2024_new.fout',
                      dtype=str, skiprows=18)
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, 0].astype(int)
    use = (ids == subID)
    use[np.where(use)[0][-2:]] = False # account for 1 kpc and integrated bins
    
    tab = data[:, 1:].astype(float)[use]
    tab = tab[binNum]
    zz_tab, metal_tab, lage_tab, dust_tab = tab[0], tab[1], tab[2], tab[3]
    lmass_tab, rr_tab, ltau_tab = tab[4], tab[7], tab[8]
    
    # get the model and observed photometry as directly output by FAST++
    infile = 'photometry_23November2024_new_63871_bin_{}.input_res.fit'.format(binNum)
    pivots, model, phot, phot_e = np.loadtxt(inDir + 'best_fits/' + infile, unpack=True)
    pivots /= 1e4 # angstrom to micron
    # print(np.sum(np.square((phot - model)/phot_e)))
    
    # get the bestfit spectrum as directly output by FAST++
    infile = 'photometry_23November2024_new_63871_bin_{}.fit'.format(binNum)
    _, fl = np.loadtxt(inDir + 'best_fits/' + infile, unpack=True)
    
    # get the bestfit SFH as directly output by FAST++
    infile = 'photometry_23November2024_new_63871_bin_{}.sfh'.format(binNum)
    _, tmp = np.loadtxt(inDir + 'best_fits/' + infile, unpack=True)
    
    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(zz_tab) # Myr
    
    # get the library wavelengths for the constructed SED
    waves = get_bc03_waves(metal_tab)
    
    # construct the SED from the bestfit SFH
    sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, rr_tab, norm=np.sum(1e6*tmp)) # [Msol/Myr]
    # plt.plot_simple_multi([_/1e6, ts], [tmp, sfh/1e6], ['FAST++ output', 'constructed'],
    #                       ['k', 'r'], ['', ''], ['-', '-'], [1, 1], xmin=7000)
    sed = sfh2sed_fastpp(metal_tab, tobs, ts, sfh) # [Lsol AA^-1]
    sed *= np.power(10, -0.4*dust_tab*calzetti2000(waves)) # correct the SED for dust
    sed *= get_lum2fl(zz_tab) # convert luminosity to flux -> [10^-19 ergs s^-1 cm^-2 AA^-1]
    
    plt.plot_seds(waves*1.5, [sed], pivots, model, phot, phot_e, ymax=3,
                  title='central bin (bin 0) for subID 63871')
    
    grid = Table.read('63871_default_bin_0/chi2.grid.fits')[0]
    
    chi2 = grid['CHI2'].T[0]
    # sort = np.argsort(chi2)
    select = (grid['SFR100'] >= 0.71) & (grid['SFR100'] <= 0.73) & (grid['METAL'] == 0.05) #& (grid['LOG_TAU'] == 8.8)
    select = select.T[0]
    
    sort = np.argsort(chi2[select])
    
    metals = (grid['METAL'].T[0])[select]
    masses = (grid['LMFORM'].T[0])[select]
    ages = (grid['LAGE'].T[0])[select]
    taus = (grid['LOG_TAU'].T[0])[select]
    rs = (grid['R'].T[0])[select]
    Avs = (grid['AV'].T[0])[select]
    
    metals = metals[sort][:10]
    masses = masses[sort][:10]
    ages = ages[sort][:10]
    taus = taus[sort][:10]
    rs = rs[sort][:10]
    Avs = Avs[sort][:10]
    
    # idx = [0, 1, 14, 268] # for subID 324125
    
    seds = []
    for metal, mass, la, lt, RR, dust in zip(metals, masses, ages, taus, rs, Avs) :
        sfh = 1e6*dtt_from_fit(ts, lt, la, RR, norm=mass)
        sed = sfh2sed_fastpp(metal, tobs, ts, sfh)*np.power(10, -0.4*dust*calzetti2000(waves))*get_lum2fl(0.5)
        seds.append(sed)
    plt.plot_seds(waves*1.5, seds, pivots, np.full(9, np.nan), phot, phot_e, ymax=3,
                  title='central bin (bin 0) for subID 63871')
    
    # now check to see what the resulting SED would look like if we use the
    # dust free image of subID 63871 produced by SKIRT and apply the Calzetti
    # dust law to that photometry
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    if plot : # compare the SEDs for the direct output with the constructed
        plt.plot_seds(waves*1.5, [sed, fl], pivots, model, phot, phot_e)
    
    if plot : # construct the SED from the SFH for bin 4 using the different available metallicities
        sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, rr_tab, norm=np.sum(1e6*tmp)) # [Msol/Myr]
        seds = []
        for metallicity in [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05] :
            sed = sfh2sed_fastpp(metallicity, tobs, ts, sfh) # [Lsol AA^-1]
            sed *= np.power(10, -0.4*dust_tab*calzetti2000(waves)) # correct the SED for dust
            sed *= get_lum2fl(zz_tab) # convert luminosity to flux
            seds.append(sed) # [10^-19 ergs s^-1 cm^-2 AA^-1]
        plt.plot_seds(waves*(1 + zz_tab), seds, pivots, [np.nan]*7, phot, phot_e,
                      xmin=0.13, xmax=3, ymin=0.1, ymax=10)
    
    if check_chi2 :
        # compute by hand the model fluxes, using the direct output SED
        models = get_model_fluxes(['castor_uv', 'castor_u', 'castor_g', 'roman_f106',
            'roman_f129', 'roman_f158', 'roman_f184'], waves*1.5, fl)
        
        # check chi2 values using the direct output model fluxes and the by-hand version
        print(np.sum(np.square((phot - model)/phot_e)))
        print(np.sum(np.square((phot - models)/phot_e)))
    
    # get the grid of results, which includes every possible parameter combination
    grid = Table.read(inDir + 'chi2.grid.fits')[0]
    lage = grid['LAGE'][:, binNum]    # parameters which control the shape of the SFH, and
    ltau = grid['LOG_TAU'][:, binNum] # therefore ultimately the SED
    rr = grid['R'][:, binNum]
    lmass = grid['LMASS'][:, binNum]
    metal = grid['METAL'][:, binNum]  # parameters which control the shape of the SED
    dust = grid['AV'][:, binNum]
    redshift = grid['Z'][:, binNum]
    chi2 = grid['CHI2'][:, binNum]    # derived chi2 from the bestfit SED
    lsfr = grid['LSFR'][:, binNum]    # derived physical parameters - instantaneous SFR at t_lookback = 0
    sfr = grid['SFR100'][:, binNum]   # SFR average over past 100 Myr in lookback time
    
    # explore other grid space with the same lage, ltau, and metallicity
    # as the bestfit values
    mask = ((np.abs(lage - lage_tab) <= 0.01) &
            (np.abs(ltau - ltau_tab) <= 0.01) &
            (np.abs(metal - metal_tab) <= 0.01))
    sort = np.argsort(chi2[mask])
    print(chi2[mask][sort])
    
    seds = []
    print('dust  lmass  sfr100  rchi2')
    for lmass, dust, sfr, chi2, RR in zip(lmass[mask][sort][:], dust[mask][sort][:],
                                      sfr[mask][sort][:], chi2[mask][sort][:], rr[mask][sort][:]) :
        print('{:4.1f}  {:5.2f}  {:6.2f}  {:5.2f}'.format(dust, np.log10(lmass), sfr, chi2))
        # sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, RR, norm=np.power(10, 9.38))
        sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, rr_tab, norm=np.power(10, lmass_tab))
        sed = sfh2sed_fastpp(metal_tab, tobs, ts, sfh)
        sed *= np.power(10, -0.4*dust*calzetti2000(waves))
        sed *= get_lum2fl(zz_tab)
        seds.append(sed)
    plt.plot_seds(waves*(1 + zz_tab), seds, pivots, [np.nan]*7, phot, phot_e,
                  xmin=0.13, xmax=3, ymin=0.1, ymax=10)
    '''
    
    return

def compute_metallicity_per_bin(subID) :
    
    # define the subIDs and the relevant snapshot that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, snaps = table['subID'].data, table['snapshot'].data
    
    # find the snapshot for the subID of interest
    loc = np.where(subIDs == subID)[0][0]
    
    # load relevant properties for the galaxy of interest
    _, mpbsubID, _, Re, center = load_galaxy_attributes_massive(subID, snaps[loc])
    
    # get all particles, noting that star ages is in units of scale factor
    _, _, _, _, _, _, star_coords, star_metals = get_rotation_input(
        snaps[loc], mpbsubID)
    
    # get the 2D projected distances to each stellar particle
    rs = np.sqrt(np.square(star_coords[:, 0] - center[0]) +
                 np.square(star_coords[:, 1] - center[1]))
    
    # define the edges of the radial bins
    edges = np.arange(0, 5.25, 0.25)*Re # kpc
    
    Z_los, Z_mes, Z_his = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) :
        if lo == 0. :
            mask = (rs >= lo) & (rs <= hi)
        else :
            mask = (rs > lo) & (rs <= hi)
        
        # plt.histogram(star_metals[mask], r'$Z$', bins=50,
        #     title='bin {}'.format(i), save=False,
        #     outfile='subID_324125_bin_{}_metallicity.png'.format(i))
        
        six, fif, eig = np.percentile(star_metals[mask], [16, 50, 84])
        Z_los.append(six)
        Z_mes.append(fif)
        Z_his.append(eig)
    
    xs = np.linspace(0.125, 4.875, 20)
    ys = np.array(Z_mes)
    ylo = ys - np.array(Z_los)
    yhi = np.array(Z_his) - ys
    plt.plot_scatter_err_both(xs, ys, np.zeros_like(xs),
        np.zeros_like(xs), ylo, yhi, xmin=0, xmax=5, ymin=0,
        xlabel=r'$R/R_{\rm e}$', ylabel='$Z$')
    
    return

def compute_dust_per_bin(subID, model_redshift=0.5) :
    
    cutoutDir = 'cutouts/quenched/{}_noMedium/'.format(subID)
    
    # define the subIDs and the relevant information that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, Res = table['subID'].data, table['Re'].data
    
    # find the effective radius for the subID of interest
    loc = np.where(subIDs == subID)[0][0]
    Re = Res[loc]
    
    _, shape, _, _, _, _, plate_scale = open_cutout(
        cutoutDir + 'castor_uv_ultradeep.fits')
    shape = (35, 35)
    cent = int((shape[0] - 1)/2)
    center = (cent, cent)
    
    # convert Re, 1 kpc into pixels
    Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    kpc_pix = (1*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    
    # get the edges of the circular annuli in units of pixels for masking
    edges_pix = np.linspace(0, 5, 21)*Re_pix # edges in units of Re
    
    observs = ['GALEX', 'GALEX',
               'CASTOR', 'CASTOR', 'CASTOR', 'CASTOR', 'CASTOR',
               'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST',
               'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST', 'HST',
               'Av', 'Av', 'Av',
               'Roman', 'Roman', 'Roman', 'Roman', 'Roman', 'Roman', 'Roman', 'Roman',
               'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST',
               'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST', 'JWST',
               'Euclid', 'Euclid', 'Euclid', 'Euclid',
               'WISE', 'WISE', 'WISE', 'WISE',
               'Spitzer', 'Spitzer', 'Spitzer', 'Spitzer', 'Spitzer', 'Spitzer', 'Spitzer',
               'Herschel', 'Herschel', 'Herschel', 'Herschel', 'Herschel', 'Herschel']
    frames = [0, 1,
              0, 1, 2, 3, 4,
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              0, 1, 2,
              0, 1, 2, 3, 4, 5, 6, 7,
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
              0, 1, 2, 3,
              0, 1, 2, 3,
              0, 1, 2, 3, 4, 5, 6,
              0, 1, 2, 3, 4, 5]
    pivots = [0.153507951, 0.230078481, # GALEX
              0.226285654, 0.252405812, 0.32341151, 0.345079402, 0.474768346, # CASTOR
              0.222561574, 0.236455146, 0.270639705, 0.335457495, 0.392219682, # HST
              0.432568836, 0.432986835, 0.474695726, 0.536103019, 0.592189089,
              0.631184789, 0.769346747, 0.804552899, 0.903145766, 1.055104691,
              1.153445856, 1.248605979, 1.392290735, 1.536917571,
              0.547935189, 0.684918986, 0.821902783, # Av
              0.627504234, 0.871731516, 1.06176475, 1.284057751, 1.433831782, # Roman
              1.580001399, 1.838306984, 2.131095341,
              0.704711139, 0.902400432, 1.154078206, 1.500898649, 1.987600631, # JWST
              2.776229904, 3.56548902, 4.082988274, 4.401743343, 5.635010957,
              7.63926556, 9.952965258, 11.30866196, 12.81008243, 15.06351806,
              17.98389962, 20.79455325, 25.36497697,
              0.710337049, 1.078538774, 1.362067544, 1.764879018, # Euclid
              3.368224943, 4.617910561, 12.07302227, 22.19429619, # WISE
              3.550445449, 4.495318189, 5.724775374, 7.885304989, 23.75888699, # Spitzer
              71.98518038, 156.4274591,
              71.54149314, 102.0073134, 165.3561821, 250.4474421, 351.3685615, # Herschel
              507.8456954]
    
    ext_sum_profile = np.full(76, -1.0)
    nPixel_profile = np.full(76, -1.0)
    for i, (obs, frame) in enumerate(zip(observs, frames)) :
        # get the extincted image and the un-extincted image
        infile = '63871_050_noMedium/63871_z_050_noMedium_{}_total.fits'.format(obs)
        with fits.open(infile) as hdu :
            raw = hdu[0].data[frame]*u.MJy/u.sr
        infile = '63871_050_default/63871_z_050_{}_total.fits'.format(obs)
        with fits.open(infile) as hdu :
            ext = hdu[0].data[frame]*u.MJy/u.sr
        
        # create the magnitude image showing the effect of dust
        mag = -2.5*np.log10(ext/raw)
        mag[mag < 0] = 0
        
        # calculate the average extinction in the central bin
        ap = CircularAperture(center, edges_pix[1])
        ext_sum_profile[i] = ap.do_photometry(mag)[0]
        nPixel_profile[i] = ap.area
        
        # calculate the average extinction per annulus
        # patches = []
        # for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        #     if start == 0 :
        #         ap = CircularAperture(center, end)
        #         patches.append(ap)
        #     else :
        #         ap = CircularAnnulus(center, start, end)
        #         patches.append(ap)
        #     ext_sum_profile[i] = ap.do_photometry(mag)[0]
        #     nPixel_profile[i] = ap.area
    
    sort = np.argsort(pivots)
    ext_sum_profile = ext_sum_profile[sort]
    pivots = np.array(pivots)[sort]
    
    xx = np.linspace(0.12, 3.3, 1001)
    cal = calzetti2000(xx)
    cal[cal < 0] = 0
    
    ext_profile = ext_sum_profile/nPixel_profile
    plt.plot_simple_multi([pivots, xx*1.5],
        [ext_profile/ext_profile[25], cal],
        ['subID 63871 bin 0', 'Calzetti et al. (2000)'],
        ['k', 'k'], ['', ''], ['-', '--'], [1, 1],
        xlabel=r'wavelength ($\rm \mu m$)',
        ylabel=r'${\rm A}(\lambda)/{\rm A}_{\rm V}$',
        xmin=0.1, xmax=3.3, ymin=0, ymax=3,
        scale='linear')
    
    
    
    
    
    
    # complete an additional inner 1 kpc aperture
    # inner_kpc = CircularAperture(center, kpc_pix)
    # ext_sum_profile[20] = inner_kpc.do_photometry(mag)[0]
    # nPixel_profile[20] = inner_kpc.area
    
    # complete an additional integrated aperture out to 5 Re
    # integrated_ap = CircularAperture(center, edges_pix[-1])
    # ext_sum_profile[21] = integrated_ap.do_photometry(mag)[0]
    # nPixel_profile[21] = integrated_ap.area
    
    # plt.display_image_simple(mag.data, lognorm=False, vmin=0, vmax=3,
    #     title=r'$A_{\rm V}$ (mag)', patches=patches)
    
    # create an average extinction profile
    # ext_profile = ext_sum_profile/nPixel_profile
    # ext_profile = ext_profile[:-2]
    # xs = np.arange(0.125, 5, 0.25)
    
    # popt, _ = curve_fit(exponential, xs, ext_profile, p0=[3.222, -1.478, 0.01431])
    # xx = np.linspace(0, 5, 1000)
    # yy = exponential(3.222, -1.478, 0.01431, xx)
    
    # plt.plot_simple_multi([xs, xx], [ext_profile, yy], ['', 'fit'], ['k', 'k'],
    #     ['o', ''], ['', '-'], [1, 1],
    #     xlabel=r'$R/R_{\rm e}$', ylabel=r'$\langle A_{\rm V} \rangle$ (mag)',
    #     xmin=0, xmax=5, ymin=0, ymax=3, ms=5)
    
    
    
    return

def compare_mappings_with_bc03() :
    
    # get the CASTOR UV image using the MAPPINGS-III setup for young stellar populations
    infile = '63871_050_noMedium/63871_z_050_noMedium_CASTOR_total.fits'
    with fits.open(infile) as hdu :
        noMedium = hdu[0].data[0]*u.MJy/u.sr
    
    # get the CASTOR UV image using BC03 for everything
    infile = '63871_050_bc03noMedium/63871_z_050_bc03_CASTOR_total.fits'
    with fits.open(infile) as hdu :
        bc03noMedium = hdu[0].data[0]*u.MJy/u.sr
    
    # create the magnitude image showing the effect of using MAPPINGS-III
    mag = -2.5*np.log10(noMedium/bc03noMedium)
    
    from matplotlib import cm
    plt.display_image_simple(mag.data, lognorm=False, vmin=-1.3, vmax=1.3,
        cmap=cm.RdBu, title='CASTOR UV\nred = MAPPINGS brighter, blue = BC03 brighter')
    
    # repeat for the Roman F158 (H) band
    with fits.open('63871_050_noMedium/63871_z_050_noMedium_Roman_total.fits') as hdu :
        noMedium = hdu[0].data[5]*u.MJy/u.sr
    with fits.open('63871_050_bc03noMedium/63871_z_050_bc03_Roman_total.fits') as hdu :
        bc03noMedium = hdu[0].data[5]*u.MJy/u.sr
    mag = -2.5*np.log10(noMedium/bc03noMedium)
    plt.display_image_simple(mag.data, lognorm=False, vmin=-1.3, vmax=1.3, cmap=cm.RdBu,
        title='Roman F158 (H) band\nred = MAPPINGS brighter, blue = BC03 brighter')
    
    return

def exponential(aa, bb, cc, xx) :
    return aa*np.exp(bb*xx) + cc

# compute_metallicity_per_bin(63871) # 324125
compute_dust_per_bin(63871)
# compare_mappings_with_bc03()

check_bin_results(63871)

'''
print()
# check if we can find a better SFH normalization that fits the data better
# than the grid lmass values, using fixed dust and 
for dust, RR, chi2 in zip(dust_bin[mask][sort][:], rr_bin[mask][sort][:], chi2_bin[mask][sort][:]) :
    sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, RR, norm=np.sum(1e6*tmp))
    sed = sfh2sed_fastpp(metal_tab, tobs, ts, sfh)
    sed *= np.power(10, -0.4*dust*calzetti2000(waves))
    sed *= get_lum2fl(zz_tab)
    models = get_model_fluxes(['castor_uv', 'castor_u', 'castor_g', 'roman_f106',
        'roman_f129', 'roman_f158', 'roman_f184'], waves*1.5, sed) # get the model fluxes
    # print(models)
    # print(np.sum(np.square((phot - models)/phot_e)))


dust = dust_bin[mask][sort][0]
rr = rr_bin[mask][sort][0]
chi2 = chi2_bin[mask][sort][0]
for offset in np.arange(0, 0.31, 0.01) :
    sfh = 1e6*dtt_from_fit(ts, ltau_tab, lage_tab, 1, norm=np.power(10, 8.56+offset))
    sed = sfh2sed_fastpp(metal_tab, tobs, ts, sfh)
    sed *= np.power(10, -0.4*dust*calzetti2000(waves))
    sed *= get_lum2fl(zz_tab)
    models = get_model_fluxes(['castor_uv', 'castor_u', 'castor_g', 'roman_f106',
        'roman_f129', 'roman_f158', 'roman_f184'], waves, sed) # get the model fluxes
    # print(np.sum(np.square((phot - models)/phot_e)))
    
    chi2 = calculate_chi2(pivots, models, phot, phot_e)
    # print(chi2)
    # plt.plot_sed(waves*(1 + zz_tab), sed, pivots, models, phot, phot_e,
    #     xmin=0.13, xmax=3, ymin=0.1, ymax=10)

# plt.plot_sed(waves*(1 + zz_tab), sed, pivots, [np.nan]*7, phot, phot_e,
#              xmin=0.13, xmax=3, ymin=0.1, ymax=10)

'''






