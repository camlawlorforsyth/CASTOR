
from os.path import exists
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import h5py
from photutils.aperture import CircularAnnulus, CircularAperture

from core import (calculate_distance_to_center, get_rotation_input,
                  load_massive_galaxy_sample)
from fitting import get_fastpp_profiles, get_tng_profiles
from fastpy import calzetti2000
from galaxev import determine_distance_from_SFMS, determine_dust_radial_profile
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)

def compare_fitted_to_simulated_values(model_redshift=0.5) :
    
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
    
    # get the subIDfinals for the entire massive sample
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDfinals = subIDfinals[hf['logM'][:, -1] >= 9.5] # select massive galaxies
    
    outfile = 'tools/comparisons_z_{:03}.hdf5'.format(
       str(model_redshift).replace('.', ''))
    
    if not exists(outfile) :
        # process every galaxy/snapshot pair
        tng_logMs = np.zeros(len(sample)*20)
        tng_SFRs = np.zeros_like(tng_logMs)
        fit_logMs = np.zeros_like(tng_logMs)
        fit_SFRs = np.zeros_like(tng_logMs)
        tng_metals = np.zeros_like(tng_logMs)
        fit_metals = np.zeros_like(tng_logMs)
        tng_dusts = np.zeros_like(tng_logMs)
        fit_dusts = np.zeros_like(tng_logMs)
        for i, (subIDfinal, subID, snap, Re) in enumerate(zip(sample['subIDfinal'],
            sample['subID'], sample['snapshot'], sample['Re'])) :
            
            # get the mass and SFR profiles from TNG and the fits
            tng_logM, tng_SFR, _ = get_tng_profiles(subIDfinal, snap,
                                                    surfacedensity=False)
            fit_logM, fit_SFR, _ = get_fastpp_profiles(subID, snap,
                model_redshift=model_redshift, surfacedensity=False)
            
            # find the location of the galaxy within the TNG massive sample
            loc = np.where(subIDfinals == subIDfinal)[0][0]
            
            # get the metallicity and dust profiles from TNG/GALAXEV
            _, tng_metal, _ = get_metallicity_profile(loc, snap)
            tng_dust = get_dust_profile(loc, snap, model_redshift=model_redshift)
            
            # get the fitted metallicity and dust profiles
            fit_metal, fit_dust = get_fastpp_metallicity_and_dust_profiles(subID, snap)
            
            # insert the values into the array for all values
            tng_logMs[i*20:i*20+20] = np.log10(tng_logM)
            tng_SFRs[i*20:i*20+20] = tng_SFR
            tng_metals[i*20:i*20+20] = tng_metal
            tng_dusts[i*20:i*20+20] = tng_dust
            fit_logMs[i*20:i*20+20] = np.log10(fit_logM)
            fit_SFRs[i*20:i*20+20] = fit_SFR
            fit_metals[i*20:i*20+20] = fit_metal
            fit_dusts[i*20:i*20+20] = fit_dust
            print('snap {} subID {} done'.format(snap, subID))
        
        # save the data
        with h5py.File(outfile, 'w') as hf :
            hf.create_dataset('tng_logM', data=tng_logMs)
            hf.create_dataset('tng_SFR', data=tng_SFRs)
            hf.create_dataset('tng_metal', data=tng_metals)
            hf.create_dataset('tng_dust', data=tng_dusts)
            hf.create_dataset('fit_logM', data=fit_logMs)
            hf.create_dataset('fit_SFR', data=fit_SFRs)
            hf.create_dataset('fit_metal', data=fit_metals)
            hf.create_dataset('fit_dust', data=fit_dusts)
    else :
        with h5py.File(outfile, 'r') as hf :
            tng_logM = hf['tng_logM'][:]
            tng_SFR = hf['tng_SFR'][:]
            tng_metal = hf['tng_metal'][:]
            tng_dust = hf['tng_dust'][:]
            fit_logM = hf['fit_logM'][:]
            fit_SFR = hf['fit_SFR'][:]
            fit_metal = hf['fit_metal'][:]
            fit_dust = hf['fit_dust'][:]
    
    maxR = 5
    dists = np.array([np.arange(0.125, 5, 0.25)]*len(sample)).flatten()
    mask = (dists <= maxR)
    
    select = np.isfinite(tng_logM) & np.isfinite(fit_logM) & (dists <= maxR)
    plt.plot_scatter_dumb(tng_logM[select], fit_logM[select], dists[select], 'logM', 'o',
        xmin=6, xmax=11, ymin=6, ymax=11,
        vmin=0, vmax=5, cbar_label='R/Re', xlabel='TNG logM',
        ylabel='FAST++ logM')
    
    tng_SFR[tng_SFR == 0] = 1e-4
    fit_SFR[fit_SFR < 1e-4] = 1e-4
    
    plt.plot_scatter_dumb(np.log10(tng_SFR)[mask], np.log10(fit_SFR)[mask], dists[mask], 'SFR', 'o',
        xmin=-4, xmax=1, ymin=-4, ymax=1,
        vmin=0, vmax=5, cbar_label='R/Re', xlabel='TNG log(SFR)',
        ylabel='FAST++ log(SFR)')
    
    np.random.seed(0)
    
    plt.plot_scatter_dumb(np.log10(tng_metal)[mask],
        np.random.normal(np.log10(fit_metal), 0.03)[mask], dists[mask], r'$Z$', 'o',
        xmin=-4.1, xmax=0, ymin=-4.1, ymax=0,
        vmin=0, vmax=5, cbar_label='R/Re', xlabel='TNG median log(metallicity)',
        ylabel='FAST++ log(metallicity)')
    
    plt.plot_scatter_dumb(tng_dust[mask], np.random.normal(fit_dust, 0.03)[mask], dists[mask],
        r'$\langle {\rm A}_{\rm V} \rangle$', 'o',
        xmin=-0.1, xmax=2.6, ymin=-0.1, ymax=2.6,
        vmin=0, vmax=5, cbar_label='R/Re', xlabel='TNG average dust',
        ylabel='FAST++ average dust')
    
    return

def compute_all_dust_profiles(model_redshift=0.5) :
    
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
    
    # create an output file to store the metallicity profiles
    outfile = 'tools/TNG_massive_dust_datacube_z_{:03}.hdf5'.format(
        str(model_redshift).replace('.', ''))
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            hf.create_dataset('Av', data=np.zeros((1666, 100, 20)))
    
    # get the subIDfinals for the entire massive sample
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDfinals = subIDfinals[hf['logM'][:, -1] >= 9.5] # select massive galaxies
    
    # read the output file to ensure we're not over-writing any profiles
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            Av = hf['Av'][:]
    
    # process every galaxy/snapshot pair and store the values into an array
    for subIDfinal, snap, subID, logM, SFR, Re in zip(sample['subIDfinal'],
        sample['snapshot'], sample['subID'], sample['logM'], sample['SFR'],
        sample['Re']) :
        
        # find the location of the galaxy within the TNG massive sample
        loc = np.where(subIDfinals == subIDfinal)[0][0]
        
        if (np.count_nonzero(Av[loc, snap]) == 0) :
            
            # get the average dust profile
            dust_profile = compute_dust_profile(snap, subID, logM, np.log10(SFR),
                Re, model_redshift=model_redshift)
            
            # place the profiles into the array
            with h5py.File(outfile, 'a') as hf :
                hf['Av'][loc, snap] = dust_profile
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def compute_all_metallicities() :
    
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
    
    # create an output file to store the metallicity profiles
    outfile = 'tools/TNG_massive_metallicity_datacube.hdf5'
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            hf.create_dataset('metallicity_16', data=np.zeros((1666, 100, 20)))
            hf.create_dataset('metallicity_50', data=np.zeros((1666, 100, 20)))
            hf.create_dataset('metallicity_84', data=np.zeros((1666, 100, 20)))
    
    # get the subIDfinals for the entire massive sample
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDfinals = subIDfinals[hf['logM'][:, -1] >= 9.5] # select massive galaxies
    
    # read the output file to ensure we're not over-writing any profiles
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            metallicity_16 = hf['metallicity_16'][:]
            metallicity_50 = hf['metallicity_50'][:]
            metallicity_84 = hf['metallicity_84'][:]
    
    # process every galaxy/snapshot pair and store the values into an array
    for subIDfinal, snap, subID, Re, center in zip(sample['subIDfinal'],
        sample['snapshot'], sample['subID'], sample['Re'], sample['center']) :
        
        # find the location of the galaxy within the TNG massive sample
        loc = np.where(subIDfinals == subIDfinal)[0][0]
        
        if ((np.count_nonzero(metallicity_16[loc, snap]) == 0) and
            (np.count_nonzero(metallicity_50[loc, snap]) == 0) and
            (np.count_nonzero(metallicity_84[loc, snap]) == 0)) :
            
            # get the profiles
            los, mes, his = compute_metallicity_profile(snap, subID, Re, center)
            
            # place the profiles into the array
            with h5py.File(outfile, 'a') as hf :
                hf['metallicity_16'][loc, snap] = los
                hf['metallicity_50'][loc, snap] = mes
                hf['metallicity_84'][loc, snap] = his
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def compute_dust_profile(snap, subID, logM, logSFR, Re, model_redshift=0.5,
                         fov=10) :
    
    # get the requisite information for creating a matched-size dust map
    infile = 'GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(
        snap, subID, str(model_redshift).replace('.', ''))
    with fits.open(infile) as hdu :
        infile_redshift = hdu[0].header['REDSHIFT']
        shape = (hdu[0].data.shape[1], hdu[0].data.shape[2])
        plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix
    assert infile_redshift == model_redshift
    
    # determine the center of the image
    cent = int((shape[1] - 1)/2)
    center = (cent, cent)
    
    # convert Re into pixels
    Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    
    # get the edges of the circular annuli in units of pixels for masking
    edges_pix = np.linspace(0, 5, 21)*Re_pix # edges in units of Re
    
    # next create the dust map as a function of spatial position, stellar mass,
    # and distance from the star forming main sequence
    dist_from_SFMS = determine_distance_from_SFMS(snap, logM, logSFR)
    dists = calculate_distance_to_center(shape)/(shape[0]/fov)
    dust_map = determine_dust_radial_profile(logM, dist_from_SFMS, dists) # [Av]
    
    # determine the average dust profile
    dust_profile = np.full(20, -1.0)
    for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
        if start == 0 :
            ap = CircularAperture(center, end)
        else :
            ap = CircularAnnulus(center, start, end)
        dust_profile[i] = ap.do_photometry(dust_map)[0][0]/ap.area
    
    return dust_profile

def compute_metallicity_profile(snap, subID, Re, center) :
    
    # get the stellar particle positions and metallicties
    _, _, _, _, _, _, star_coords, star_metals = get_rotation_input(snap, subID)
    
    # get the 2D projected distances to each stellar particle
    rs = np.sqrt(np.square(star_coords[:, 0] - center[0]) +
                 np.square(star_coords[:, 1] - center[1]))
    
    # define the edges of the radial bins
    edges = np.arange(0, 5.25, 0.25)*Re # kpc
    
    # determine the metallicity profile
    los = np.zeros(20)
    mes = np.zeros(20)
    his = np.zeros(20)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) :
        if lo == 0. :
            mask = (rs >= lo) & (rs <= hi)
        else :
            mask = (rs > lo) & (rs <= hi)
        
        if np.sum(mask) > 0 :
            los[i], mes[i], his[i] = np.percentile(star_metals[mask], [16, 50, 84])
        else :
            los[i], mes[i], his[i] = np.nan, np.nan, np.nan
    
    return los, mes, his

def get_fastpp_metallicity_and_dust_profiles(subID, snap, skiprows=18) :
    
    # load fitted data coming out of FAST++
    data = np.loadtxt('fits/fits_2April2025.fout', dtype=str, skiprows=skiprows)
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, :2].astype(int)
    use = (ids[:, 0] == snap) & (ids[:, 1] == subID)
    use[np.where(use)[0][-2:]] = False # account for 1 kpc and integrated bins
    
    # get the metallicities and extinction values
    metallicity_profile = data[:, 2].astype(float)[use]
    dust_profile = data[:, 4].astype(float)[use]
    
    return metallicity_profile, dust_profile

def get_dust_profile(loc, snap, model_redshift=0.5) :
    
    # get the determined average dust profiles based on the GALAXEV pipeline
    with h5py.File('tools/TNG_massive_dust_datacube_z_{:03}.hdf5'.format(
        str(model_redshift).replace('.', '')), 'r') as hf :
        Av = hf['Av'][loc, snap]
    
    return Av

def get_metallicity_profile(loc, snap) :
    
    # get the raw metallicity profiles from TNG
    with h5py.File('tools/TNG_massive_metallicity_datacube.hdf5', 'r') as hf :
        metallicity_16 = hf['metallicity_16'][loc, snap]
        metallicity_50 = hf['metallicity_50'][loc, snap]
        metallicity_84 = hf['metallicity_84'][loc, snap]
    
    return metallicity_16, metallicity_50, metallicity_84
