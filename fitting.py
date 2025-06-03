
from os.path import exists
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import h5py

from core import load_galaxy_attributes_massive, load_massive_galaxy_sample
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def compare_all_fits(model_redshift=0.5) :
    
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
    for subIDfinal, subID, snap, logM, Re in zip(sample['subIDfinal'],
        sample['subID'], sample['snapshot'], sample['logM'], sample['Re']) :
        outfile = 'figures/radial_profiles_GALAXEV/{}_{}_z_{:03}.pdf'.format(
            snap, subID, str(model_redshift).replace('.', ''))
        if not exists(outfile) :
            compare_fits_to_tng(subIDfinal, snap, subID, logM, Re,
                model_redshift=model_redshift)
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def compare_fits_to_tng(subIDfinal, snap, subID, logM, Re, model_redshift=0.5) :
    
    # get the profiles from TNG
    tng_Sigma, tng_Sigma_SFR, tng_Sigma_sSFR = get_tng_profiles(
        subIDfinal, snap, Re=Re)
    
    # get the fitted profiles from FAST++, which uses a delayed tau model
    # with an additional burst/truncation in the final 100 Myr
    fast_Sigma, fast_Sigma_SFR, fast_Sigma_sSFR = get_fastpp_profiles(
        subID, snap, model_redshift=model_redshift)
    
    # get the SNR for the profiles from the photometry that was used in FAST++
    castor_avg_snr, roman_avg_snr = get_snr_profiles(subID, snap)
    
    # prepare quantities for plotting
    radial_bin_centers = np.linspace(0.125, 4.875, 20) # units of Re
    xs = np.array([radial_bin_centers, radial_bin_centers, radial_bin_centers,
                   radial_bin_centers, radial_bin_centers, radial_bin_centers])
    ys = np.array([tng_Sigma, fast_Sigma, tng_Sigma_SFR, fast_Sigma_SFR,
                   tng_Sigma_sSFR, fast_Sigma_sSFR])
    
    mass_vals = ys[:2].flatten()
    ymin1 = np.power(10, np.log10(np.nanmin(mass_vals)) - 0.1)
    ymax1 = np.power(10, np.log10(np.nanmax(mass_vals)) + 0.1)
    
    sfr_vals = ys[2:4].flatten()
    sfr_vals = sfr_vals[sfr_vals > 0] # mask out zeros for plotting
    ymin2 = np.power(10, np.log10(np.nanmin(sfr_vals)) - 0.1)
    ymax2 = np.power(10, np.log10(np.nanmax(sfr_vals)) + 0.1)
    
    ssfr_vals = ys[4:].flatten()
    ssfr_vals = ssfr_vals[ssfr_vals > 0] # mask out zeros for plotting
    ymin3 = np.power(10, np.log10(np.nanmin(ssfr_vals)) - 0.1)
    ymax3 = np.power(10, np.log10(np.nanmax(ssfr_vals)) + 0.1)
    
    # plot the radial profiles
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    labels = ['TNG', 'BC03+dust+noise+PSF', '', '', '', '']
    outfile = 'figures/radial_profiles_GALAXEV/{}_{}_z_{:03}.pdf'.format(
        snap, subID, str(model_redshift).replace('.', ''))
    plt.plot_radial_profiles(xs, ys, labels, castor_avg_snr, roman_avg_snr,
        xlabel=r'$R/R_{\rm e}$',
        ylabel1=r'$\Sigma_{*}/{\rm M}_{\odot}~{\rm kpc}^{-2}$',
        ylabel2=r'$\Sigma_{\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2}$',
        ylabel3=r'${\rm sSFR}/{\rm yr}^{-1}$',
        xmin=0, xmax=5, ymin1=ymin1, ymax1=ymax1,
        ymin2=ymin2, ymax2=ymax2, ymin3=ymin3, ymax3=ymax3,
        title='subID {}, logM={:.3f}, Re={:.3f}'.format(subIDfinal, logM, Re),
        figsizeheight=textheight/1.5, figsizewidth=textwidth,
        outfile=outfile, save=True, loc=1)
    
    return

def concatenate_all_fits(model_redshift=0.5, save=True) :
    
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
    
    # from pypdf import PdfWriter
    # merger = PdfWriter()
    # for subID, snap in zip(sample['subID'], sample['snapshot']) :
    #     outfile = 'figures/radial_profiles_GALAXEV/{}_{}_z_{:03}.pdf'.format(
    #         snap, subID, str(model_redshift).replace('.', ''))
    #     merger.append(outfile)
    # merger.write('figures/radial_profiles_GALAXEV/z_{:03}.pdf'.format(
    #     str(model_redshift).replace('.', '')))
    # merger.close()
    
    if save : # concatenate the output images by mechanism
        from pypdf import PdfWriter
        mechanisms = [1, 3]
        labels = ['inside-out', 'outside-in']
        sample = sample[np.argsort(sample['logM'])] # sort the galaxies by stellar mass
        for mechanism, label in zip(mechanisms, labels) :
            merger = PdfWriter()
            subIDs = sample['subID'][sample['mechanism'] == mechanism]
            snaps = sample['snapshot'][sample['mechanism'] == mechanism]
            for subID, snap in zip(subIDs, snaps) :
                outfile = 'figures/radial_profiles_GALAXEV/{}_{}_z_{:03}.pdf'.format(
                    snap, subID, str(model_redshift).replace('.', ''))
                merger.append(outfile)
            merger.write('figures/radial_profiles_GALAXEV/{}_z_{:03}.pdf'.format(
                label, str(model_redshift).replace('.', '')))
            merger.close()
    
    return

def get_fastpp_profiles(subID, snap, model_redshift=0.5, skiprows=18,
                        surfacedensity=True) :
    
    # load fitted data coming out of FAST++
    data = np.loadtxt('fits/fits_2April2025.fout', dtype=str, skiprows=skiprows)
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, :2].astype(int)
    use = (ids[:, 0] == snap) & (ids[:, 1] == subID)
    use[np.where(use)[0][-2:]] = False # account for 1 kpc and integrated bins
    
    # get the stellar mass and star formation rates
    mass_profile = np.power(10, data[:, 5].astype(float)[use])
    sfr_profile = np.power(10, data[:, 14].astype(float)[use]) # sfr100
    
    if surfacedensity :
        # determine the projected physical area of every circular annulus
        with fits.open('GALAXEV/{}_{}_z_{:03}_idealized_extincted.fits'.format(
            snap, subID, str(model_redshift).replace('.', ''))) as hdu :
            plate_scale = hdu[0].header['CDELT1']*u.arcsec/u.pix
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(model_redshift).to(u.kpc/u.arcsec)
        pixel_area_physical = np.square(kpc_per_arcsec)*np.square(plate_scale*u.pix)
        nPixel_profile = Table.read(
            'photometry/{}_{}_z_{:03}.fits'.format(snap, subID,
            str(model_redshift).replace('.', '')))['nPix'].value
        physical_area_profile = pixel_area_physical*nPixel_profile[:20]
        
        # convert stellar masses to surface mass densities
        fast_Sigma = mass_profile/physical_area_profile.value
        
        # convert star formation rates to surface star formation rate densities
        fast_Sigma_SFR = sfr_profile/physical_area_profile.value
    else :
        fast_Sigma = mass_profile
        fast_Sigma_SFR = sfr_profile
    
    return fast_Sigma, fast_Sigma_SFR, fast_Sigma_SFR/fast_Sigma

def get_snr_profiles(subID, snap) :
    
    # load photometry data input into FAST++
    data = np.loadtxt('photometry/photometry_2April2025.cat',
                      dtype=str)[:, :-1] #.astype(float)
    
    # define which rows to use, based on the 'id' containing the subID
    ids = data[:, 0]
    ids = np.stack(np.char.split(ids, sep='_').ravel())[:, :2].astype(int)
    use = (ids[:, 0] == snap) & (ids[:, 1] == subID)
    use[np.where(use)[0][-2:]] = False # account for 1 kpc and integrated bins
    photometry = data[:, 1:].astype(float)[use]
    
    snrs = np.array([photometry[:, 0]/photometry[:, 1],
                     photometry[:, 2]/photometry[:, 3],
                     photometry[:, 4]/photometry[:, 5],
                     photometry[:, 6]/photometry[:, 7],
                     photometry[:, 8]/photometry[:, 9],
                     photometry[:, 10]/photometry[:, 11],
                     photometry[:, 12]/photometry[:, 13],
                     photometry[:, 14]/photometry[:, 15],
                     photometry[:, 16]/photometry[:, 17]]).T
    
    castor_avg_snr = np.mean(snrs[:, :4], axis=1)
    roman_avg_snr = np.mean(snrs[:, 4:], axis=1)
    
    return castor_avg_snr, roman_avg_snr

def get_tng_profiles(subID, snap, Re=1.0, surfacedensity=True) :
    
    # get galaxy location in the massive sample
    loc = load_galaxy_attributes_massive(subID, snap, loc_only=True)
    
    # get the information about the raw radial profiles from TNG
    with h5py.File('D:/Documents/GitHub/TNG/TNG50-1/' +
        'TNG50-1_99_massive_radial_profiles(t)_2D.hdf5', 'r') as hf :
        edges = hf['edges'][:] # units of Re
        mass_profiles = hf['mass_profiles'][:] # (1666, 100, 20)
        SFR_profiles = hf['SFR_profiles'][:]   # (1666, 100, 20)
        sSFR_profiles = hf['sSFR_profiles'][:] # (1666, 100, 20)
    
    if surfacedensity :
        # determine the area for the raw TNG computed profiles
        tng_area_profile = np.full(20, np.nan)
        for i, (start, end) in enumerate(zip(edges, edges[1:])) :
            tng_area_profile[i] = np.pi*(np.square(end*Re) - np.square(start*Re))
        
        # convert stellar masses to surface mass densities
        tng_Sigma = mass_profiles[loc, snap]/tng_area_profile
        
        # convert star formation rates to surface star formation rate densities
        tng_Sigma_SFR = SFR_profiles[loc, snap]/tng_area_profile
    else :
        tng_Sigma = mass_profiles[loc, snap]
        tng_Sigma_SFR = SFR_profiles[loc, snap]
    
    return tng_Sigma, tng_Sigma_SFR, sSFR_profiles[loc, snap]
