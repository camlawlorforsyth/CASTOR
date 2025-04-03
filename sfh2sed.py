
from os.path import exists
import numpy as np

from astropy.table import Table

from fastpy import apply_redshift, dt_from_fit, madau, sfh2sed_fastpp
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def compare_sfh2sed() :
    
    # read the model spectra from Bruzual & Charlot (2003) for a Chabrier (2003)
    # initial mass function with solar metallicity
    table = Table.read('tools/bc03_lr_ch_z02.ised_ASCII.fits')[0]
    ages = table['AGE']/1e6     # (221), Myr
    masses = table['MASS']      # (221), Msol
    waves = table['LAMBDA']/1e4 # (1221), micron
    seds = table['SED']         # (221, 1221), 10^-19 ergs s^-1 cm^-2 AA^-1
    
    # read the output catalog after fitting all integrated photometry
    cat = np.loadtxt('fits/photometry_23November2024_integrated.fout',
                      dtype=str, skiprows=17)
    ID = cat[0, 0] # the ID of the galaxy that we're interested in
    # Av = float(cat[0, 5])
    
    # read the SFH as output by FAST++
    fast_ts, sfr_t = np.loadtxt('best_fits_and_SFHwithSFR/' +
        'photometry_23November2024_integrated_{}.sfh'.format(ID), unpack=True)
    
    # read the lookup table which includes locations and weights for
    # interpolating the model spectra at other ages, but limit to stellar
    # populations with ages less than the age of the universe at z = 0.5 (8608 Myr)
    ts, locs, weights = lookup_table_weights(ages[:176])
    
    # get the SED from the SFH by using the weights of every age at 1 Myr
    # increments, along with the model spectra included in the library
    flux = sfh_to_sed(sfr_t[fast_ts/1e6 <= ts[-1]], seds[:176], locs, weights)
    # flux *= np.power(10, -0.4*Av*calzetti(waves)) # correct for dust
    
    # get the SED from the SFH using the C++ method of FAST++
    tpl_flux = sfh2sed_fastpp(ages, masses, seds, fast_ts[-1]/1e6,
                              fast_ts/1e6, sfr_t)
    # tpl_flux *= np.power(10, -0.4*Av*calzetti(waves)) # correct for dust
    
    # read the SED as directly output by FAST++
    wl, fl = np.loadtxt('best_fits_and_SFHwithSFR/' +
        'photometry_23November2024_integrated_{}.fit'.format(ID), unpack=True)
    
    # read the SED after using the command line program fast++-sfh2sed on the output SFH
    wl_sfh2sed, fl_sfh2sed = np.loadtxt('best_fits_and_SFHwithSFR/' +
        'sfh2sed_{}.txt'.format(ID), unpack=True)
    fl_sfh2sed *= np.square(apply_redshift(0.5)) # command line version not
        # normalized?
    
    # get the SED from the SFH, constructed using the bestfit SFH parameters,
    # and correcting the SFH for total mass formed
    flux_from_bestfit = sed_from_catalog(cat, ID, ages, masses, seds, ts) # produces
        # identical output as reading the SFH as output by FAST++ (lines 31-33)
    
    # plot the results for comparison
    xs = [(wl/1e4)[fl > 0], (wl_sfh2sed/1e4)[fl_sfh2sed > 0],
          waves*(1 + 0.5), waves*(1 + 0.5), waves*(1 + 0.5)]
    ys = [fl[fl > 0], fl_sfh2sed[fl_sfh2sed > 0],
          tpl_flux, flux, flux_from_bestfit]
    labels = ['FAST++ bestfit SED', 'FAST++ sfh2sed CL',
              'FAST++ sfh2sed manual', 'CLF sfh_to_sed', 'catalog bestfit']
    colors = ['k', 'r', 'b', 'g', 'm']
    markers = ['']*5
    styles = ['-', '-', '--', '--', ':']
    alphas = [1]*5
    plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
        xlabel='wavelength (micron)',
        ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
        scale='log', xmin=0.13, xmax=6, ymin=1, ymax=3e4)
    
    return

def lookup_table_weights(ages) :
    # save lookup table for SEDs at 1 Myr increments, up to age of the
    # universe at z = 0.5 (8608 Myr)
    infile = 'tools/bc03_lookup_table_weights.npz'
    if exists(infile) :
        arr = np.load(infile)
        ts = arr['ts']
        locs = arr['locs']
        weights = arr['weights']
    else :
        ts = np.arange(1, 8608 + 1)
        weights = np.full((8608, 2), -1.) # (8609, 2)
        locs = np.full((8608, 2), -1)     # (8609, 2)
        for i, tt in enumerate(ts) :
            # check if the (1 Myr age increments) age exists in the library
            loc = np.where(ages == tt)[0]
            if len(loc) > 0 :
                locs[i] = [loc[0], -1] # store the exact locations into an array
                weights[i] = [1., -1.]
            else :
            # if the age doesn't exist in the library, use inverse distance
            # weighting to determine the weights for the closest pairs of ages
                gtr = np.where(ages > tt)[0][0] # find the first age greater than
                locs[i] = [gtr - 1, gtr]        # the desired age, and the previous
                wht = 1/np.array([tt - ages[gtr - 1], ages[gtr] - tt])
                weights[i] = wht/np.sum(wht)
        np.savez(infile, ts=ts, locs=locs, weights=weights)
    
    return ts, locs, weights

def lookup_table_totMstar_formed(ids) :
    # save a lookup table for the total stellar mass formed per galaxy, by
    # integrating the SFH of the integrated fits
    infile = 'tools/delaytau_total_mass_formed.npy'
    if exists(infile) :
        total_mass_formed = np.load(infile)
    else :
        total_mass_formed = np.zeros(ids.shape)
        for i, ID in enumerate(ids) :
            _, sfh = np.loadtxt('best_fits_and_SFHwithSFR/' +
                'photometry_23November2024_integrated_{}.sfh'.format(ID),
                unpack=True)
            total_mass_formed[i] = np.log10(np.sum(sfh*1e6))
        np.save(infile, total_mass_formed)
    
    return total_mass_formed

def sed_from_catalog(cat, ID, ages, masses, seds, ts) :
    
    # find the location for the ID of interest
    loc = np.where(cat[:, 0] == ID)[0][0]
    
    # determine the correction factor that must be applied to the SFH to
    # recover the correct SED, given that the SED is based on total formed mass,
    # and not the final total mass (which accounts for mass loss). The total
    # formed mass is found by integrating the SFH, which used a delay tau model
    totalMstarformed = np.power(10, lookup_table_totMstar_formed(cat[:, 0]))[loc]
    
    # construct the SFH from for the ID in the catalog
    # Mstar = np.power(10, float(cat[loc, 6])) # solMass
    sfh = dt_from_fit(ts, float(cat[loc, 2]), float(cat[loc, 4]), totalMstarformed)
    
    # convert the bestfit (corrected) SFH to an SED
    flux_from_bestfit = sfh2sed_fastpp(ages, masses, seds, np.max(ts),
                                       ts, sfh)
    
    return flux_from_bestfit

def sfh_to_sed(sfh, seds, locs, weights) :
    # personal implementation, using weights based on distance (in time) to
    # bounding SEDs
    
    flux = np.zeros(seds.shape[1]) 
    for loc, weight, sfr in zip(locs, weights, np.flip(sfh)) :
        if loc[1] == -1 :
            flux += sfr*seds[loc[0]]
        else :
            flux += sfr*(weight[0]*seds[loc[0]] + weight[1]*seds[loc[1]])
    
    return flux

def test_implementation() :
    
    # read the model spectra from Bruzual & Charlot (2003) for a Chabrier (2003)
    # initial mass function with solar metallicity
    table = Table.read('tools/bc03_lr_ch_z02.ised_ASCII.fits')[0]
    ages = table['AGE']/1e6     # (221), Myr
    masses = table['MASS']      # (221), Msol
    waves = table['LAMBDA']/1e4 # (1221), micron
    seds = table['SED']         # (221, 1221), 10^-19 ergs s^-1 cm^-2 AA^-1
    
    ltaus = [8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9]
    lages = [9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8]
    ys = []
    for ltau, lage in zip(ltaus, lages) :
        sed = sfh2sed_fastpp(ages, masses, seds, 8608, np.arange(1, 8609),
                dt_from_fit(ages, masses, seds, np.arange(1, 8609), ltau, lage,
                            norm=np.power(10, 10)))
        sed *= madau(0.5, waves)
        ys.append(sed)
    xs = [waves*1.5]*9
    colors = ['k', 'r', 'b', 'g', 'm', 'cyan', 'gold', 'grey', 'hotpink']
    plt.plot_simple_multi(xs, ys, ['test'] + ['']*8, colors, ['']*9, ['-']*9, [1]*9,
        xlabel='wavelength (micron)',
        ylabel=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
        scale='log', xmin=0.1368, xmax=10, ymin=0.001, ymax=5)
    
    return
