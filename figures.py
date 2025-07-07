
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from core import load_massive_galaxy_sample
from galaxev import determine_dust_radial_profile
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology
# cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

colwidth = 3.35224200913242
textwidth = 7.10000594991006
textheight = 9.095321710253218

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def save_castor_roman_throughput_plots() :
    
    castor = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g']
    # roman = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
    #          'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213',]
    roman = ['roman_f087', 'roman_f106', 'roman_f129', 'roman_f158',
             'roman_f184', 'roman_f213', 'roman_f146']
    
    # waves = []
    # trans = []
    # for filt in castor :
    #     ww, tt = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt),
    #                         unpack=True)
    #     waves.append(ww)
    #     trans.append(tt)
    # plt.plot_simple_multi(waves, trans, castor,
    #     ['hotpink', 'violet', 'darkviolet', 'dodgerblue', 'cyan'],
    #     ['']*5, ['-', '--', '-', '--', '-'], [1]*5, [], xlabel=r'Wavelength ($\mu$m)',
    #     ylabel='Transmission', xmin=0.1, xmax=0.6, ymin=0, ymax=0.7,
    #     save=False, outfile='figures/castor_filters.pdf')
    
    # for filt in roman :
    #     ww, tt = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt),
    #                         unpack=True)
    #     waves.append(ww)
    #     trans.append(tt)
    # plt.plot_simple_multi(waves[5:], trans[5:], roman,
    #     ['gold', 'darkorange', 'orangered', 'red'],
    #     ['']*4, ['--', '-', '--', '-'], [1]*4, [], xlabel=r'Wavelength ($\mu$m)',
    #     ylabel='Transmission', xmin=0.8, xmax=2.1, ymin=0, ymax=0.7,
    #     save=False, outfile='figures/roman_filters.pdf')
    
    # make a combined plot with all 9 filter curves visible
    waves = []
    trans = []
    for filt in castor+roman :
        ww, tt = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt),
                            unpack=True)
        if filt in roman :
            mask = (tt > 0)
        elif filt == 'castor_uv' :
            mask = (ww >= 0.135) & (ww <= 0.3)
        elif filt == 'castor_uvL' :
            mask = (ww >= 0.215) & (ww <= 0.3)
        elif filt == 'castor_uS' :
            mask = (ww >= 0.293) & (ww <= 0.364)
        elif filt == 'castor_u' :
            mask = (ww >= 0.293) & (ww <= 0.406)
        elif filt == 'castor_g' :
            mask = (ww >= 0.393) & (ww <= 0.557)
        waves.append(ww[mask])
        trans.append(tt[mask])
    plt.plot_simple_multi(waves, trans,
        [r'$\it{UV}$', r'$\it{UV}^{\rm L}$', r'$\it{u}^{\rm S}$', r'$\it{u}$', r'$\it{g}$',
         r'$\it{Z}$', r'$\it{Y}$', r'$\it{J}$', r'$\it{H}$', r'$\it{F}$', r'$\it{K}$', r'$\it{W}$'],
        ['darkviolet', 'violet', 'dodgerblue', 'cyan', 'limegreen',
         'yellow', 'gold', 'darkorange', 'orangered', 'red', 'darkred', 'grey'],
        ['']*12, ['-', '--', '--', '-', '-', '--', '-', '-', '-', ':', '--', ':'],
        [1, 1, 1, 1, 1, 0.8, 1, 1, 1, 0.8, 0.6, 0.8],
        [], xlabel=r'Wavelength ($\mu$m)',
        ylabel='Transmission', xmin=0.1, xmax=2.5, ymin=0, ymax=0.7,
        figsizeheight=textheight/3, figsizewidth=textwidth,
        save=True, outfile='figures/filters_new.pdf')
    
    return

def save_dust_profiles_plot() :
    
    xs = np.linspace(0, 5, 101)
    
    profiles = []
    labels = []
    for mass, offset in zip([9.5, 9.5, 9.5, 10.5, 11.5], [0, -1, -2, 0, 0]) :
        Av_r = determine_dust_radial_profile(mass, offset, xs)
        profiles.append(Av_r)
        label = (r'$\log(M_{*}/{\rm M}_{\odot}) = $' + ' {}, '.format(mass) +
                 r'$d = {}$'.format(offset))
        labels.append(label)
    
    length = len(profiles)
    colors = ['b', 'g', 'gold', 'darkorange', 'r']
    
    plt.plot_simple_multi([xs]*length, profiles, labels, colors, ['']*length,
        ['-']*length, [1]*length, [],
        xlabel='distance from center (pixel)', ylabel=r'$A(\lambda)/A_{\rm V}$',
        figsizewidth=colwidth, figsizeheight=textheight/3,
        xmin=0, xmax=5, ymin=0, ymax=2.5, save=False, outfile='dust_profiles.pdf')
    
    return



def save_expected_galaxy_counts_plot() :
    
    xx = np.linspace(9.5, 12, 1000)
    # baldry2004 = schechter_log(xx, 11.05, -0.87, 1.87e-3)
    # baldry2012 = schechter_double_log(xx, 10.72, -0.45, -1.45, 3.25e-3, 0.08e-3)
    lawlor2025 = schechter_log(xx, 11.112393152086234, -1.1038669419887588,
                               0.0005216463019846095)
    
    # adapted from https://astronomy.stackexchange.com/questions/44380
    area = (1*u.deg**2).to(u.steradian).value
    
    # zlos = np.arange(0, 1, 0.1)
    zlos = [0.0, 0.17, 0.3, 0.5]
    zhis = [0.17, 0.3, 0.5, 0.8]
    xs = []
    expected = []
    labels = []
    for zlo, zhi in zip(zlos, zhis) :
        xs.append(xx)
        
        delta_dc = cosmo.comoving_volume(zhi) - cosmo.comoving_volume(zlo)
        volume = (area/(4*np.pi)*delta_dc).value # Mpc^3
        expected.append(np.log10(lawlor2025*volume))
        
        if zlo == 0.17 :
            labels.append(r'${:.2f} < z < {:.1f}$'.format(zlo, zhi))
        elif zhi == 0.17 :
            labels.append(r'${:.1f} < z < {:.2f}$'.format(zlo, zhi))
        else :
            labels.append(r'${:.1f} < z < {:.1f}$'.format(zlo, zhi))
    expected = np.array(expected)
    
    # colors = ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lime',
              # 'limegreen', 'gold', 'orange', 'darkorange', 'red', 'darkred']
    colors = ['blue', 'limegreen', 'gold', 'red', 'k']
    length = len(colors)
    markers = ['']*length
    styles = ['-']*length
    alphas = np.ones(length)
    
    # plt.plot_simple_multi([xx, xx, xx],
    #     [np.log10(baldry2004), np.log10(baldry2012), np.log10(lawlor2025)],
    #     ['Baldry et al. (2004)', 'Baldry et al. (2012)', 'Paper I'],
    #     ['r', 'b', 'k'], ['', '', ''], [':', '--', '-'],
    #     [1, 1, 1], [], xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
    #     ylabel=r'$\log{({\rm number~density}/{\rm dex}^{-1}~{\rm Mpc}^{-3})}$',
    #     xmin=9.5, xmax=11.75, ymin=-5, ymax=-2, loc=3,
    #     figsizeheight=7, figsizewidth=7)
    
    plt.plot_simple_multi(xs, expected, labels, colors, markers, styles, alphas, [],
        xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
        ylabel=r'$\log{({\rm number}/{\rm dex}^{-1})}$',
        # ylabel=r'number density/dex$^{-1}/10^{3}$',
        xmin=9.5, xmax=11.75, ymin=0, ymax=3.5, # ymax=4000, loc=1,
        # ymin=1.343413556530888, ymax=4.343413556530888,
        figsizeheight=textheight/3, figsizewidth=colwidth,
        save=False, outfile='expected.pdf')
    
    return

def schechter_log(logM, Mstar, alpha, phi) :
    return np.log(10)*np.exp(-np.power(10, logM - Mstar))*np.power(
        10, (logM - Mstar)*(alpha + 1))*phi

def schechter_double_log(logM, Mstar, alpha1, alpha2, phi1, phi2) :
    return np.log(10)*np.exp(-np.power(10, logM - Mstar))*(
        phi1*np.power(10, (logM - Mstar)*(alpha1 + 1)) +
        phi2*np.power(10, (logM - Mstar)*(alpha2 + 1)))

def save_example_sfhs_and_seds() :
    
    # load fitted data coming out of FAST++, and look for peaks in the distributions
    # data = np.loadtxt('fits/fits_10June2025_050_revised.fout',
    #     dtype=str, skiprows=18)[:, 2:8].astype(float)
    # data_mask = np.full(data.shape[0], True)
    # data_mask[20::22] = False
    # data_mask[21::22] = False
    # data = data[data_mask]
    # for i, lab in zip(range(6), ['Z', 'lage', 'ltau', 'R', 'Av', 'lmform']) :
    #     plt.histogram(data[:, i], lab, bins=30)
    
    from fastpy import dtt_from_fit, get_bc03_waves, get_lum2fl, get_times, sfh2sed_fastpp
    
    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(0.5) # Myr
    
    xs1 = []
    ys1 = []
    labels = [r'$\log{(\tau)} = 9$, $R = 1$',
              r'$\log{(\tau)} = 9.3$, $R = 0.001$',
              r'$\log{(\tau)} = 9.3$, $R = 1$']
    # colors = ['k', 'b', 'k']
    # markers = ['', '', '']
    # styles = ['--', '-', ':']
    # alphas = [1, 0.7, 1]
    for ltau, RR in zip([9, 9.3, 9.3], [0, -3, 0]) :
        sfh = dtt_from_fit(ts, ltau, 9.9, RR, norm=np.power(10, 9)) # [Msol/yr]
        xs1.append(ts/1e3)
        ys1.append(sfh)
        labels.append('')
    # plt.plot_simple_multi(xs1, ys1, labels, colors, markers,
    #     styles, alphas, xmin=0, xmax=tobs/1e3,
    #     xlabel=r'$t$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    ww = get_bc03_waves(0.02)*u.um # nu = ww.to(u.Hz, equivalencies=u.spectral())
    
    xs2 = []
    ys2 = []
    for sfh in ys1 :
        sed = sfh2sed_fastpp(0.02, tobs, ts, 1e6*sfh) # [Lsol AA^-1]
        sed *= get_lum2fl(0.5) # convert luminosity to flux -> [10^-19 ergs s^-1 cm^-2 AA^-1]
        xs2.append(ww.value*1.5)
        ys2.append(sed)
    # plt.plot_simple_multi(xs2, ys2, labels, colors, markers, styles, alphas,
    #     xmin=0.1365, xmax=2.5, ymin=1e-4, ymax=1, scale='log',
    #     xlabel='wavelength', ylabel='flux')
    
    # for metallicity in [0.008, 0.02, 0.05]
    # for Av in [0, 0.5, 1] :
    
    castor = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g']
    roman = ['roman_f087', 'roman_f106', 'roman_f129', 'roman_f158',
             'roman_f184', 'roman_f213', 'roman_f146']
    waves = []
    trans = []
    for filt in castor+roman :
        ww, tt = np.loadtxt('passbands/passbands_micron/{}.txt'.format(filt),
                            unpack=True)
        if filt in roman :
            mask = (tt > 0)
        elif filt == 'castor_uv' :
            mask = (ww >= 0.135) & (ww <= 0.3)
        elif filt == 'castor_uvL' :
            mask = (ww >= 0.215) & (ww <= 0.3)
        elif filt == 'castor_uS' :
            mask = (ww >= 0.293) & (ww <= 0.364)
        elif filt == 'castor_u' :
            mask = (ww >= 0.293) & (ww <= 0.406)
        elif filt == 'castor_g' :
            mask = (ww >= 0.393) & (ww <= 0.557)
        waves.append(ww[mask])
        trans.append(tt[mask])
    
    plt.side_by_side(xs1, ys1, xs2, ys2, waves, trans,
        xlabel1=r'$t$ (Gyr)', ylabel1=r'SFR (M$_{\odot}$ yr$^{-1}$)',
        xlabel2=r'$\lambda$ ($\mu$m)',
        ylabel2=r'flux (10$^{-19}$ ergs s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',
        xmin1=0, xmax1=tobs/1e3, ymin1=0, ymax1=0.4, # xmax1=tobs/1e3
        xmin2=0.134, xmax2=2.5, ymin2=4e-3, ymax2=15,
        figsizeheight=textheight/3, figsizewidth=textwidth,
        save=False, outfile='sfh_and_seds.pdf')
    
    return

'''
# determine potential SF galaxies to include in a the mock observation mosaic
sample = load_massive_galaxy_sample()
io_possible = sample[(sample['snapshot'] == 85) & (sample['subID'] == 575984) # subIDfinal 625281
                     & sample['mechanism'] == 1]
oi_possible = sample[(sample['snapshot'] == 45) & (sample['subID'] == 6)] # subIDfinal 47

snap_lo = min(io_possible['snapshot'], oi_possible['snapshot']).value[0]
snap_hi = max(io_possible['snapshot'], oi_possible['snapshot']).value[0]
logM_lo = min(io_possible['logM'], oi_possible['logM']).value[0]
logM_hi = max(io_possible['logM'], oi_possible['logM']).value[0]
sfr_lo = min(io_possible['SFR'], oi_possible['SFR']).value[0]
sfr_hi = max(io_possible['SFR'], oi_possible['SFR']).value[0]
Re_lo = min(io_possible['Re'], oi_possible['Re']).value[0]
Re_hi = max(io_possible['Re'], oi_possible['Re']).value[0]
ep_lo = min(io_possible['episode_progress'], oi_possible['episode_progress']).value[0]
ep_hi = max(io_possible['episode_progress'], oi_possible['episode_progress']).value[0]

sf_mask = ((sample['snapshot'] >= snap_lo) & (sample['snapshot'] <= snap_hi) &
           (sample['logM'] >= logM_lo) & (sample['logM'] <= logM_hi) &
           (sample['SFR'] >= sfr_lo) & (sample['SFR'] <= sfr_hi) &
           (sample['Re'] >= Re_lo) & (sample['Re'] <= Re_hi) &
           (sample['episode_progress'] >= ep_lo) & (sample['episode_progress'] <= ep_hi))

quenched_subIDfinals = np.unique(sample['subIDfinal'][sample['mechanism'] > 0].value)

# select only control star forming galaxies which will never be quenched galaxies
potential_sf = sample[sf_mask]
mask = np.full(len(potential_sf), True)
for i, ID in enumerate(potential_sf['subIDfinal']) :
    if ID in quenched_subIDfinals :
        mask[i] = False
potential_sf = potential_sf[mask]
potential_sf.write('potential_SF_mosaic_galaxies.fits')
'''

# sample = load_massive_galaxy_sample()
# s = sample[(sample['subIDfinal'] == 198186) & (sample['mechanism'] == 3)]
# print(s)

