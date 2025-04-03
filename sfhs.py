
import numpy as np

from astropy.table import Table
import h5py
from scipy.ndimage import gaussian_filter1d

from core import (convert_scalefactor_to_Gyr, get_rotation_input,
                  load_galaxy_attributes_massive)
from fastpy import dpl_from_fit, dplt_from_fit, dtt_from_fit, ftb_from_fit, get_times
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)

def check_dtt() :
    # delayed-tau + truncation (for last 100 Myr) -> 5 parameters (ltau, lage,
    # rr, metal, Av)
    
    from matplotlib import cm

    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(0.5) # Myr
    
    ys = []
    for ltau in np.arange(8.1, 9.501, 0.1) :
        for lage in np.arange(9.0, 9.901, 0.1)[-1:] :
            sfh = dtt_from_fit(ts, ltau, lage, 0.7, norm=np.power(10, 10)) # [Msol/yr]
            ys.append(sfh)
    plt.plot_simple_multi([ts/1e3]*len(ys), ys, ['']*len(ys),
        [cm.viridis(i/len(ys)) for i in range(len(ys))], ['']*len(ys),
        ['-']*len(ys), [0.6]*len(ys),
        xlabel=r'$t$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    return

def check_dpl() :
    # double power law -> 5 parameters (alpha, beta, tau, metal, Av),
    # and set lage=9.93 (fixed)
    
    from matplotlib import cm
    
    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(0.5) # Myr
    
    ys = []
    for tau in [4] :
        for alpha in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] :
            for beta in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] :
                sfh = dpl_from_fit(ts, alpha, beta, tau*1e3, norm=np.power(10, 10))
                ys.append(sfh)
    plt.plot_simple_multi([ts/1e3]*len(ys), ys, ['']*len(ys),
        [cm.viridis(i/len(ys)) for i in range(len(ys))], ['']*len(ys),
        ['-']*len(ys), [0.3]*len(ys),
        xlabel=r'$t$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    return

def check_dplt() :
    # double power law + truncation -> 6 parameters (alpha, beta, tau, rr,
    # metal, Av), and set lage=9.93 (fixed)
    
    from matplotlib import cm
    
    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(0.5) # Myr
    
    ys = []
    for tau, alpha, beta in zip([4, 5, 5, 6, 5, 2, 2, 1, 3, 3, 5, 5, 5, 3, 2, 2, 5, 3, 1, 3],
                                [7, 9, 8, 10, 9, 3, 4, 3, 6, 5, 10, 9, 8, 4, 10, 4, 10, 6, 3, 5],
                                [9, 5, 6, 4, 7, 1, 10, 2, 6, 2, 10, 10, 5, 1, 10, 1, 4, 4, 3, 1]) :
        sfh = dplt_from_fit(ts, alpha, beta, tau*1e3, 1, norm=np.power(10, 10))
        ys.append(sfh)
    plt.plot_simple_multi([ts/1e3]*len(ys), ys, ['']*len(ys),
        [cm.viridis(i/len(ys)) for i in range(len(ys))], ['']*len(ys),
        ['-']*len(ys), [0.3]*len(ys),
        xlabel=r'$t$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    return

def check_ftb() :
    # fixed time bins -> 7 parameters (r0, r1, r2, r3, r4, metal, Av),
    # and set lage=9.93 (fixed)
    
    # get the time array for the SFHs, and the observation time
    tobs, ts = get_times(0.5) # Myr
    
    from matplotlib import cm
    
    ys = []
    for r0 in [-3, -2, -1, 0, 1, 2] : # < 100 Myr
        for r1 in [-3, -2, -1, 0, 1, 2] : # 100 - 300 Myr
            for r2 in [-3, -2, -1, 0, 1, 2][2:4] : # 300 - 1000 Myr
                for r3 in [-3, -2, -1, 0, 1, 2][2:4] : # 1 - 3 Gyr
                    for r4 in [-3, -2, -1, 0, 1, 2][2:4] : # > 3 Gyr
                        sfh = ftb_from_fit(ts, r0, r1, r2, r3, r4, norm=np.power(10, 10))
                        ys.append(sfh)
    plt.plot_simple_multi([ts/1e3]*len(ys), ys, ['']*len(ys),
        [cm.viridis(i/len(ys)) for i in range(len(ys))], ['']*len(ys),
        ['-']*len(ys), [0.03]*len(ys),
        xlabel=r'$t$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)')
    
    return

# def smoothstep(xx, lo, hi) :
#     xx = clamp((xx - lo)/(hi - lo), lo, hi)
#     return xx*xx*(3 - 2*xx)

# def clamp(xx, lo, hi) :
#     if xx < lo :
#         return lo
#     if xx > hi :
#         return hi
#     return xx

def compute_sfh_per_bin(subID) :
    
    from matplotlib import cm
    
    # define the subIDs and the relevant snapshot that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, snaps = table['subID'].data, table['snapshot'].data
    
    # find the snapshot for the subID of interest
    loc = np.where(subIDs == subID)[0][0]
    
    # load relevant properties for the galaxy of interest
    tobs, mpbsubID, _, Re, center = load_galaxy_attributes_massive(subID, snaps[loc])
    
    # get all particles, noting that star ages is in units of scale factor
    _, _, _, star_ages, star_gfm, star_masses, star_coords, star_metals = get_rotation_input(
        snaps[loc], mpbsubID)
    
    # convert star ages to Myr
    ages = convert_scalefactor_to_Gyr(star_ages)*1e3
    
    # get the 2D projected distances to each stellar particle
    rs = np.sqrt(np.square(star_coords[:, 0] - center[0]) +
                 np.square(star_coords[:, 1] - center[1]))
    
    # define the edges of the radial bins
    edges = np.arange(0, 5.25, 0.25)
    
    # set the edges of the time bins for future histogram
    time_bin_edges = np.arange(0, np.ceil(tobs*1e3) + 1, 1)
    ts = time_bin_edges[1:]
    
    # get the time array for the SFHs, and the observation time
    # _, ts = get_times(0.5) # Myr
    
    tlb = np.max(ts) - ts # Myr
    # print(tlb)
    
    sigma = np.zeros_like(tlb)
    sigma[tlb <= 100] = 10
    sigma[(tlb > 100) & (tlb <= 300)] = 20
    sigma[(tlb > 300) & (tlb <= 1000)] = 70
    sigma[(tlb > 1000) & (tlb <= 3000)] = 200
    sigma[tlb > 3000] = 318
    
    # smooth the step function
    # xx = (tlb - 2682)/(3318 - 2682)
    # sm = 118*(3*np.square(xx) - 2*np.power(xx, 3)) + 200
    # sigma[(tlb >= 3000 - 318) & (tlb <= 3000 + 318)] = sm[(tlb >= 3000 - 318) & (tlb <= 3000 + 318)]
    # plt.plot_simple_dumb(tlb[(tlb >= 3000 - 318) & (tlb <= 3000 + 318)],
    #                      sm[(tlb >= 3000 - 318) & (tlb <= 3000 + 318)])
    # a = smoothstep(tlb, 3000-318, 3000+318)
    # print(a)
    # a = 3*np.square(tlb) - 2*np.power(tlb, 3)
    # a = 
    # print(a[(tlb > 3000 - 318) & (tlb < 3000 + 318)])
    # sigma[(tlb > 3000 - 318) & (tlb < 3000 + 318)] = 
    # tmp = gaussian_filter1d(sigma, 25)
    # plt.plot_simple_dumb(tlb, sigma)
    
    # a = np.arange(100)
    # b = a/20
    # b[b < 1] = 1
    # print(gaussian_filter1d(a, b))
    
    sfhs = np.full((20, len(ts)), -1.)
    # bin the stellar particles by distance from the center
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) :
        if lo == 0. :
            mask = (rs >= lo) & (rs <= hi)
        else :
            mask = (rs > lo) & (rs <= hi)
        
        # now take the histogram of those stellar ages, creating a SFH
        sfh = np.histogram(ages[mask], bins=time_bin_edges,
                           weights=star_gfm[mask])[0]
        
        # smooth the SFH for plotting, accounting for better precision at more
        # recent lookback times
        vyoung = gaussian_filter1d(sfh, 10)
        young = gaussian_filter1d(sfh, 20)
        inter = gaussian_filter1d(sfh, 70)
        old = gaussian_filter1d(sfh, 200)
        vold = gaussian_filter1d(sfh, 318)
        sm = np.full(ts.shape, 0.)
        sm[tlb <= 100] = vyoung[tlb <= 100]
        sm[(tlb > 100) & (tlb <= 300)] = young[(tlb > 100) & (tlb <= 300)]
        sm[(tlb > 300) & (tlb <= 1000)] = inter[(tlb > 300) & (tlb <= 1000)]
        sm[(tlb > 1000) & (tlb <= 3000)] = old[(tlb > 1000) & (tlb <= 3000)]
        sm[tlb > 3000] = vold[tlb > 3000]
        sfhs[i] = sm
    
    plt.plot_simple_multi([tlb/1e3]*10, sfhs[::2]/1e6,
        [('bin {}'.format(i) if i % 3 == 0 else '') for i in range(10)],
        [cm.viridis(i/9) for i in range(10)], ['']*10, ['-']*10, [0.5]*10, 
        xmin=0.001, xmax=np.max(ts)/1e3 + 1, ymin=1e-5, ymax=10, scale='log', loc=2,
        xlabel=r'$t_{\rm lookback}$ (Gyr)', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)',
        outfile='SFHs/subID_{}.pdf'.format(subID), save=False)
    
    '''
    from fastpy import calzetti, get_bc03_waves, get_lum2fl, sfh2sed_fastpp
    
    # get the library wavelengths for the constructed SED
    waves = get_bc03_waves(0.02) # use solar metallicity as a default
    
    print(waves)
    
    pivots = [0.23509513237303722, 0.3487746961275257, 0.4806630218140071,
              1.0652071101121712, 1.3005014257959315, 1.587536913485213, 1.8466132581936199]
    phot = [0.48760241600745774, 0.32302687808220926, 0.6785522069132754,
            2.8610850798561898, 3.0796223525924833, 3.1757241469855484, 3.006993287943341]
    phot_e = [0.014756999823473648, 0.010829416319564845, 0.007436682127017,
              0.08704662040732804, 0.06743443277819186, 0.053120128934523454, 0.05409862132375962]
    
    # construct the SED from the SFH for the innermost radial bin using the
    # different available metallicities
    metals = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]
    for metal in metals[-1:] :
        sed = sfh2sed_fastpp(metal, tobs*1e3, ts, sfhs[0]) # [Lsol AA^-1]
        
        # correct the SED for dust using the Calzetti dust law
        Av = 0.
        sed *= np.power(10, -0.4*Av*calzetti(waves))
        
        # convert the luminosity units to flux units
        sed *= get_lum2fl(0.5) # [10^-19 ergs s^-1 cm^-2 AA^-1]
        
        plt.plot_sed(waves*1.5, sed, pivots, [np.nan]*7, phot, phot_e,
            xmin=0.13, xmax=3, ymin=0.1, ymax=10)
    '''
    
    return

# compute_sfh_per_bin(198186)

compute_sfh_per_bin(324125)

'''
# define the subIDs and the relevant snapshot that we're interested in
table = Table.read('tools/subIDs.fits')
subIDs, snaps = table['subID'].data, table['snapshot'].data

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

# for subID in subIDs :
#     compute_sfh_per_bin(subID)

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
sort = np.argsort(logM)
subIDs = subIDs[sort]
snaps = snaps[sort]
mechs = mechs[sort]

# from pypdf import PdfWriter
# merger = PdfWriter()
# for subID in subIDs[mechs == 1] :
#     merger.append('SFHs/subID_{}.pdf'.format(subID))
# merger.write('SFHs/SFHs_per_bin_inside-out.pdf')
# merger.close()

# merger = PdfWriter()
# for subID in subIDs[mechs == 3] :
#     merger.append('SFHs/subID_{}.pdf'.format(subID))
# merger.write('SFHs/SFHs_per_bin_outside-in.pdf')
# merger.close()
'''

# print(np.log10(np.random.rand(100)))
