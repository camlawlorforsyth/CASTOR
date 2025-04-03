
import numpy as np

import astropy.units as u
from scipy.optimize import curve_fit

from core import find_nearest
import plotting as plt
from projection import radial_distances

def check_Ngas_particles() :
    
    # open requisite information about the sample
    file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(file, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:]
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:]
        tterms = hf['termination_times'][:]
    
    # define a mask to select the quenched galaxies with sufficient solar mass
    mask = quenched & (logM[:, -1] >= 9.5)
    
    # mask relevant properties
    subIDfinals = subIDfinals[mask]
    subIDs = subIDs[mask]
    logM = logM[mask]
    Res = Res[mask]
    centers = centers[mask]
    ionsets = ionsets[mask]
    tonsets = tonsets[mask]
    iterms = iterms[mask]
    tterms = tterms[mask]
    
    # find the snapshot corresponding to roughly 75% of the way through the
    # quenching episode, and the redshift at that snapshot
    iseventys = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    z_75 = redshifts[iseventys]
    
    # get stellar masses and sizes at those snapshots
    firstDim = np.arange(278)
    subs = subIDs[firstDim, iseventys]
    masses = logM[firstDim, iseventys]
    rads = Res[firstDim, iseventys]*u.kpc # ckpc/h
    cents = centers[firstDim, iseventys]
    
    dim = (10*rads)*cosmo.arcsec_per_kpc_comoving(z_75)/(0.05*u.arcsec/u.pix)
    dim = np.ceil(dim).astype(int)
    
    '''
    for subIDfinal, iseventy, subID, Re, center in zip(subIDfinals,
        iseventys, subs, rads, cents) :
        params = [subIDfinal, iseventy, subID, Re.value, center.tolist()]
        print('save_skirt_input' + str(tuple(params)))
    '''
    
    Nstars, Ngass = [], []
    Nstar_maskeds, Ngas_maskeds = [], []
    for snap, subID, Re, center in zip(iseventys, subs, rads.value, cents) :
        inDir = 'F:/TNG50-1/mpb_cutouts_099/'
        cutout_file = inDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
        
        with h5py.File(cutout_file, 'r') as hf :
            star_coords = hf['PartType4/Coordinates'][:]
            star_rs = radial_distances(center, star_coords)
            
            # only one galaxy have no gas particles at all
            if 'PartType0' not in hf.keys() :
                gas_rs = []
            else :
                gas_coords = hf['PartType0/Coordinates'][:]
                gas_rs = radial_distances(center, gas_coords)
        
        Nstar = len(star_rs)
        Ngas = len(gas_rs)
        
        Nstar_masked = len(star_rs[star_rs <= 5*Re])
        
        if Ngas == 0 :
            Ngas_masked = 0
        else :
            Ngas_masked = len(gas_rs[gas_rs <= 5*Re])
        
        Nstars.append(Nstar)
        Ngass.append(Ngas)
        Nstar_maskeds.append(Nstar_masked)
        Ngas_maskeds.append(Ngas_masked) # an additional galaxy has no gas <= 5Re

    tt = Table([subIDfinals, logM[:, -1],
                subs, masses, iseventys, z_75, rads,
                Nstars, Nstar_maskeds, Ngass, Ngas_maskeds, dim],
               names=('subID', 'logM',
                      'subID_75', 'logM_75', 'snap_75', 'z_75', 'Re_75',
                      'Nstar', 'Nstar_5Re', 'Ngas', 'Ngas_5Re', 'dim'))
    tt.pprint(max_lines=-1, max_width=-1)
    # tt.write('SKIRT/Ngas_particles_with_dim.fits')
    
    # from pypdf import PdfWriter
    # merger = PdfWriter()
    # inDir = 'TNG50-1/figures/comprehensive_plots/'
    # for subID in subIDfinals[np.argsort(z_75)] :
    #     merger.append(inDir + 'subID_{}.pdf'.format(subID))
    # outfile = 'SKIRT/comprehensive_plots_by_z75.pdf'
    # merger.write(outfile)
    # merger.close()
    
    return

def determine_runtime_with_photons() :
    
    xs = np.log10([1e6, 3162278, 1e7, 31622777, 1e8, 316227767, 1e9, 1e10])
    ys = np.log10([62, 70, 78, 124, 255, 696, 2177, 19333])
    
    popt_quad, _ = curve_fit(parabola, xs, ys, p0=[0.15, -1.8, 7.1])
    popt_exp, _ = curve_fit(exponential, xs, ys, p0=[0.040, 0.44, 1.1])
    
    xlin = np.linspace(6, 10, 100)
    ylin_para = parabola(xlin, *popt_quad)
    ylin_exp = exponential(xlin, *popt_exp)
    
    plt.plot_simple_multi([xlin, xlin, xs], [ylin_para, ylin_exp, ys],
        [r'$f(x) = ax^2 + bx + c$', r'$f(x) = Ae^{Bx} + C$', 'data'],
        ['r', 'b', 'k'], ['', '', 'o'], ['-', '--', ''], [1, 0.3, 1],
        xlabel=r'$\log(N_{\rm photons})$',
        ylabel=r'$\log({\rm runtime}/{\rm s})$', scale='linear')
    
    return

def exponential(xx, AA, BB, CC) :
    return AA*np.exp(BB*xx) + CC

def make_rgb() :
    
    from astropy.io import fits
    from astropy.visualization import make_lupton_rgb
    
    inDir = 'SKIRT/subID_513105_1e10/'
    infile = 'TNG_v0.5_fastTest_subID_513105_sed_cube_obs_total.fits'
    
    filters = ['HST_F218W',   'CASTOR_UV',   'HST_F225W',   'HST_F275W',
               'HST_F336W',   'CASTOR_U',    'HST_F390W',   'HST_F435W',
               'CASTOR_G',    'HST_F475W',   'HST_F555W',   'HST_F606W',
               'ROMAN_F062',  'HST_F625W',   'JWST_F070W',  'HST_F775W',
               'HST_F814W',   'ROMAN_F087',  'JWST_F090W',  'HST_F850LP',
               'HST_F105W',   'ROMAN_F106',  'HST_F110W',   'JWST_F115W',
               'HST_F125W',   'ROMAN_F129',  'HST_F140W',   'ROMAN_F146',
               'JWST_F150W',  'HST_F160W',   'ROMAN_F158',  'ROMAN_F184',
               'JWST_F200W',  'ROMAN_F213',  'JWST_F277W',  'JWST_F356W',
               'JWST_F410M',  'JWST_F444W'] # 'JWST_F560W',  'JWST_F770W',
               # 'JWST_F1000W', 'JWST_F1130W', 'JWST_F1280W', 'JWST_F1500W',
               # 'JWST_F1800W', 'JWST_F2100W', 'JWST_F2550W']
    
    # with h5py.File('SKIRT/SKIRT_cube_filters_and_waves_new.hdf5', 'w') as hf :
    #     add_dataset(hf, np.arange(47), 'index')
    #     add_dataset(hf, filters, 'filters', dtype=str)
    #     add_dataset(hf, waves, 'waves')
    
    with fits.open(inDir + infile) as hdu :
        # hdr = hdu[0].header
        data = hdu[0].data*u.MJy/u.sr # 38, 491, 491
        dim = data.shape
        # waves_hdr = hdu[1].header
        # waves = np.array(hdu[1].data.astype(float)) # effective wavelengths, in um
    
    # Re = 2.525036573410034*u.kpc
    # size = 10*Re*cosmo.arcsec_per_kpc_comoving(0.0485236299818059)/pixel_scale
    
    # define the area of a pixel, based on CASTOR resolution after dithering
    # pixel_scale = 0.05*u.arcsec/u.pix
    area = 0.0025*u.arcsec**2
    
    image_r = np.full(dim[1:], 0.0)
    for frame in data[20:] :
        image_r += (frame*area).to(u.Jy).value
    image_r = image_r/18
    
    image_g = np.full(dim[1:], 0.0)
    for frame in data[7:20] :
        image_g += (frame*area).to(u.Jy).value
    image_g = image_g/13
    
    image_b = np.full(dim[1:], 0.0)
    for frame in data[:7] :
        image_b += (frame*area).to(u.Jy).value
    image_b = image_b/7
    
    # image = make_lupton_rgb(image_r, image_g, image_b, Q=10, stretch=0.5)
    # plt.imshow(image)
    
    # for frame in data :
    #     m_AB = -2.5*np.log10(np.sum(frame*area).to(u.Jy)/(3631*u.Jy))
    
    return

def parabola(xx, aa, bb, cc) :
    return aa*np.square(xx) + bb*xx + cc

def prepare_for_saving_skirt_input(print_strings=False) :
    
    from scipy.ndimage import gaussian_filter1d
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    with h5py.File('../TNG/TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:, -1]
        SMHs = hf['logM'][:]
        SFHs = hf['SFH'][:]
        lo_SFHs = hf['lo_SFH'][:]
        hi_SFHs = hf['hi_SFH'][:]
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        # comparison = hf['comparison'][:]
        
        Res = hf['Re'][:]
        centers = hf['centers'][:]
    
    quenched = (quenched & (logM >= 9.5))
    i75s = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    np.random.seed(0)
    
    plot = False
    outDir = '../TNG/TNG50-1/figures/quenched_SFHs(t)_logM-gtr-9.5/'
    
    seventyfives = []
    for i, (subIDfinal, subIDhist, ReHist, centerHist, SMH, SFH, lo, hi, ionset, tonset,
        iterm, tterm, i75, val) in enumerate(zip(subIDfinals[quenched],
        subIDs[quenched], Res[quenched], centers[quenched], SMHs[quenched],
        SFHs[quenched], lo_SFHs[quenched], hi_SFHs[quenched], ionsets[quenched],
        tonsets[quenched], iterms[quenched], tterms[quenched], i75s[quenched],
        np.random.rand(278))) :
        
        SFH = gaussian_filter1d(SFH, 2)
        lo = gaussian_filter1d(lo, 2)
        hi = gaussian_filter1d(hi, 2)
        
        # defined relative to only the onset of quenching
        onset_SFR = SFH[ionset]
        SFH_after_peak = SFH[ionset:]
        drop75_list = np.where(SFH_after_peak - 0.25*onset_SFR <= 0)[0]
        
        # the last 4 subIDs don't have a drop of 75% of the SFR within the
        # quenching episode, and subID 43 has some issue, but can't remember
        if subIDfinal not in [43, 514274, 656524, 657979, 680429] : # len(drop75_list) > 0
            
            # the first snapshot where the SFR is 75% below the SFR at quenching onset
            drop75 = ionset + drop75_list[0]
            # drop75 = ionset + np.argmin(np.abs(SFH_after_peak - 0.25*onset_SFR))
            
            drop_times = [times[i75], times[drop75]]
            drop_labels = [r'$t_{\rm 0.75~through~episode}$',
                            r'$t_{\rm 0.25~SFR_{\rm onset}}$']
            
            Re = ReHist[drop75]
            center = centerHist[drop75]
            
            string = 'save_skirt_input({}, {}, {}, {}*u.kpc, {})'.format(
                subIDfinal, drop75, subIDhist[drop75], Re, list(center))
            if print_strings :
                print(string)
            
            seventyfives.append(times[drop75])
        else :
            seventyfives.append(np.nan)
            
            if (val < 0.025) and plot :
                plt.plot_simple_multi_with_times([times, times, times], [SFH, lo, hi],
                ['SFH', r'$\pm 2 \sigma$', ''], ['k', 'grey', 'grey'], ['', '', ''],
                ['-', '-.', '-.'], [1, 1, 1], np.nan, tonset, tterm, drop_times, drop_labels,
                scale='linear', xmin=-0.1, xmax=13.8,
                xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
                outfile=outDir + 'SFH_subID_{}.png'.format(subIDfinal), save=False)
        
            plt.plot_simple_multi_with_times([times, times, times], [SFH, lo, hi],
            ['SFH', r'$\pm 2 \sigma$', ''], ['k', 'grey', 'grey'], ['', '', ''],
            ['-', '-.', '-.'], [1, 1, 1], np.nan, tonset, tterm, [times[i75], np.nan], drop_labels,
            scale='linear', xmin=-0.1, xmax=13.8,
            xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
            outfile='D:/Desktop/SFH_q_subID_{}.png'.format(subIDfinal), save=False)
    
    plt.plot_scatter_dumb(times[i75s[quenched]], seventyfives, 'k', '', 'o',
        xlabel=r'$t_{\rm 75}$ (Gyr)', ylabel=r'$t_{\rm 0.3~max}$',
        xmin=0, xmax=14, ymin=0, ymax=14)
    
    return
