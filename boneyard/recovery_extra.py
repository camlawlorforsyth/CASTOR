
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
import h5py

from core import open_cutout
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def all_metrics(fast_file) :
    
    # define the subIDs and the relevant snapshot that we're interested in
    table = Table.read('tools/subIDs.fits')
    subIDs, snaps = table['subID'].data, table['snapshot'].data
    # subIDs = [13, 36, 40, 63891, 63917, 96808, 117261, 294867, 355734, 446665,
    #           515296, 576705]
    # snaps = [88, 33, 52, 57, 62, 41, 64, 55, 79, 54, 74, 89]
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
    
    # get the morphological metrics for the quenched galaxies
    # redshifts = Table.read('D:/Documents/GitHub/TNG/TNG50-1/snapshot_redshifts.fits')
    metrics_file = 'D:/Documents/GitHub/TNG/TNG50-1/morphological_metrics_-10.5_+-1_2D.fits'
    table = Table.read(metrics_file)
    tmask = (table['quenched_status'] == True) & (table['episode_progress'] >= 0.75)
    table = table[tmask]
    table.remove_columns(['quenched_status', 'sf_status', 'below_sfms_status',
                          'control_subID', 'quenched_comparison_mechanism'])
    metrics = np.full((len(subIDs), 4), -1.)
    for i, subID in enumerate(subIDs) :
        temp = table[table['quenched_subID'] == subID][0]
        metrics[i] = np.array([temp['C_SF'], np.log10(temp['R_SF']),
                               temp['Rinner'], temp['Router']])
    
    # get basic info about the quenched galaxy
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        # times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        # subIDs = hf['subIDs'][:].astype(int)
        logMfinal = hf['logM'][:, -1]
        Res = hf['Re'][:]
        # centers = hf['centers'][:]
        quenched = hf['quenched'][:]
    
    # get the quenching mechanisms
    with h5py.File('D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    
    # get the information about the raw radial profiles from TNG
    # profile_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_massive_radial_profiles(t)_3D.hdf5'
    # with h5py.File(profile_file, 'r') as hf :
        # edges = hf['edges'][:] # units of Re
        # radial_bin_centers = hf['midpoints'][:] # units of Re
        # mass_profiles = hf['mass_profiles'][:] # shape (1666, 100, 20)
        # SFR_profiles = hf['SFR_profiles'][:] # shape (1666, 100, 20)
        # sSFR_profiles = hf['sSFR_profiles'][:] # shape (1666, 100, 20)
    
    # mask to the massive galaxies for the sample information
    subIDfinals = subIDfinals[logMfinal >= 9.5]
    Res = Res[logMfinal >= 9.5]
    mechs = mechs[logMfinal >= 9.5]
    mechs = mechs[mechs > 0] # mask to the quenched sample
    mechs = mechs[mask] # mask to the SKIRT-processed sample
    mech_mask = (mechs == 1) | (mechs == 3) # mask to the IO and OI populations
    
    # load fitted data coming out of FAST++
    # fast_file = 'fits/photometry_21November2024.fout'
    # fast_file = 'fits/photometry_21November2024_allFilters.fout'
    # fast_file = 'fits/photometry_21November2024_smallPSF.fout'
    # fast_file = 'fits/photometry_21November2024_smallPSF_allFilters.fout'
    # fast_file = 'fits/photometry_21November2024_025.fout'
    # fast_file = 'fits/photometry_21November2024_025_allFilters.fout'
    # data = np.loadtxt(fast_file, dtype=str, skiprows=17) # delay-tau
    # data = np.loadtxt(fast_file, dtype=str, skiprows=19) # dpl
    data = np.loadtxt(fast_file, dtype=str, skiprows=18) # dtt with burst
    
    bin_edges = np.linspace(0, 5, 21) # units of Re
    bin_centers = np.linspace(0.125, 4.875, 20) # units of Re
    
    CSFs = np.full(np.sum(mech_mask), -1.0)
    RSFs = np.full(np.sum(mech_mask), -1.0)
    Rinners = np.full(np.sum(mech_mask), -1.0)
    Routers = np.full(np.sum(mech_mask), -1.0)
    
    idx = np.where(subIDs[mech_mask] == 282790)[0][0]
    for i, (subID, snap) in enumerate(zip(subIDs[mech_mask][idx:idx+1], snaps[mech_mask][idx:idx+1])) :
        # find the location of the subID within the entire massive sample
        loc = np.where(subIDfinals == subID)[0][0]
        
        # get the redshift and plate scale for the galaxy, in order to determine Re
        im, _, model_redshift, _, _, _, plate_scale = open_cutout(
            'cutouts/quenched/{}/castor_uv_ultradeep.fits'.format(subID))
        
        # convert Re, 1 kpc into pixels
        Re = Res[loc, snap]
        Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value # in pixels
        kpc_pix = (1*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value # in pixels
        
        # load the number of pixels per annulus
        # phot_file = 'photometry/quenched_0p5_test/subID_{}_photometry.fits'.format(subID)
        # phot = Table.read(phot_file)
        # pixel_area = phot['nPix']
        
        # get the profiles from TNG
        # tng_mass_profile = np.log10(mass_profiles[loc, snap])
        # tng_sfr_profile = np.log10(SFR_profiles[loc, snap])
        
        # define which rows to use, based on the 'id' containing the subID
        ids = data[:, 0]
        # use = np.char.find(ids, str(subID))
        # use[use < 0] = 1
        # use = np.invert(use.astype(bool))
        ids = np.stack(np.char.split(ids, sep='_').ravel())[:, 0].astype(int)
        use = (ids == subID)
        
        # delayed tau model with no uncertainties
        # lmass_loc, lsfr_loc = 6, 7 # for regular delay-tau model
        lmass_loc, lsfr_loc = 5, 6 # for delay-tau model with burst (ie. quench), or dpl
        lmass = data[:, lmass_loc].astype(float)[use]
        # lsfr = data[:, lsfr_loc].astype(float)[use] # 'lsfr'
        lsfr = data[:, 14].astype(float)[use] # 'sfr100' for delay-tau model
        
        # calculate C_SF and R_SF
        CSFs[i] = calculate_CSF(np.power(10, lsfr[:-2]), kpc_pix, Re_pix, bin_edges)
        
        # sf_mass_cumul_prof = recent_mass_cumulative_profile/recent_mass_cumulative_profile[-1]
        # Re_sf = np.interp(0.5, sf_mass_cumul_prof, bin_centers) # units of Re
        # full_mass_cumul_prof = np.nancumsum(np.power(10, lmass[:-2]))
        # full_mass_cumul_prof = full_mass_cumul_prof/full_mass_cumul_prof[-1]
        # Re_fit = np.interp(0.5, full_mass_cumul_prof, bin_centers) # units of Re
        # RSFs[i] = np.log10(Re_sf/Re_fit)
        
        RSFs[i] = calculate_RSF(subID) # to implement
        
        # get the profile for calculating Rinner and Router
        profile = lsfr[:-2] - lmass[:-2]
        
        # calculate Rinner and Router
        Rinners[i] = calculate_Rinner(bin_centers, profile)
        Routers[i] = calculate_Router(bin_centers, profile)
        
        # print(subID)
    
    # np.savez('metrics_all-from-FAST.npz', CSFs=CSFs, RSFs=RSFs, Rinners=Rinners,
    #          Routers=Routers)
    
    # plot raw quantities and corrected quantities for the metrics
    arrs = np.load('metrics_all-from-FAST.npz')
    # arrs_rev = np.load('metrics_CSFs_revised.npz')
    # arrs_rev = np.load('CSFs_with_delay-tau+burst.npz')
    # arrs_rev = np.load('CSFs_with_dpl.npz')
    
    tng_CSFs, CSFs = metrics[:, 0][mech_mask], arrs['CSFs']
    xx = np.linspace(0, 1, 1000)
    plt.plot_simple_multi([xx, tng_CSFs[mechs[mech_mask] == 1], tng_CSFs[mechs[mech_mask] == 3]],
        [xx, CSFs[mechs[mech_mask] == 1], CSFs[mechs[mech_mask] == 3]],
        ['', 'inside-out', 'outside-in'],
        ['k', 'm', 'r'],
        ['', 'o', 'o'],
        ['-', '', ''],
        [1, 1, 1],
        xlabel=r'$C_{\rm SF}$ from TNG', ylabel=r'$C_{\rm SF}$ fitted from FAST++',
        xmin=-0.02, xmax=1.02, ymin=-0.02, ymax=1.02,
        title='custom delay-tau SFH with 100 Myr burst/truncation')
    # mask = np.isfinite(CSFs)
    # tng_CSFs = tng_CSFs[mask]
    # mechs_C = mechs[mech_mask][mask]
    # CSFs = CSFs[mask]
    # popt, _ = curve_fit(linear, tng_CSFs, CSFs)
    # slope, intercept = popt[0], popt[1]
    # affine = (1 - slope)*tng_CSFs + CSFs - intercept
    # affine = (affine - np.min(affine))/(np.max(affine) - np.min(affine))
    # plt.plot_simple_multi([xx, tng_CSFs[mechs_C == 1], tng_CSFs[mechs_C == 3]],
    #     [xx, affine[mechs_C == 1], affine[mechs_C == 3]],
    #     ['', 'inside-out affine', 'outside-in affine'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     xlabel=r'$C_{\rm SF}$ from TNG', ylabel=r'$C_{\rm SF}$ fitted from FAST++',
    #     xmin=-0.02, xmax=1.02, ymin=-0.02, ymax=1.02)
    
    # tng_RSFs, RSFs = metrics[:, 1], arrs['RSFs']
    # xx = np.linspace(-1.5, 1.6, 1000)
    # plt.plot_simple_multi([xx, tng_RSFs[mechs == 1], tng_RSFs[mechs == 3]],
    #     [xx, RSFs[mechs == 1], RSFs[mechs == 3]],
    #     ['', 'inside-out', 'outside-in'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     xlabel=r'$R_{\rm SF}$ from TNG', ylabel=r'$R_{\rm SF}$ fitted from FAST++',
    #     xmin=-1.5, xmax=1.6, ymin=-1.5, ymax=1.6)
    # popt, _ = curve_fit(linear, tng_RSFs, RSFs)
    # slope, intercept = popt[0], popt[1]
    # affine = (1 - slope)*tng_RSFs + RSFs - intercept
    # plt.plot_simple_multi([xx, tng_RSFs[mechs == 1], tng_RSFs[mechs == 3]],
    #     [xx, affine[mechs == 1], affine[mechs == 3]],
    #     ['', 'inside-out affine', 'outside-in affine'],
    #     ['k', 'm', 'r'],
    #     ['', 'o', 'o'],
    #     ['-', '', ''],
    #     [1, 1, 1],
    #     xlabel=r'$R_{\rm SF}$ from TNG', ylabel=r'$R_{\rm SF}$ fitted from FAST++',
    #     xmin=-1.5, xmax=1.6, ymin=-1.5, ymax=1.6)
    
    tng_Rinners, Rinners = metrics[:, 2], arrs['Rinners']
    tng_Rinners = tng_Rinners[mech_mask]
    
    masses = logMfinal[(logMfinal >= 9.5) & quenched][mask][mech_mask]
    sort = np.argsort(masses[mechs[mech_mask] == 3])
    aa = (subIDs[mechs > 0][mech_mask][mechs[mech_mask] == 3])[sort]
    bb = (tng_Rinners[mechs[mech_mask] == 3])[sort]
    cc = (Rinners[mechs[mech_mask] == 3])[sort]
    t = Table([aa, bb, cc], names=('subID', 'TNG Rinner', 'FAST++ Rinner'))
    print()
    t.pprint(max_lines=-1)
    
    np.random.seed(0)
    tng_Rinners, Rinners = np.random.normal(tng_Rinners, 0.03), np.random.normal(Rinners, 0.03)
    
    xx = np.linspace(0, 5, 1000)
    plt.plot_simple_multi([xx, tng_Rinners[mechs[mech_mask] == 1], tng_Rinners[mechs[mech_mask] == 3]],
        [xx, Rinners[mechs[mech_mask] == 1], Rinners[mechs[mech_mask] == 3]],
        ['', 'inside-out', 'outside-in'],
        ['k', 'm', 'r'],
        ['', 'o', 'o'],
        ['-', '', ''],
        [1, 1, 1],
        xlabel=r'$R_{\rm inner}$ from TNG', ylabel=r'$R_{\rm inner}$ fitted from FAST++',
        xmin=-0.1, xmax=5.1, ymin=-0.1, ymax=5.1)
    
    tng_Routers, Routers = np.random.normal(metrics[:, 3], 0.03), np.random.normal(arrs['Routers'], 0.03)
    # tng_Routers, Routers = metrics[:, 3], arrs['Routers']
    tng_Routers = tng_Routers[mech_mask]
    plt.plot_simple_multi([xx, tng_Routers[mechs[mech_mask] == 1], tng_Routers[mechs[mech_mask] == 3]],
        [xx, Routers[mechs[mech_mask] == 1], Routers[mechs[mech_mask] == 3]],
        ['', 'inside-out', 'outside-in'],
        ['k', 'm', 'r'],
        ['', 'o', 'o'],
        ['-', '', ''],
        [1, 1, 1],
        xlabel=r'$R_{\rm outer}$ from TNG', ylabel=r'$R_{\rm outer}$ fitted from FAST++',
        xmin=-0.1, xmax=5.1, ymin=-0.1, ymax=5.1)
    
    return

'''
tng = np.array([[0.6334728598594666,      0.03692294620723261,  0.125, 5.0],
                [0.7426193356513977,     -0.31540891995090337,  0.0,   4.625],
                [0.8793418407440186,     -1.0120033429431345,   0.0,   0.375],
                [0.14572876691818237,    -0.08760659593296842,  0.0,   1.625],
                [0.43508708477020264,    -0.3790148082803399,   0.0,   0.875],
                [0.3421374559402466,     -0.3126131186937464,   0.0,   4.875],
                [0.03397173434495926,     0.756217233754058,    3.875, 0.625],
                [0.5130791068077087,     -0.025526589473554614, 0.0,   5.0],
                [0.6938819289207458,     -0.4668517233312857,   0.0,   4.375],
                [0.0017239546868950129,   1.4623318732040524,   4.125, 1.375],
                [0.00014158782141748816,  0.6115879636532506,   1.375, 5.0],
                [0.8564828038215637,     -0.11836962888068847,  0.0,   5.0]]).T

files = ['fits/photometry_21November2024.fout',
         'fits/photometry_21November2024_allFilters.fout',
         'fits/photometry_21November2024_smallPSF.fout',
         'fits/photometry_21November2024_smallPSF_allFilters.fout',
         'fits/photometry_21November2024_025.fout',
         'fits/photometry_21November2024_025_allFilters.fout']
for file in files :
    CSF, RSF = all_metrics(file)
    plt.plot_scatter(tng[0], CSF, 'k', '', 'o', xlabel='CSF TNG', ylabel='CSF fit')
    # plt.plot_scatter(tng[1], RSF, 'k', '', 'o', xlabel='RSF TNG', ylabel='RSF fit')

default = np.array([[0.45708818961487496,     0.2515768039570851,   0.0,   5.0],
                    [0.034673685045253165,    0.26726288784280067,  0.0,   3.875],
                    [0.23442288153199226,    -0.13157776349599687,  0.125, 1.125],
                    [0.0251188643150958,      0.02290932927116237,  0.375, 1.625],
                    [0.16982436524617445,    -0.23303301017279804,  3.125, 3.625],
                    [0.07943282347242814,     0.09042927410035154,  0.0,   4.875],
                    [0.10232929922807545,     0.3880592499021868,   2.125, 3.125],
                    [0.36307805477010135,     0.26704512705603517,  0.0,   5.0],
                    [0.5623413251903491,     -0.060217117461968526, 0.0,   1.625],
                    [0.11748975549395294,     0.3527976791544291,   3.625, 5.0],
                    [0.0014454397707459282,   0.3653327011629049,   1.375, 5.0],
                    [0.5623413251903491,      0.2739412607640349,   2.125, 5.0]]).T
default_allFilters = np.array([[0.4570881896148749,      0.2515768039570851,   3.875, 5.0],
                               [0.20892961308540392,     0.26726288784280067,  0.0,   5.0],
                               [1.2589254117941675,     -0.13157776349599687,  0.0,   0.375],
                               [0.02818382931264454,     0.02290932927116237,  0.125, 1.625],
                               [0.1445439770745927,     -0.23303301017279804,  0.125, 0.625],
                               [0.12589254117941673,     0.09042927410035154,  0.0,   4.375],
                               [0.10232929922807545,     0.3880592499021868,   2.375, 2.875],
                               [0.3548133892335755,      0.26704512705603517,  0.0,   5.0],
                               [0.3235936569296283,     -0.060217117461968526, 0.0,   2.875],
                               [0.07585775750291837,     0.3527976791544291,   3.875, 5.0],
                               [0.0014125375446227546,   0.3653327011629049,   1.375, 5.0],
                               [0.6918309709189364,      0.2739412607640349,   0.0,   5.0]]).T

smallPSF = np.array([[0.5248074602497725, 0.30718118870455985, 3.375, 5.0],
                     [0.04073802778041128, 0.29075478242831126, 0.0, 4.125],
                     [0.6918309709189366, -0.12324400978792743, 0.0, 1.125],
                     [0.02884031503126606, 0.013224116439307793, 0.375, 1.625],
                     [0.16982436524617442, -0.19109269115721803, 3.375, 3.875],
                     [0.08912509381337455, 0.08326546086433075, 0.0, 4.375],
                     [0.11748975549395295, 0.49225460353792394, 2.875, 2.375],
                     [0.3801893963205613, 0.3008966375910109, 0.0, 5.0],
                     [0.6165950018614822, -0.10878956569304851, 0.0, 3.125],
                     [0.14125375446227542, 0.4676584507627278, 3.875, 5.0],
                     [0.0014791083881682072, 0.3610588760764763, 1.375, 5.0],
                     [0.6309573444801932, 0.3410415937735402, 1.875, 5.0]]).T
smallPSF_allFilters = np.array([[0.5888436553555889, 0.30718118870455985, 4.125, 5.0],
                                [0.251188643150958, 0.29075478242831126, 0.0, 3.875],
                                [1.0471285480508996, -0.12324400978792743, 0.0, 1.125],
                                [0.03311311214825911, 0.013224116439307793, 0.375, 1.625],
                                [0.5623413251903491, -0.19109269115721803, 0.0, 0.625],
                                [0.14125375446227542, 0.08326546086433075, 0.0, 5.0],
                                [0.1479108388168207, 0.49225460353792394, 2.125, 2.375],
                                [0.3801893963205613, 0.3008966375910109, 0.0, 5.0],
                                [0.34673685045253166, -0.10878956569304851, 0.0, 3.125],
                                [0.08709635899560807, 0.4676584507627278, 4.125, 5.0],
                                [0.0015848931924611145, 0.3610588760764763, 1.375, 5.0],
                                [0.7413102413009176, 0.3410415937735402, 0.125, 5.0]]).T

zero25 = np.array([[0.43651583224016616, 0.35055771180474776, 3.625, 5.0],
                   [0.2137962089502232, 0.3599763159236498, 1.125, 2.125],
                   [0.45708818961487496, -0.026928776131850518, 4.375, 0.125],
                   [0.023988329190194908, 0.05916219508726297, 3.875, 1.375],
                   [0.39810717055349737, -0.10841968668741056, 4.625, 0.875],
                   [0.08511380382023764, 0.12649479874626765, 0.125, 4.875],
                   [0.07079457843841382, 0.39177588959687226, 4.125, 3.625],
                   [0.06309573444801934, 0.3411756093831031, 0.625, 5.0],
                   [0.19054607179632474, -0.021757516970177648, 2.375, 2.875],
                   [0.0831763771102671, 0.2023784385142959, 4.375, 4.625],
                   [0.005370317963702529, 0.3720612902782372, 1.375, 3.875],
                   [0.7244359600749901, 0.3503975955860739, 1.625, 5.0]]).T
zero25_allFilters = np.array([[0.06918309709189366, 0.35055771180474776, 3.875, 4.875],
                              [np.nan, 0.3599763159236498, 4.375, 2.875], # 16218.100973589291
                              [0.00380189396320561, -0.026928776131850518, 4.875, 0.875],
                              [np.nan, 0.05916219508726297, 0.125, 0.625], # 9.120108393559135e+17
                              [0.0075857757502918385, -0.10841968668741056, 3.375, 0.125],
                              [np.nan, 0.12649479874626765, 3.625, 0.375], # 43651583224016.66
                              [0.43651583224016605, 0.39177588959687226, 4.125, 3.125],
                              [0.436515832240166, 0.3411756093831031, 0.0, 3.625],
                              [0.007244359600749905, -0.021757516970177648, 0.0, 2.875],
                              [0.5495408738576245, 0.2023784385142959, 3.875, 3.375],
                              [np.nan, 0.3720612902782372, 4.875, 2.125], # 97.72372209558112
                              [np.nan, 0.3503975955860739, 0.0, 4.375]]).T # 44.66835921509634

for array in [default, default_allFilters, smallPSF, smallPSF_allFilters, zero25, zero25_allFilters] :
    plt.plot_scatter(
                     tng[0], array[0], 'k', '', 'o',
                     # np.random.normal(tng[3], 0.03), np.random.normal(array[3], 0.03), 'k', '', 'o',
                     xlabel='CSF TNG', ylabel='CSF fit'
                     # xlabel='RSF TNG', ylabel='RSF fit'
                     # xlabel='Rinner TNG', ylabel='Rinner fit'
                     # xlabel='Router TNG', ylabel='Router fit'
                     )
'''

# all_metrics('fits/photometry_23November2024_burst.fout')
# all_metrics('extra_metals_dtt_all_revisedR_withBurst/photometry_23November2024.fout')
