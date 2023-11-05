
from os import makedirs
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.optimize import curve_fit
from scipy.stats import truncnorm
import xml.etree.ElementTree as ET

from core import find_nearest
import plotting as plt
from projection import (calculate_MoI_tensor, radial_distances,
                        rotation_matrix_from_MoI_tensor)

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def check_Ngas_particles() :
    
    # open requisite information about the sample
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
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
    
    import matplotlib.pyplot as plt
    
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

def save_skirt_input(subIDfinal, snap, subID, Re, center, gas_setup='voronoi',
                     star_setup='mappings', save_gas=True, save_stars=True,
                     save_ski=True, faceon_projection=False) :
    
    infile = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    outDir = 'SKIRT/SKIRT_input_quenched/{}'.format(subIDfinal)
    outfile_gas = outDir + '/gas.txt'
    outfile_stars = outDir + '/stars.txt'
    outfile_oldstars = outDir + '/oldstars.txt'
    outfile_youngstars = outDir + '/youngstars.txt'
    outfile_ski = outDir + '/{}.ski'.format(subIDfinal)
    
    # create the output directory if it doesn't exist
    makedirs(outDir, exist_ok=True)
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        redshift = hf['redshifts'][snap]
    
    with h5py.File(infile, 'r') as hf :
        gas_coords = hf['PartType0/Coordinates'][:]
        Mgas = hf['PartType0/Masses'][:]*1e10/cosmo.h # in units of solMass
        Zgas = hf['PartType0/GFM_Metallicity'][:]
        gas_sfrs = hf['PartType0/StarFormationRate'][:]
        uu = hf['PartType0/InternalEnergy'][:]
        x_e = hf['PartType0/ElectronAbundance'][:]
        rho_gas = hf['PartType0/Density'][:]
        
        star_coords = hf['PartType4/Coordinates'][:]
        stellarHsml = hf['PartType4/StellarHsml'][:]
        Mstar = hf['PartType4/GFM_InitialMass'][:]*1e10/cosmo.h # solMass
        Zstar = hf['PartType4/GFM_Metallicity'][:]
        
        # formation times in units of scalefactor
        formation_scalefactors = hf['PartType4/GFM_StellarFormationTime'][:]
    
    # formation times in units of age of the universe (ie. cosmic time)
    formation_times = cosmo.age(1/formation_scalefactors - 1).value
    
    # calculate the rotation matrix to project the galaxy face on
    if faceon_projection :
        rot = rotation_matrix_from_MoI_tensor(calculate_MoI_tensor(
            Mgas, gas_sfrs, gas_coords, formation_times, Mstar, star_coords,
            Re, center))
        g_dx, g_dy, g_dz = np.matmul(np.asarray(rot['face-on']),
                                     (gas_coords-center).T)
        s_dx, s_dy, s_dz = np.matmul(np.asarray(rot['face-on']),
                                     (star_coords-center).T)
    else :
        # don't project the galaxy face-on
        g_dx, g_dy, g_dz = (gas_coords - center).T
        s_dx, s_dy, s_dz = (star_coords - center).T
    
    if save_gas : # save the input for the gas particles
        
        # adapted from https://www.tng-project.org/data/docs/faq/#gen6
        mu = 4/(1 + 3*0.76 + 4*0.76*x_e)*c.m_p.value # mean molecular weight
        k_B = c.k_B.to(u.erg/u.K).value # Boltzmann constant in cgs
        temp = (5/3 - 1)*uu/k_B*1e10*mu # temperature in Kelvin
        mask = (np.log10(temp) <= 6 + 0.25*np.log10(rho_gas)) # only cool gas
        
        if gas_setup == 'particle' :
            # find the distance to the 32nd other gas particle, for smoothing
            gasHsml = np.full(len(Mgas), np.nan)
            for i, coord in enumerate(gas_coords) :
                if i % 1000 == 0.0 :
                    print(i, len(gas_coords))
                gasHsml[i] = np.sort(np.sqrt(np.sum(
                    np.square(gas_coords - coord), axis=1)))[32]
            
            # https://skirt.ugent.be/skirt9/class_particle_medium.html
            g_hdr = ('subID {}\n'.format(subIDfinal) +
                     'Column 1: x-coordinate (kpc)\n' +
                     'Column 2: y-coordinate (kpc)\n' +
                     'Column 3: z-coordinate (kpc)\n' +
                     'Column 4: smoothing length (kpc)\n' +
                     'Column 5: gas mass (Msun)\n' +
                     'Column 6: metallicity (1)\n')
            gas = np.array([g_dx, g_dy, g_dz, gasHsml, Mgas, Zgas]).T
            
            # save the output to disk
            np.savetxt(outfile_gas, gas[mask], delimiter=' ', header=g_hdr)
        
        if gas_setup == 'voronoi' :
            # https://skirt.ugent.be/skirt9/class_voronoi_mesh_medium.html
            g_hdr = ('subID {}\n'.format(subIDfinal) +
                     'Column 1: x-coordinate (kpc)\n' +
                     'Column 2: y-coordinate (kpc)\n' +
                     'Column 3: z-coordinate (kpc)\n' +
                     'Column 4: gas mass (Msun)\n' +
                     'Column 5: metallicity (1)\n')
            gas = np.array([g_dx, g_dy, g_dz, Mgas, Zgas]).T
            
            # save the output to disk
            np.savetxt(outfile_gas, gas[mask], delimiter=' ', header=g_hdr)
    
    if save_stars : # save the input for the star particles
        
        # limit star particles to those that have positive formation times
        mask = (formation_scalefactors > 0)
        star_coords = star_coords[mask]
        stellarHsml = stellarHsml[mask]
        Mstar = Mstar[mask]
        Zstar = Zstar[mask]
        formation_times = formation_times[mask]
        s_dx = s_dx[mask]
        s_dy = s_dy[mask]
        s_dz = s_dz[mask]
        
        # convert the formation times to actual ages at the time of observation,
        # while also imposing a lower age limit of 1 Myr
        ages = cosmo.age(redshift).value - formation_times
        ages[ages < 0.001] = 0.001
        
        # https://skirt.ugent.be/skirt9/class_bruzual_charlot_s_e_d_family.html
        s_hdr = ('subID {}\n'.format(subIDfinal) +
                 'Column 1: x-coordinate (kpc)\n' +
                 'Column 2: y-coordinate (kpc)\n' +
                 'Column 3: z-coordinate (kpc)\n' +
                 'Column 4: smoothing length (kpc)\n' +
                 'Column 5: initial mass (Msun)\n' +
                 'Column 6: metallicity (1)\n' +
                 'Column 7: age (Gyr)\n')
        stars = np.array([s_dx, s_dy, s_dz, stellarHsml, Mstar, Zstar, ages]).T
        
        if star_setup == 'bc03' :
            # for all stellar populations, use the default Bruzual & Charlot
            # (2003) SEDs for simple stellar populations, with a Chabrier IMF,
            # and save the output to disk
            np.savetxt(outfile_stars, stars, delimiter=' ', header=s_hdr)
        
        if star_setup == 'mappings' :
            # make a mask for the young and old star particles
            oldmask = (ages > 0.01) # star particles older than 10 Myr
            youngmask = (ages <= 0.01) # star particles younger than 10 Myr
            length = np.sum(youngmask)
            
            # for old stellar populations, use the default Bruzual & Charlot
            # (2003) SEDs for simple stellar populations, with a Chabrier IMF,
            # and save the output to disk
            np.savetxt(outfile_oldstars, stars[oldmask], delimiter=' ',
                       header=s_hdr)
            
            # for young stellar populations, use the MAPPINGS-III library
            # https://skirt.ugent.be/skirt9/class_mappings_s_e_d_family.html
            ys_hdr = ('subID {}\n'.format(subIDfinal) +
                      'Column 1: x-coordinate (kpc)\n' +
                      'Column 2: y-coordinate (kpc)\n' +
                      'Column 3: z-coordinate (kpc)\n' +
                      'Column 4: smoothing length (kpc)\n' +
                      'Column 5: SFR (Msun/yr)\n' +
                      'Column 6: metallicity (1)\n' +
                      'Column 7: compactness (1)\n' +
                      'Column 8: pressure (Pa)\n' +
                      'Column 9: PDR fraction (1)\n')
            
            # define the SFR, compactness, ISM pressure, and PDR covering
            # factor, following a similar prescription as Trcka et al. (2022),
            # but use a truncated normal (ie. clipped Gaussian) for the age
            # distribution when calculating the PDR covering fraction, as we
            # want to maintain the lower age limit of 1 Myr from above, and an
            # upper limit of 10 Myr
            massrate = Mstar[youngmask]/1e7 # averaged over 10 Myr
            metallicity = Zstar[youngmask]
            compactness = np.random.normal(5*np.ones(length), 0.4)
            pressure = (1e5*c.k_B*u.K*np.power(u.cm, -3)).to(u.Pa)*np.ones(length)
            
            ages = ages[youngmask]
            aa, bb = (0.001 - ages)/0.0002, (0.01 - ages)/0.0002
            ages = truncnorm.rvs(aa, bb, loc=ages, scale=0.0002)
            fpdr = np.exp(-ages/0.003)
            if length == 1 :
                fpdr = [fpdr]
            
            youngstars = np.array([s_dx[youngmask], s_dy[youngmask],
                s_dz[youngmask], stellarHsml[youngmask], massrate,
                metallicity, compactness, pressure, fpdr]).T
            
            # save the output to disk
            np.savetxt(outfile_youngstars, youngstars, delimiter=' ',
                       header=ys_hdr)
    
    if save_ski : # save the SKIRT configuration file for processing
        
        # select the SKIRT configuration template file, based on the setup
        # for gas and star particles
        if (gas_setup == 'particle') and (star_setup == 'bc03') :
            template_file = 'SKIRT/TNG_v0.7_particle_BC03.ski'
        if (gas_setup == 'particle') and (star_setup == 'mappings') :
            template_file = 'SKIRT/TNG_v0.8_particle_MAPPINGS.ski'
        if (gas_setup == 'voronoi') and (star_setup == 'bc03') :
            template_file = 'SKIRT/TNG_v0.9_voronoi_BC03.ski'
        if (gas_setup == 'voronoi') and (star_setup == 'mappings') :
            template_file = 'SKIRT/TNG_v1.0_voronoi_MAPPINGS.ski'
        
        # define basic properties of the SKIRT run
        numPackets = int(1e8) # number of photon packets
        model_redshift = 0.5 # the redshift of the model run
        distance = 0*u.Mpc
        fdust = 0.2 # dust fraction
        minX, maxX = -10*Re.to(u.pc), 10*Re.to(u.pc) # extent of model space
        
        # define the FoV, and number of pixels for the redshift of interest
        plate_scale = 0.05*u.arcsec/u.pix
        fov = 20*Re
        nPix_raw = fov*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale
        nPix = np.ceil(nPix_raw).astype(int).value
        
        # parse the template ski file
        ski = ET.parse(template_file)
        root = ski.getroot()
        
        # access attributes of the configuration and set basic properties
        sim = root.findall('MonteCarloSimulation')[0]
        sim.set('numPackets', str(numPackets))
        sim.findall('cosmology')[0].findall('FlatUniverseCosmology')[0].set(
            'redshift', str(model_redshift))
        
        # update medium attributes
        medium = sim.findall('mediumSystem')[0].findall('MediumSystem')[0]
        if gas_setup == 'particle' :
            dust = medium.findall('media')[0].findall('ParticleMedium')[0]
        if gas_setup == 'voronoi' :
            dust = medium.findall('media')[0].findall('VoronoiMeshMedium')[0]
            dust.set('minX', str(minX))
            dust.set('maxX', str(maxX))
            dust.set('minY', str(minX))
            dust.set('maxY', str(maxX))
            dust.set('minZ', str(minX))
            dust.set('maxZ', str(maxX))
        dust.set('massFraction', str(fdust))
        grid = medium.findall('grid')[0].findall('PolicyTreeSpatialGrid')[0]
        grid.set('minX', str(minX))
        grid.set('maxX', str(maxX))
        grid.set('minY', str(minX))
        grid.set('maxY', str(maxX))
        grid.set('minZ', str(minX))
        grid.set('maxZ', str(maxX))
        
        # update instrument attributes
        instruments = sim.findall('instrumentSystem')[0].findall(
            'InstrumentSystem')[0].findall('instruments')[0]
        for instrument in instruments :
            instrument.set('fieldOfViewX', str(fov))
            instrument.set('fieldOfViewY', str(fov))
            instrument.set('distance', str(distance))
            instrument.set('numPixelsX', str(nPix))
            instrument.set('numPixelsY', str(nPix))
        
        # write the configuration file to disk
        ski.write(outfile_ski, encoding='UTF-8', xml_declaration=True)
    
    print('{} done'.format(subIDfinal))
    
    return

# from os import listdir, mkdir
# subIDs = list(np.sort(np.int_(listdir('SKIRT/SKIRT_input_quenched'))))
