
from os import makedirs
from os.path import exists
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.stats import truncnorm
import xml.etree.ElementTree as ET

from projection import calculate_MoI_tensor, rotation_matrix_from_MoI_tensor

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def save_all_skirt_input(print_input=False) :
    
    # open requisite information about the sample
    file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(file, 'r') as hf :
        # redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        quenched = hf['quenched'][:]
        # ionsets = hf['onset_indices'][:]
        tonsets = hf['onset_times'][:]
        # iterms = hf['termination_indices'][:]
        tterms = hf['termination_times'][:]
    
    # get the quenching mechanisms
    mech_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_mechanism.hdf5'
    with h5py.File(mech_file, 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    
    # define a mask to select the quenched galaxies with sufficient solar mass
    mask = quenched & (logM[:, -1] >= 9.5)
    
    # mask relevant properties
    subIDfinals = subIDfinals[mask]
    subIDs = subIDs[mask]
    logM = logM[mask]
    Res = Res[mask]
    centers = centers[mask]
    # ionsets = ionsets[mask]
    tonsets = tonsets[mask]
    # iterms = iterms[mask]
    tterms = tterms[mask]
    mechs = mechs[mask]
    
    # find the first snapshot >=75% of the way through the quenching episode
    t75s = tonsets + 0.75*(tterms - tonsets)
    i75s = np.full(278, -1)
    for i, t75 in enumerate(t75s) :
        i75s[i] = np.where(times >= t75)[0][0]
    
    firstDim = np.arange(278)
    subIDs = subIDs[firstDim, i75s]
    Res = Res[firstDim, i75s]
    centers = centers[firstDim, i75s]
    
    # save a table for future use with core information for saving, adding noise,
    # creating photometry tables, and fitting
    tools_file = 'tools/subIDs.fits'
    if not exists(tools_file) :
        table = Table([subIDfinals, i75s, Res*u.kpc, mechs],
                      names=('subID', 'snapshot', 'Re', 'mech'))
        table.write(tools_file)
    
    # print the input calls
    if print_input :
        for subIDfinal, i75, subID, Re, center in zip(subIDfinals, i75s,
            subIDs, Res, centers) :
            params = [int(subIDfinal), int(i75), int(subID), float(Re), center.tolist()]
            # if not (save_gas or save_stars) :
            #     params += ['save_gas=False, save_stars=False']
            params = str(tuple(params)).replace("'", "")
            print('save_skirt_input' + params)
    
    # mask the sample to only the inside-out and outside-in population
    mech_mask = (mechs == 1) | (mechs == 3)
    
    # save the input
    for subIDfinal, i75, subID, Re, center in zip(subIDfinals[mech_mask],
        i75s[mech_mask], subIDs[mech_mask], Res[mech_mask], centers[mech_mask]) :
        # subIDfinal 43 at snap 94 (mpbsubID 53) does not have any gas cells
        if subIDfinal != 43 :
            save_skirt_input(subIDfinal, i75, subID, Re, center) # default
            # save_skirt_input(subIDfinal, i75, subID, Re, center, # z = 0.25
            #     save_gas=False, save_stars=False, model_redshift=0.25)
            # save_skirt_input(subIDfinal, i75, subID, Re, center,
            #     save_gas=False, save_stars=False) # rerun to save image for Av
            # save_skirt_input(subIDfinal, i75, subID, Re, center, # z = 0.25 rerun
            #     save_gas=False, save_stars=False, model_redshift=0.25)
    
    # for subIDfinal, i75, subID, Re, center in zip(subIDfinals, i75s,
    #     subIDs, Res, centers) :
    #     inDir = 'F:/TNG50-1/mpb_cutouts_099/'
    #     infile = inDir + 'cutout_{}_{}.hdf5'.format(i75, subID)
    #     if (i75 == 94) and (subID == 53) :
    #         print(subIDfinal)
    #         with h5py.File(infile, 'r') as hf :
    #             print(hf['PartType4'].keys())
    #     with h5py.File(infile, 'r') as hf :
    #         if 'PartType0' not in hf.keys() :
    #             print('snap = {}'.format(i75))
    #             print('mpbsubID = {}'.format(subID))
    #         if 'GFM_Metallicity' not in hf['PartType4'].keys() :
    #             print('snap = {}'.format(i75))
    #             print('mpbsubID = {}'.format(subID))
    
    return

def save_skirt_input(subIDfinal, snap, subID, Re, center, gas_setup='voronoi',
                     star_setup='mappings', save_gas=True, save_stars=True,
                     save_ski=True, model_redshift=0.5, fov=10,
                     faceon_projection=False, no_medium=True) :
    
    file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    infile = 'S:/Cam/University/GitHub/TNG/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    outDir = 'SKIRT/SKIRT_input_quenched/{}'.format(subIDfinal)
    outfile_gas = outDir + '/gas.txt'
    outfile_stars = outDir + '/stars.txt'
    outfile_oldstars = outDir + '/oldstars.txt'
    outfile_youngstars = outDir + '/youngstars.txt'
    outfile_ski = outDir + '/{}_z_{:03}.ski'.format(
        subIDfinal, str(model_redshift).replace('.', ''))
    outfile_NoMedium = outDir + '/{}_z_{:03}_NoMedium.ski'.format(subIDfinal,
        str(model_redshift).replace('.', ''))
    
    # create the output directory if it doesn't exist
    makedirs(outDir, exist_ok=True)
    
    # attach units to the effective radius
    Re = Re*u.kpc
    
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
        
        # define basic properties of the SKIRT run, using the model_redshift
        # above as the redshift of the model run
        numPackets = int(1e8) # number of photon packets
        distance = 0*u.Mpc
        fdust = 0.2 # dust fraction
        minX, maxX = -(fov/2*Re).to(u.pc), (fov/2*Re).to(u.pc) # extent of model space
        
        # define the FoV, and number of pixels for the redshift of interest
        plate_scale = 0.05*u.arcsec/u.pix
        nPix_raw = fov*Re*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale
        nPix = np.ceil(nPix_raw).astype(int).value
        if nPix % 2 == 0 : # ensure all images have an odd number of pixels,
            nPix += 1      # so that a central pixel exists
        
        # for lower resolution instruments (GALEX, Euclid-NISP, WISE, Spitzer,
        # Herschel), update the number of pixels as necessary (assuming a second
        # pointing where dithering is completed, improving the native pixel
        # scale by a factor of two)
        plate_scales_lr = {'galex':0.75, 'euclid_nisp':0.15, 'wise':0.65,
            'spitzer_irac':0.6, 'spitzer_24':1.25,  'spitzer_70':5,
            'spitzer_160':8.5, 'herschel_pacs':1.6, 'herschel_160':3.2,
            'herschel_250':3, 'herschel_350':5, 'herschel_500':7}
        nPix_lrs = np.full(12, -1)
        for i, plate_scale_lr in enumerate(plate_scales_lr.values()) :
            plate_scale_lr *= u.arcsec/u.pix
            nPix_lr_raw = fov*Re*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale_lr
            nPix_lr = np.ceil(nPix_lr_raw).astype(int).value
            if nPix_lr % 2 == 0 : # ensure images have an odd number of pixels,
                nPix_lr += 1      # so that a central pixel exists
            nPix_lrs[i] = nPix_lr
        
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
        for instrument in instruments[:10] :
            instrument.set('fieldOfViewX', str(fov*Re))
            instrument.set('fieldOfViewY', str(fov*Re))
            instrument.set('distance', str(distance))
            instrument.set('numPixelsX', str(nPix))
            instrument.set('numPixelsY', str(nPix))
        # update low-resolution instrument attributes
        for instrument, nPix_lr in zip(instruments[10:], nPix_lrs) :
            instrument.set('fieldOfViewX', str(fov*Re))
            instrument.set('fieldOfViewY', str(fov*Re))
            instrument.set('distance', str(distance))
            instrument.set('numPixelsX', str(nPix_lr))
            instrument.set('numPixelsY', str(nPix_lr))
        
        # write the configuration file to disk
        ski.write(outfile_ski, encoding='UTF-8', xml_declaration=True)
        
        if no_medium : # save a modified SKIRT configuration file, without dust
                       # for checking Av input values for FAST++ fitting
            nodust = ski # copy the ski parameter file from above
            root = nodust.getroot()
            
            # set the simulation mode
            sim = root.findall('MonteCarloSimulation')[0]
            sim.set('simulationMode', 'NoMedium')
            
            # remove the medium system as there is no medium
            sim.remove(sim.findall('mediumSystem')[0])
            
            # get all the instruments
            system = sim.findall('instrumentSystem')[0].findall(
                'InstrumentSystem')[0]
            instruments = system.findall('instruments')[0]
            
            # remove the unnecessary instruments
            for instrument in instruments[:4]+instruments[5:]:
                instruments.remove(instrument)
            
            # rename the remaining instrument
            instruments[0].set('instrumentName', 'Av_raw')
            
            # write the configuration file to disk
            nodust.write(outfile_NoMedium, encoding='UTF-8', xml_declaration=True)
    
    if save_gas or save_stars :
        with h5py.File(file, 'r') as hf :
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
    
    print('{} done'.format(subIDfinal))
    
    return

'''
table = Table.read('tools/subIDs.fits')
mask = (table['mech'] == 1) | (table['mech'] == 3)

for subID in table[mask]['subID'].data[-1:] :
    infile = 'skirt/SKIRT_input_quenched/{}/gas.txt'.format(subID)
    _, _, _, Mgas, Zgas = np.loadtxt(infile, unpack=True)
    
    twelvePluslogOH = np.log10(Zgas/0.0127) + 8.69
    
    logDTG = np.where(twelvePluslogOH <= 8.1, 3.1*np.log10(Zgas/0.0127) - 0.96,
                      np.log10(Zgas/0.0127) - 2.21)
    
    Mdust_frac = 0.2*Zgas # from Trcka+2022 and others, usual perscription
    Mdust_frac_alt = np.power(10, logDTG) # from Remy-Ruyer+2014, also in Bottrell+2024
    
    print(np.sort(Mdust_frac_alt/Mdust_frac)[:1000])
'''

# from GALEXEV_pipeline
# https://github.com/vrodgom/galaxev_pipeline/blob/master/galaxev_pipeline/create_images.py#L135-L156
# from scipy.spatial import KDTree
# pos = np.loadtxt('SKIRT/SKIRT_input_quenched/96795/oldstars.txt')[:, :3]
# posyoung = np.loadtxt('SKIRT/SKIRT_input_quenched/96795/youngstars.txt')[:, :3]
# pos = np.vstack((pos, posyoung))
# tree = KDTree(pos)
# res = tree.query(pos, k=32 + 1)
# hsml = res[0][:, -1]
