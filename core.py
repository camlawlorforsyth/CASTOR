
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import h5py

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def calculate_distance_to_center(shape) :
    
    # get the shape of the image and determine the center
    dim = shape[0]
    center = (dim - 1)/2
    
    # determine the distance to every pixel from the center of the image
    YY, XX = np.ogrid[:dim, :dim]
    dist_from_center = np.sqrt(np.square(XX - center) + np.square(YY - center))
    
    return dist_from_center

def convert_scalefactor_to_Gyr(scalefactors) :
    
    # look-up table for converting scalefactors to cosmological ages
    infile = 'D:/Documents/GitHub/TNG/TNG50-1/scalefactor_to_Gyr.hdf5'
    with h5py.File(infile, 'r') as hf :
        sf, age = hf['scalefactor'][:], hf['age'][:]
    
    return np.interp(scalefactors, sf, age)

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def get_particles(snap, subID, center) :
    
    cutout_file = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # convert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # calculate the 3D distances from the galaxy center
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        # limit particles to those that have positive formation times
        mask = (formation_ages > 0)
        
        return formation_ages[mask], masses[mask], rs[mask]
    
    except (KeyError, OSError) :
        return None, None, None

def get_rotation_input(snap, subID) :
    
    # define the mpb cutouts file
    inDir = 'S:/Cam/GitHub/TNG/mpb_cutouts_099/'
    cutout_file = inDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file, 'r') as hf :
            gas_coords = hf['PartType0']['Coordinates'][:]
            gas_sfrs = hf['PartType0']['StarFormationRate'][:]
            
            # convert mass units
            gas_masses = hf['PartType0']['Masses'][:]*1e10/cosmo.h
    except KeyError :
        gas_masses, gas_sfrs, gas_coords = None, None, None
    
    try :
        with h5py.File(cutout_file, 'r') as hf :
            star_coords = hf['PartType4']['Coordinates'][:]
            star_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
            star_metals = hf['PartType4']['GFM_Metallicity'][:]
            
            # convert mass units
            star_gfm = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h
            star_masses = hf['PartType4']['Masses'][:]*1e10/cosmo.h
        
        # limit particles to those that have positive formation times
        mask = (star_ages > 0)
        
        star_gfm, star_masses = star_gfm[mask], star_masses[mask]
        star_coords, star_ages = star_coords[mask], star_ages[mask]
        star_metals = star_metals[mask]
    except KeyError :
        star_ages, star_gfm, star_masses, star_coords = None, None, None, None
    
    return (gas_masses, gas_sfrs, gas_coords, star_ages, star_gfm, star_masses,
            star_coords, star_metals)

def load_galaxy_attributes_massive(subID, snap, loc_only=False) :
    
    infile = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Re = hf['Re'][:]
        centers = hf['centers'][:]
    
    # mask to the massive galaxies
    mask = (logM[:, -1] >= 9.5)
    subIDfinals = subIDfinals[mask]
    subIDs = subIDs[mask]
    Re = Re[mask]
    centers = centers[mask]
    
    # find the location of the subID within the entire massive sample
    loc = np.where(subIDfinals == subID)[0][0]
    
    if loc_only :
        return loc
    else :
        return (times[snap], subIDs[loc, snap], (logM[mask])[loc],
                Re[loc, snap], list(centers[loc, snap]))

def open_cutout(infile, shape=False, simple=False) :
    
    with fits.open(infile) as hdu :
        if simple :
            data = hdu[0].data
            shape = data.shape
        else :
            data = hdu[0].data
            shape = data.shape
            hdr = hdu[0].header
            redshift = hdr['Z']
            exptime = hdr['EXPTIME']
            area = hdr['AREA']
            photfnu = hdr['PHOTFNU']
            scale = hdr['SCALE']
    
    if simple :
        return data, shape
    else :
        return data, shape, redshift, exptime, area, photfnu, scale

def save_cutout(data, outfile, exposure, det_area, photfnu, scale, redshift) :
    
    hdu = fits.PrimaryHDU(data)
    
    hdr = hdu.header
    hdr['Z'] = redshift
    hdr.comments['Z'] = 'object spectroscopic redshift--by definition'
    hdr['EXPTIME'] = exposure
    hdr.comments['EXPTIME'] = 'exposure duration (seconds)--calculated'
    hdr['AREA'] = det_area
    hdr.comments['AREA'] = 'detector area (cm2)--calculated'
    hdr['PHOTFNU'] = photfnu
    hdr.comments['PHOTFNU'] = 'inverse sensitivity, Jy*sec*cm2/electron'
    hdr['SCALE'] = scale
    hdr.comments['SCALE'] = 'Pixel size (arcsec) of output image'
    hdr['BUNIT'] = 'electron'
    hdr.comments['BUNIT'] = 'Physical unit of the array values'
    
    hdu.writeto(outfile)
    
    return

def save_massive_galaxy_sample() :
    
    outfile = 'tools/TNG_massive_sample.fits'
    
    # open requisite information about the sample
    sample_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(sample_file, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    # get the quenching mechanisms
    mechanisms_file = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_mechanism.hdf5'
    with h5py.File(mechanisms_file, 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    
    # select massive galaxies
    mask = (logM[:, -1] >= 9.5) # 1666 entries, but len(mask) = 8260
    
    # mask all attributes to select only the massive population
    subIDs = subIDs[mask]     # (1666, 100)
    logM = logM[mask]         # (1666, 100)
    SFHs = SFHs[mask]         # (1666, 100)
    SFMS = SFMS[mask]         # (1666, 100)
    Res = Res[mask]           # (1666, 100)
    centers = centers[mask]   # (1666, 100, 3)
    ionsets = ionsets[mask]   # (1666)
    tonsets = tonsets[mask]   # (1666)
    iterms = iterms[mask]     # (1666)
    tterms = tterms[mask]     # (1666)
    quenched = quenched[mask] # (1666)
    mechs = mechs[mask]       # (1666)
    
    # loop through all quenched galaxies
    all_subIDfinals = []
    all_subIDs = []
    all_snaps = []
    all_logMs = []
    all_SFRs = []
    all_Res = []
    all_centers = []
    all_redshifts = []
    all_episode_progresses = []
    all_mechanisms = []
    for (q_subID, q_logM, q_SFR, q_Res, q_centers, ionset, tonset, iterm, tterm, 
         q_mech) in zip(subIDs[quenched], logM[quenched], SFHs[quenched],
        Res[quenched], centers[quenched], ionsets[quenched], tonsets[quenched],
        iterms[quenched], tterms[quenched], mechs[quenched]) :
        for snap in range(ionset, iterm+1) :
            # get the relevant quantities and append those values
            all_subIDfinals.append(q_subID[-1])
            all_subIDs.append(q_subID[snap])
            all_snaps.append(snap)
            all_logMs.append(q_logM[snap])
            all_SFRs.append(q_SFR[snap])
            all_Res.append(q_Res[snap])
            all_centers.append(q_centers[snap])
            all_redshifts.append(redshifts[snap])
            all_episode_progresses.append((times[snap] - tonset)/(tterm - tonset))
            all_mechanisms.append(q_mech)
        
        # get the stellar mass of the quenched galaxy at onset
        quenched_logM_onset = q_logM[ionset]
        
        # find galaxies that are on the SFMS from onset until termination
        always_on_SFMS = np.all(SFMS[:, ionset:iterm+1] > 0, axis=1) # (1666)
        
        # compare stellar masses for all galaxies, looking for small differences
        similar_mass = (np.abs(logM[:, ionset] - quenched_logM_onset) <= 0.1) # (1666)
        
        # create a final mask where both conditions are true
        final = (similar_mass & always_on_SFMS) # (1666)
        N_final = np.sum(final)
        
        if N_final > 0 : # 13 galaxies don't have any
            # loop over every control SF galaxy for the quenched galaxy
            for loc in np.argwhere(final) :
                for snap in range(ionset, iterm+1) :
                    # get the relevant quantities and append those values
                    all_subIDfinals.append(subIDs[loc, -1][0]) # the SF subID
                    all_subIDs.append(subIDs[loc, snap][0])
                    all_snaps.append(snap)
                    all_logMs.append(logM[loc, snap][0])
                    all_SFRs.append(SFHs[loc, snap][0])
                    all_Res.append(Res[loc, snap][0])
                    all_centers.append(centers[loc, snap][0])
                    all_redshifts.append(redshifts[snap])
                    all_episode_progresses.append((times[snap] - tonset)/(tterm - tonset))
                    all_mechanisms.append(0)
    
    table = Table([all_subIDfinals, all_subIDs, all_snaps, all_logMs,
                   all_SFRs, all_Res, all_centers, all_redshifts,
                   all_episode_progresses, all_mechanisms],
                  names=('subIDfinal', 'subID', 'snapshot', 'logM', 'SFR',
                         'Re', 'center', 'redshift', 'episode_progress',
                         'mechanism'))
    table.write(outfile)
    
    return

def load_massive_galaxy_sample() :
    return Table.read('tools/TNG_massive_sample.fits')
