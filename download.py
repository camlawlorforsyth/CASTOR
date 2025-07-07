
from os.path import exists
import numpy as np

import h5py
import requests

from core import load_massive_galaxy_sample

def download_all_necessary_mpb_cutouts() :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # create IDs for each galaxy
    ids = sample['snapshot'].value.astype(str) + '_' + sample['subID'].value.astype(str)
    
    # determine the unique galaxy/snapshot pairs
    unique_snaps, unique_subIDs = np.stack(
        np.char.split(np.unique(ids), sep='_')).astype(int).T # 32,710
    
    '''
    snaps_to_download = []
    subIDs_to_download = []
    # missing_metallicities = 0
    for snap, subID in zip(unique_snaps, unique_subIDs) :
        infile = 'S:/Cam/University/GitHub/TNG/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(
            snap, subID)
        with h5py.File(infile, 'r') as hf :
            if 'GFM_Metallicity' not in hf['PartType4'].keys() :
                # missing_metallicities += 1 # 30,133
                snaps_to_download.append(snap)
                subIDs_to_download.append(subID)
    '''
    
    download_mpb_cutouts(unique_snaps, unique_subIDs)
    
    return

def download_mpb_cutouts(snaps, subIDs) :
    
    outDir = 'S:/Cam/University/GitHub/TNG/mpb_cutouts_099/'
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime'}
    star_params = set(['Coordinates', 'GFM_InitialMass', 'GFM_Metallicity',
                       'GFM_StellarFormationTime'])
    
    # loop over all the required mpb cutouts
    for snap, subID in zip(snaps, subIDs) :
        
        # define the URL for the galaxy at the redshift of interest
        url = 'https://www.tng-project.org/api/TNG50-1/snapshots/{}/subhalos/{}'.format(
            snap, subID)
        
        # save the cutout file into the output directory if it doesn't exist
        filename = 'cutout_{}_{}.hdf5'.format(snap, subID)
        
        # check to make sure we download only necessary files
        # if not exists(outDir + filename) :
            # proceed = True
        # else :
        try :
            with h5py.File(outDir + filename, 'r') as hf :
                star_keys = set(list(hf['PartType4'].keys()))
            arrays_present = (star_params <= star_keys)
            
            if not arrays_present :
            #     print('re-downloading {}'.format(filename))
                get(url + '/cutout.hdf5', directory=outDir, params=params,
                    filename=filename)
        except OSError :
            get(url + '/cutout.hdf5', directory=outDir, params=params,
                filename=filename)
        
        print('{} done'.format(filename))
    
    return

def get(path, directory=None, params=None, filename=None) :
    # https://www.tng-project.org/data/docs/api/
    
    try :
        # make HTTP GET request to path
        headers = {'api-key':'0890bad45ac29c4fdd80a1ffc7d6d27b'}
        rr = requests.get(path, params=params, headers=headers)
        
        # raise exception if response code is not HTTP SUCCESS (200)
        # rr.raise_for_status()
        if rr.status_code == 200 :
        
            if rr.headers['content-type'] == 'application/json' :
                return rr.json() # parse json responses automatically
            
            if 'content-disposition' in rr.headers :
                if not filename :
                    filename = rr.headers['content-disposition'].split('filename=')[1]
                
                with open(directory + filename, 'wb') as ff :
                    ff.write(rr.content)
                return filename # return the filename string
            
            return rr
        else :
            print('Error: {} for file {}'.format(rr.status_code, filename))
    except Exception :
        print('Exception: for file {}'.format(filename))

def unique_galaxies_count() :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # create IDs for each galaxy
    ids = sample['snapshot'].value.astype(str) + '_' + sample['subID'].value.astype(str)
    
    # determine the unique galaxy/snapshot pairs
    unique_snaps, unique_subIDs = np.stack(
        np.char.split(np.unique(ids), sep='_')).astype(int).T
    
    # process every galaxy/snapshot pair, where directory holds 145,947 files
    missing_snaps = []
    missing_subIDs = []
    for snap, subID in zip(unique_snaps, unique_subIDs) :
        file = 'S:/Cam/University/GitHub/TNG/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(
            snap, subID)
        if not exists(file) :
            missing_snaps.append(snap)
            missing_subIDs.append(subID)
    missing_snaps = np.array(missing_snaps).tolist()
    missing_subIDs = np.array(missing_subIDs).tolist()
    
    return
