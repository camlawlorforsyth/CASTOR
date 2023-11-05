
import os
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u

from core import open_cutout

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def determine_fluxes(cluster, filters) :
    '''
    Determine the flux in every annulus/bin for a given object for a given
    filter. Then move to the next subsequent filter and determine the fluxes
    in the corresponding bins for that filter. Repeat for all filters.
    Create a table which includes all determined fluxes. Save table to file
    for subsequent use with Prospector.
    
    Parameters
    ----------
    cluster : string
        Operate on the files of this cluster.
    filters : list
        The filterset for the cluster.
    
    Returns
    -------
    None.
    
    '''
    
    use_table = Table.read('{}/{}_sample-with-use-cols.fits'.format(
        cluster, cluster))
    
    use_columns = [col for col in use_table.colnames if col.endswith('_use')]
    IDs = use_table['id']
    
    os.makedirs('{}/photometry'.format(cluster), exist_ok=True) # ensure the
        # output directory for the photometric tables is available
    
    for ID in IDs :
        row = np.where(IDs == ID)
        use_vals = (np.array(list(use_table[use_columns][row][0])) == 'TRUE')
        
        use_dict = dict(zip(use_columns, use_vals))
        
        outfile = '{}/photometry/{}_ID_{}_photometry.fits'.format(cluster,
                                                                  cluster, ID)
        
        bin_data = np.load('{}/bins/{}_ID_{}_annuli.npz'.format(cluster,
                                                                cluster, ID))
        bins_image = bin_data['image']
        sma, smb = bin_data['sma'], bin_data['smb']
        flux, err = bin_data['flux'], bin_data['err']
        nPixels = bin_data['nPixels']
        widths, PAs = bin_data['width'], bin_data['pa']
        
        numBins = np.nanmax(bins_image) + 1 # accounts for python 0-index
        
        FUV_mag, U_mag = use_table['M_AB_FUV'][row], use_table['M_AB_U'][row]
        V_mag, J_mag = use_table['M_AB_V'][row], use_table['M_AB_J'][row]
        
        if not np.isnan(numBins) :
            photometry = Table()
            photometry['bin'] = range(int(numBins))
            photometry['sma'], photometry['smb'] = sma, smb
            photometry['flux'], photometry['err'] = flux, err
            photometry['SN'], photometry['nPixels'] = flux/err, nPixels
            photometry['width'], photometry['PA'] = widths, PAs
            
            photometry['FUV_mag'] = [FUV_mag]*int(numBins)
            photometry['U_mag'] = [U_mag]*int(numBins)
            photometry['V_mag'] = [V_mag]*int(numBins)
            photometry['J_mag'] = [J_mag]*int(numBins)
            
            for filt in filters :
                sci_file = '{}/cutouts/{}_ID_{}_{}.fits'.format(
                    cluster, cluster, ID, filt)
                noise_file = '{}/cutouts/{}_ID_{}_{}_noise.fits'.format(
                    cluster, cluster, ID, filt)
                segmap_file = '{}/cutouts/{}_ID_{}_segmap.fits'.format(
                    cluster, cluster, ID)
                bcg_segmap_file = '{}/cutouts/{}_ID_{}_segmap-bCG.fits'.format(
                    cluster, cluster, ID)
                
                (sci, dim, photfnu, r_e,
                 redshift, sma, smb, pa) = open_cutout(sci_file)
                noise = open_cutout(noise_file, simple=True)
                segMap = open_cutout(segmap_file, simple=True)
                bCGsegMap = open_cutout(bcg_segmap_file, simple=True)
                
                # make a copy of the science image and noise image
                new_sci = sci.copy()
                new_noise = noise.copy()
                
                # mask the copied images based on the segmentation map, but
                # don't mask out the sky
                if ID >= 20000 : # the bCGs aren't in the segmap, so mask any other galaxy
                    new_sci[(segMap > 0) | ((bCGsegMap > 0) & (bCGsegMap != ID))] = 0
                    new_noise[(segMap > 0) | ((bCGsegMap > 0) & (bCGsegMap != ID))] = 0
                else : # for the non-bCGs, mask out pixels associated with
                    new_sci[(segMap > 0) & (segMap != ID)] = 0 # other galaxies
                    new_noise[(segMap > 0) & (segMap != ID)] = 0
                
                # save relevant information for running prospector into the table
                length = len(range(int(numBins)))
                photometry['R_e'] = [r_e]*length*u.pix
                photometry['z'] = [redshift]*length
                lumDist = cosmo.luminosity_distance(redshift)
                photometry['lumDist'] = [lumDist.value]*length*u.Mpc
                
                fluxes, uncerts, invalid = [], [], []
                for val in range(int(numBins)) :
                    # mask = np.where(bins_image == val)
                    temp_sci, temp_noise = new_sci.copy(), new_noise.copy()
                    
                    # masked_sci = new_sci[mask]
                    # flux = photfnu*np.nansum(masked_sci)
                    temp_sci[bins_image != val] = np.nan
                    flux = photfnu*np.nansum(temp_sci)
                    fluxes.append(flux)
                    
                    # masked_noise = new_noise[mask]
                    # uncert = photfnu*np.sqrt(np.nansum(np.square(masked_noise)))
                    temp_noise[bins_image != val] = np.nan
                    uncert = photfnu*np.sqrt(np.nansum(np.square(temp_noise)))
                    uncerts.append(uncert)
                    
                    pix_sci = temp_sci.copy()
                    pix_sci[pix_sci != 0] = np.nan
                    pix_sci[pix_sci == 0] = 1
                    invalid_pix = np.nansum(pix_sci)
                    invalid.append(invalid_pix)
                
                photometry[filt + '_flux'] = fluxes*u.Jy
                photometry[filt + '_err'] = uncerts*u.Jy
                
                valid = nPixels - np.array(invalid)
                photometry[filt + '_nPix'] = np.int_(valid)
                
                for key, value in use_dict.items() :
                    if filt in key.split('_')[0] :
                        photometry[key] = [value]*length
            
            photometry.write(outfile)
    
    return
