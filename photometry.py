
import os
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
from photutils.aperture import CircularAnnulus, CircularAperture

from core import load_massive_galaxy_sample

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def determine_all_photometry() :
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # select only the quenched galaxies at the first snapshot >=75% of the way
    # through their quenching episodes
    mask = (((sample['mechanism'] == 1) | (sample['mechanism'] == 3)) &
        (sample['episode_progress'] >= 0.75))
    sample = sample[mask]
    
    # use the first snapshot >=75% of the way through the quenching episode,
    # but not any additional snapshots, for testing purposes
    mask = np.full(len(sample), False)
    idx = 0
    for subIDfinal in np.unique(sample['subIDfinal']) :
        mask[idx] = True
        idx += len(np.where(sample['subIDfinal'] == subIDfinal)[0])
    sample = sample[mask]
    
    # process every galaxy/snapshot pair
    for subID, snap, Re in zip(sample['subID'], sample['snapshot'], sample['Re']) :
        outfile = 'photometry/{}_{}_photometry.fits'.format(snap, subID)
        if not os.path.exists(outfile) :
            determine_photometry_circular_annuli(snap, subID, Re)
        print('snap {} subID {} done'.format(snap, subID))
    
    return

def determine_photometry_circular_annuli(snap, subID, Re, model_redshift=0.5,
                                         fov=10, save=True) :
    
    # open the input image
    infile = 'cutouts/{}_{}_z_{:03}.fits'.format(snap, subID,
        str(model_redshift).replace('.', ''))
    with fits.open(infile) as hdu :
        hdr = hdu[0].header
        images = hdu[0].data*u.Jy # Jy [per pixel]
    plate_scale = hdr['CDELT1']*u.arcsec/u.pix # the plate scale of the images
    assert hdr['REDSHIFT'] == model_redshift
    
    # get the filters
    filters = [hdr['FILTER{}'.format(i)] for i in range(len(hdr['FILTER*']))]
    
    # determine the center of the image
    cent = int((images.shape[1] - 1)/2)
    center = (cent, cent)
    
    # convert Re, 1 kpc into pixels
    Re_pix = (Re*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    kpc_pix = (1*u.kpc*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale).value
    
    # get the edges of the circular annuli in units of pixels for masking
    edges_pix = np.linspace(0, 5, 21)*Re_pix # edges in units of Re
    
    # add the photometry for every annulus and every filter into a table
    photometry = Table()
    nPixel_profile = np.full(22, -1.0)
    for i, filt in enumerate(filters) :
        # open the science images and the corresponding noise images
        sci = images[2*i]
        noise = images[2*i + 1]
        
        signal_profile = np.full(22, -99.)*u.Jy
        noise_profile = np.full(22, -99.)*u.Jy
        for i, (start, end) in enumerate(zip(edges_pix, edges_pix[1:])) :
            # dist = calculate_distance_to_center((data.shape[1], data.shape[2]))
            # if end == edges_pix[-1] :
            #     mask = (dist >= start) & (dist <= end)
            # else :
            #     mask = (dist >= start) & (dist < end)
            # signal_profile[i] = np.sum(sci[mask])
            # noise_profile[i] = np.sqrt(np.sum(np.square(noise[mask])))
            # determine the number of pixels for the pixelized TNG profiles
            # nPixel_profile[i] = np.sum(mask)
            
            if start == 0 :
                ap = CircularAperture(center, end)
            else :
                ap = CircularAnnulus(center, start, end)
            flux, err = ap.do_photometry(sci, noise)
            
            signal_profile[i] = flux[0]
            noise_profile[i] = err[0]
            nPixel_profile[i] = ap.area # the pixel areas per annulus
        
        # complete an additional inner 1 kpc aperture
        inner_kpc = CircularAperture(center, kpc_pix)
        inner_kpc_flux, inner_kpc_err = inner_kpc.do_photometry(sci, noise)
        signal_profile[20] = inner_kpc_flux[0]
        noise_profile[20] = inner_kpc_err[0]
        nPixel_profile[20] = inner_kpc.area
        
        # complete an additional integrated aperture out to 5 Re
        integrated_ap = CircularAperture(center, edges_pix[-1])
        int_flux, int_err = integrated_ap.do_photometry(sci, noise)
        signal_profile[21] = int_flux[0]
        noise_profile[21] = int_err[0]
        nPixel_profile[21] = integrated_ap.area
        
        # add the filter profiles into the photometry table
        photometry[filt + '_flux'] = signal_profile
        photometry[filt + '_err'] = noise_profile
        # photometry[filt + '_flux'] = sci.flatten()/exptime/area*photfnu*u.Jy
        # photometry[filt + '_err'] = noise.flatten()/exptime/area*photfnu*u.Jy
    
    # make unique IDs for each annulus
    pixel = ['{}_{}_bin_{}'.format(snap, subID, i) for i in np.arange(len(photometry))]
    # pixel = np.arange(len(photometry))
    
    # correct the final IDs
    pixel[20] = '{}_{}_bin_kpc'.format(snap, subID)
    pixel[21] = '{}_{}_bin_int'.format(snap, subID)
    
    # add the IDs to the photometric table, at the beginning of the table
    photometry.add_column(pixel, name='id', index=0)
    
    # add the number of pixels per bin into the table
    photometry['nPix'] = nPixel_profile
    
    # add the redshift into the table
    photometry['z_spec'] = np.full_like(pixel, model_redshift)
    
    # sort the table according to distance from the center of the image
    # sort = np.argsort(photometry['distance'].data)
    # photometry = photometry[sort]
    
    # mask the photometry table to those pixels within 5 Re
    # mask = (photometry['distance'].data <= fov/2*Re_pix)
    # photometry = photometry[mask]
    
    if save :
        os.makedirs('photometry/', exist_ok=True) # ensure the output directory
            # for the photometric tables is available
        
        outfile = 'photometry/{}_{}_photometry.fits'.format(snap, subID)
        photometry.write(outfile)
    
    return

def join_all_photometry(model_redshift=0.5, save=True) :
    
    # set a dictionary to translate the filter names to FAST++-compliant names
    translate = {'castor_uv':'F314', 'castor_uvL':'F315', 'castor_uS':'F316',
                 'castor_u':'F317', 'castor_g':'F318',
                 
                 'euclid_ie':'F319', 'euclid_ye':'F320', 'euclid_je':'F321',
                 'euclid_he':'F322',
                 
                 'hst_f218w':'F323', 'hst_f225w':'F324', 'hst_f275w':'F325', 
                 'hst_f336w':'F326', 'hst_f390w':'F327', 'hst_f438w':'F328',
                 'hst_f435w':'F329', 'hst_f475w':'F330', 'hst_f555w':'F331',
                 'hst_f606w':'F332', 'hst_f625w':'F333', 'hst_f775w':'F334',
                 'hst_f814w':'F335', 'hst_f850lp':'F336', 'hst_f105w':'F337',
                 'hst_f110w':'F338', 'hst_f125w':'F339', 'hst_f140w':'F340',
                 'hst_f160w':'F341',
                 
                 'jwst_f070w':'F342', 'jwst_f090w':'F343', 'jwst_f115w':'F344',
                 'jwst_f150w':'F345', 'jwst_f200w':'F346', 'jwst_f277w':'F347',
                 'jwst_f356w':'F348', 'jwst_f410m':'F349', 'jwst_f444w':'F350',
                 'jwst_f560w':'F351', 'jwst_f770w':'F352', 'jwst_f1000w':'F353',
                 'jwst_f1130w':'F354', 'jwst_f1280w':'F355', 'jwst_f1500w':'F356',
                 'jwst_f1800w':'F357', 'jwst_f2100w':'F358', 'jwst_f2550w':'F359',
                 
                 'roman_f062':'F360', 'roman_f087':'F361', 'roman_f106':'F362',
                 'roman_f129':'F363', 'roman_f146':'F364', 'roman_f158':'F365',
                 'roman_f184':'F366', 'roman_f213':'F367'}
    
    # get the entire massive sample, including both quenched galaxies and
    # comparison/control star forming galaxies
    sample = load_massive_galaxy_sample()
    
    # select only the quenched galaxies at the first snapshot >=75% of the way
    # through their quenching episodes
    mask = (((sample['mechanism'] == 1) | (sample['mechanism'] == 3)) &
        (sample['episode_progress'] >= 0.75))
    sample = sample[mask]
    
    # use the first snapshot >=75% of the way through the quenching episode,
    # but not any additional snapshots, for testing purposes
    mask = np.full(len(sample), False)
    idx = 0
    for subIDfinal in np.unique(sample['subIDfinal']) :
        mask[idx] = True
        idx += len(np.where(sample['subIDfinal'] == subIDfinal)[0])
    sample = sample[mask]
    
    # define the filters that we want to create images for
    filters = ['castor_uv', 'castor_uvL', 'castor_uS', 'castor_u', 'castor_g',
               'roman_f106', 'roman_f129', 'roman_f158', 'roman_f184']
    
    # determine the names that will be used in the final photometric table
    names = ['id']
    for filt in filters :
        names.append(translate[filt])
        names.append(translate[filt].replace('F', 'E'))
    names.append('z_spec')
    
    # process every galaxy/snapshot pair
    tables_to_stack = []
    for subID, snap in zip(sample['subID'], sample['snapshot']) :
        # get the photometry for an individual galaxy
        infile = 'photometry/{}_{}_photometry.fits'.format(snap, subID)
        table = Table.read(infile)
        
        # get photometry from the table, and include additional redshift info
        columns = [table['id']]
        for filt in filters :
            columns.append(table[filt + '_flux'].data)
            columns.append(table[filt + '_err'].data)
        columns.append(table['z_spec'].data)
        
        # append the translated table to the list of tables to stack
        tables_to_stack.append(Table(columns, names=names))
    
    # stack the photometry, thereby creating a master table of photometry to fit
    final = vstack(tables_to_stack)
    
    # save only the integrated apertures
    # ids = np.stack(np.char.split(
    #     np.array(final['id'].value, dtype=str), sep='_').ravel())[:, 2]
    # final = final[ids == 'int']
    
    if save :
        outfile = 'photometry/photometry_2April2025.cat'
        if not os.path.exists(outfile) :
            final.write(outfile, format='ascii.commented_header')
    
    return
