
import numpy as np

from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.table import Table
import astropy.units as u
import h5py

import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def compare_fast_output_to_tng(subID, snap, population='quenched', display=False) :
    
    # get galaxy attributes from TNG
    time, mpbsubID, tngRe, center = load_galaxy_attributes(subID, snap)
    
    # get the annuli map, derived using SourceXtractor++ morphological values
    bins_image, dim, numBins = load_annuli_map(population, subID)
    
    # get stellar mass and SFR maps direct from TNG, assuming 100 Myr for the
    # duration of star formation
    edges = np.linspace(-10*tngRe, 10*tngRe, dim + 1) # kpc
    tng_Mstar_map, tng_SFR_map = spatial_plot_info(time, snap, mpbsubID,
        center, tngRe, edges, 100*u.Myr)
    tng_Mstar_map = np.rot90(tng_Mstar_map, k=3) # rotate images to match SKIRT
    tng_SFR_map = np.rot90(tng_SFR_map, k=3)
    
    if display :
        plt.display_image_simple(tng_Mstar_map)
        plt.display_image_simple(tng_SFR_map)
        plt.display_image_simple(bins_image, lognorm=False)
    
    # get the output from FAST++
    lmass, lmass_lo, lmass_hi, lsfr, lsfr_lo, lsfr_hi = load_fast_fits(subID)
    
    # get basic information from the photometric table
    nPixels, redshift, scale, rr, Re = load_photometric_information(population, subID)
    
    # determine the area of a single pixel, in arcsec^2
    pixel_area = np.square(scale*u.pix)
    
    # determine the physical projected area of a single pixel
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc/u.arcsec)
    pixel_area_physical = np.square(kpc_per_arcsec)*pixel_area
    
    # determine the projected physical areas of every annulus
    physical_area = pixel_area_physical*nPixels/u.pix
    physical_area_pc2 = physical_area.to(np.square(u.pc))
    
    # convert the pixel values to physical sizes, mostly for plotting purposes
    rr = rr*scale*kpc_per_arcsec # kpc
    Re = Re*scale*kpc_per_arcsec # kpc
    # rs = np.log10(rr/Re)
    
    # loop through the elliptical annuli, binning together valid pixels
    tng_annuli_masses = np.full(numBins, 0.0)
    tng_annuli_sfrs = np.full(numBins, 0.0)
    for val in range(numBins) :
        
        # copy the maps so that subsequent versions aren't erroneously used
        mass_map, sfr_map = tng_Mstar_map.copy(), tng_SFR_map.copy()
        
        # mask out pixels that aren't in the annulus
        mass_map[bins_image != val] = np.nan
        sfr_map[bins_image != val] = np.nan
        
        # sum values and place into array
        tng_annuli_masses[val] = np.nansum(mass_map) # solMass
        tng_annuli_sfrs[val] = np.nansum(sfr_map) # solMass/yr
    
    # check the integrated stellar masses
    # tng_integrated_mass = np.log10(np.trapz(tng_annuli_masses))
    # fast_integrated_mass = np.log10(np.trapz(np.power(10, lmass)))
    # print(tng_integrated_mass, fast_integrated_mass)
    
    # find the surface mass/SFR densities of the idealized TNG maps
    tng_Sigma = np.log10(tng_annuli_masses/physical_area_pc2.value)
    tng_Sigma_SFR = np.log10(tng_annuli_sfrs/physical_area.value)
    
    # set the uncertainty for the TNG values
    zeros = np.zeros_like(tng_Sigma)
    
    # convert stellar masses to surface mass densities
    Sigma = np.log10(np.power(10, lmass)/physical_area_pc2.value)
    Sigma_lo = np.log10(np.power(10, lmass_lo)/physical_area_pc2.value)
    Sigma_hi = np.log10(np.power(10, lmass_hi)/physical_area_pc2.value)
    
    # convert star formation rates to surface star formation rate densities
    Sigma_SFR = np.log10(np.power(10, lsfr)/physical_area.value)
    Sigma_SFR_lo = np.log10(np.power(10, lsfr_lo)/physical_area.value)
    Sigma_SFR_hi = np.log10(np.power(10, lsfr_hi)/physical_area.value)
    
    # set plot attributes
    xlabel = r'$R$ (kpc)' # r'$\log{(R/R_{\rm e})}$'
    ylabel1 = r'$\log{(\Sigma/{\rm M}_{\odot}~{\rm pc}^{2})}$'
    ylabel2 = r'$\log{(\Sigma_{\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{2})}$'
    
    # plot the radial profiles
    xs = [rr, rr, rr, rr]
    ys = [Sigma, tng_Sigma, Sigma_SFR, tng_Sigma_SFR]
    lo = [Sigma - Sigma_lo, zeros, Sigma_SFR - Sigma_SFR_lo, zeros]
    hi = [Sigma_hi - Sigma, zeros, Sigma_SFR_hi - Sigma_SFR, zeros]
    labels = ['fit', 'TNG', '', '']
    colors = ['k', 'k', 'b', 'b']
    markers = ['', '', '', '']
    styles = ['--', '-', '--', '-']
    
    plt.plot_multi_vertical_error(xs, ys, lo, hi, labels, colors, markers,
        styles, 2, xlabel=xlabel, ylabel1=ylabel1, ylabel2=ylabel2)
    
    return

def load_annuli_map(population, subID) :
    
    # open the annuli map to use for masking
    infile = 'bins/{}/subID_{}_annuli.npz'.format(population, subID)
    bin_data = np.load(infile)
    
    bins_image = bin_data['image']
    numBins = int(np.nanmax(bins_image) + 1) # accounts for python 0-index
    
    return bins_image, bins_image.shape[0], numBins

def load_fast_fits(subID) :
    
    # load fitted data coming out of FAST++
    data = np.loadtxt('fitting/photometry.fout', dtype=str)
    
    # define which rows to use, based on the 'binNum' containing the subID
    binNum = data[:, 0]
    use = np.char.find(binNum, str(subID))
    use[use < 0] = 1
    use = np.invert(use.astype(bool))

    # get the stellar mass and star formation rates
    lmass = data[:, 16].astype(float)[use]
    lmass_lo = data[:, 17].astype(float)[use]
    lmass_hi = data[:, 18].astype(float)[use]

    lsfr = data[:, 19].astype(float)[use]
    lsfr_lo = data[:, 20].astype(float)[use]
    lsfr_hi = data[:, 21].astype(float)[use]
    
    # lssfr = data[:, 22].astype(float)[use]
    
    return lmass, lmass_lo, lmass_hi, lsfr, lsfr_lo, lsfr_hi

def load_galaxy_attributes(subID, snap) :
    
    infile = 'D:/Documents/GitHub/TNG/TNG50-1/TNG50-1_99_sample(t).hdf5'
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDs = hf['subIDs'][:].astype(int)
        # logM = hf['logM'][:]
        Re = hf['Re'][:]
        centers = hf['centers'][:]
    
    # find the location of the subID within the entire sample
    loc = np.where(subIDfinals == subID)[0][0]
    
    return times[snap], subIDs[loc, snap], Re[loc, snap], list(centers[loc, snap])

def load_photometric_information(population, subID) :
    
    # get information that was put into FAST++ about the elliptical annuli
    infile = 'photometry/{}/subID_{}_photometry.fits'.format(population, subID)
    table = Table.read(infile)
    
    nPixels = table['nPixels'].data*u.pix
    redshift = table['z'].data[0]
    scale = table['scale'].data[0]*u.arcsec/u.pix # arcsec/pixel
    
    # define the centers of the elliptical annuli for plotting purposes
    rr = (table['sma'].data - 0.5*table['width'].data)*u.pix # pixels
    
    # get the half light radius
    Re = table['R_e'].data*u.pix
    
    return nPixels.astype(int), redshift, scale, rr, Re

def spatial_plot_info(time, snap, mpbsubID, center, Re, edges, delta_t) :
    
    # open the TNG cutout and retrieve the relevant information
    cutout_file = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, mpbsubID)
    with h5py.File(cutout_file, 'r') as hf :
        coords = hf['PartType4']['Coordinates'][:]
        ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        Mstar = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h # solMass
    
    # limit particles to those that have positive formation times
    mask = (ages > 0)
    coords, ages, Mstar = coords[mask], ages[mask], Mstar[mask]
    
    # don't project using face-on version
    dx, dy, dz = (coords - center).T
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (ages >= limit)
    
    sf_masses = Mstar[mask]
    sf_dx = dx[mask]
    sf_dy = dy[mask]
    # sf_dz = dz[mask]
    
    # create 2D histograms of the particles and SF particles
    hh, _, _ = np.histogram2d(dx/Re, dy/Re, bins=(edges, edges),
                              weights=Mstar)
    hh = hh.T
    
    hh_sf, _, _ = np.histogram2d(sf_dx/Re, sf_dy/Re, bins=(edges, edges),
                                 weights=sf_masses)
    hh_sf = hh_sf.T
    
    return hh, hh_sf/delta_t.to(u.yr).value
