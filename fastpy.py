
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
from scipy.integrate import quad

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def apply_redshift(zz) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-sfh2sed.cpp#L174-L178
    
    dist_Mpc_to_cgs = 3.085677581491367e+24 # [cm/Mpc]
    lum_sol_to_cgs = 3.828e33               # [erg/s/Lsol]
    factor = 1e19*lum_sol_to_cgs/np.square(dist_Mpc_to_cgs)
    lum2fl = factor/(4.0*np.pi*(1.0 + zz)*
                     np.square(cosmo.luminosity_distance(zz).value))
    
    return lum2fl

def calculate_chi2(wphot, model, phot, phot_e, zz=0.5) :
    # from FAST++
    
    tpl_err = get_template_error(wphot, zz=zz)
    
    # determine the weights for the photometric points, based on the photometric
    # uncertainties and the template error function, which is used in FAST++
    # by default
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-fitter.cpp#L538-L543
    weight = 1.0/np.sqrt(np.square(phot_e) + np.square(tpl_err*phot))
    
    # find the weighted observed and model fluxes
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-fitter.cpp#L545-L546
    wflux = phot*weight
    wmodel = model*weight
    
    # find the scale
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-fitter.cpp#L555
    scale = np.sum(wmodel*wflux)/np.sum(wmodel*wmodel)
    
    # compute (regular) chi2
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-fitter.cpp#L579-L586
    chi2 = np.sum(np.square(wflux - scale*wmodel))
    
    return chi2

def calzetti2000(waves) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-sfh2sed.cpp#L151-L166
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-gridder.cpp#L396-L411
    
    corr = np.zeros_like(waves) # waves in microns
    iRv = 1/4.05
    
    lo = 2.659*iRv*(-2.156 + 1.509/waves - 0.198/np.square(waves) +
                    0.011/np.power(waves, 3)) + 1
    hi = 2.659*iRv*(-1.857 + 1.040/waves) + 1
    corr[waves <= 0.63] = lo[waves <= 0.63]
    corr[waves > 0.63] = hi[waves > 0.63]
    
    corr[corr < 0] = 0
    
    return corr

def cardelli1989(waves) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-gridder.cpp#L413-L445
    
    corr = np.zeros_like(waves) # waves in microns
    iRv = 1/3.1
    
    iwaves = 1/waves # [1/um]
    ir = (0.574 - 0.527*iRv)*np.power(iwaves, 1.61)
    yy = iwaves - 1.82
    opt = (1 + (0.17699 + 1.41338*iRv)*yy - (0.50447 - 2.28305*iRv)*np.square(yy) - 
           (0.02427 - 1.07233*iRv)*np.power(yy, 3) + (0.72085 - 5.38434*iRv)*np.power(yy, 4) +
           (0.01979 - 0.62251*iRv)*np.power(yy, 5) - (0.77530 - 5.30260*iRv)*np.power(yy, 6) +
           (0.32999 - 2.09002*iRv)*np.power(yy, 7))
    uv1 = (1.752 - 3.090*iRv - (0.316 - 1.825*iRv)*iwaves -
           0.104/(np.square(iwaves - 4.67) + 0.341) + 1.206/(np.square(iwaves - 4.62) + 0.263)*iRv)
    uv2 = (1.752 - 3.090*iRv - (0.316 - 1.825*iRv)*iwaves -
           0.104/(np.square(iwaves - 4.67) + 0.341) +
           1.206/(np.square(iwaves - 4.62) + 0.263)*iRv -
           0.04473*np.square(iwaves - 5.9) - 0.009779*np.power(iwaves - 5.9, 3) +
           (0.2130*np.square(iwaves - 5.9) + 0.1207*np.power(iwaves - 5.9, 3))*iRv)
    xx = iwaves - 8.0
    fuv = (-1.073 + 13.670*iRv - (0.628 - 4.257*iRv)*xx +
           (0.137 - 0.420*iRv)*np.square(xx) - (0.070 - 0.374*iRv)*np.power(xx, 3))
    
    corr[iwaves <= 1.1] = ir[iwaves <= 1.1]
    corr[(1.1 < iwaves) & (iwaves <= 3.3)] = opt[(1.1 < iwaves) & (iwaves <= 3.3)]
    corr[(3.3 < iwaves) & (iwaves <= 5.9)] = uv1[(3.3 < iwaves) & (iwaves <= 5.9)]
    corr[(5.9 < iwaves) & (iwaves <= 8)] = uv2[(5.9 < iwaves) & (iwaves <= 8)]
    corr[iwaves > 8] = fuv[iwaves > 8]
    
    corr[corr < 0] = 0
    
    return corr

def dpl_from_fit(ts, alpha, beta, tau, norm=1.0) :
    # construct a double power law (dpl) SFH based on bestfit values
    return dplt_from_fit(ts, alpha, beta, tau, 1, norm=norm)

def dplt_from_fit(ts, alpha, beta, tau, rr, norm=1.0) :
    # construct a double power law + truncation (dplt) SFH based on bestfit values
    
    # adapted from
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...873...44C/abstract (Eq. 4)
    
    # create the SFH
    sfh = 1/(np.power(ts/tau, alpha) + np.power(ts/tau, -beta))
    
    # apply the truncation for the last 100 Myr
    sfh[ts >= np.max(ts) - 100] = rr*sfh[ts >= np.max(ts) - 100]
    
    if norm != 1 :
        sfh *= norm/(np.sum(sfh)*1e6)
    
    return sfh

def dt_from_fit(ts, ltau, lage, norm=1.0) :
    # construct a delay-tau (dt) SFH based on bestfit values
    return dtt_from_fit(ts, ltau, lage, 1, norm=norm)

def dtt_from_fit(ts, ltau, lage, rr, norm=1.0) :
    # construct a delay-tau + truncation (dtt) SFH based on bestfit values
    
    # adapted from
    # https://gitlab.lam.fr/cigale/cigale/-/blob/master/pcigale/sed_modules/sfhdelayedbq.py#L77-L83
    
    # get tau and lage into Myr, and convert lage into regular time (ie. BB
    tau = np.power(10, ltau)/1e6 # Myr                               at t = 0)
    t0 = np.max(ts) - np.power(10, lage)/1e6 # Myr
    
    # create the SFH, given that the SFH before t0 must be zero
    sfh = (ts - t0)*np.exp(-(ts - t0)/tau)
    sfh[ts < t0] = 0
    
    # apply the truncation for the last 100 Myr
    sfh[ts >= np.max(ts) - 100] = np.power(10, rr)*sfh[ts >= np.max(ts) - 100]
    
    # normalize the SFH if requested
    if norm != 1 :
        sfh *= norm/(np.sum(sfh)*1e6)
    
    return sfh

def ftb_from_fit(ts, r0, r1, r2, r3, r4, norm=1.0) :
    # construct a fixed time bin (ftb) SFH based on bestfit values
    
    # adapted from
    # https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract (Fig. 4)
    
    # account for lookback time
    tlb = np.flip(np.max(ts) - ts)
    
    # create the SFH
    sfh = np.zeros_like(ts)
    sfh[tlb <= 100] = np.power(10, float(r0))
    sfh[(tlb > 100) & (tlb <= 300)] = np.power(10, float(r1))
    sfh[(tlb > 300) & (tlb <= 1000)] = np.power(10, float(r2))
    sfh[(tlb > 1000) & (tlb <= 3000)] = np.power(10, float(r3))
    sfh[tlb > 3000] = np.power(10, float(r4))
    sfh = np.flip(sfh) # flip back into regular time (ie. BB at t = 0)
    
    # normalize the SFH if requested
    if norm != 1 :
        sfh *= norm/(np.sum(sfh)*1e6)
    
    return sfh

def get_bc03_library(metal) :
    # read the model spectra from Bruzual & Charlot (2003) for a Chabrier (2003)
    # initial mass function with given metallicity
    
    metal = str(metal)[2:] # convert to string, remove leading zero and decimal
    table = Table.read('tools/bc03_lr_ch_z{}.ised_ASCII.fits'.format(metal))[0]
    
    ages = table['AGE']/1e6 # (221), [Myr]
    masses = table['MASS']  # (221), [Msol]
    seds = table['SED']     # (221, 1221), [Lsol AA^-1]
    
    return ages, masses, seds

def get_bc03_waves(metal) :
    # read the model wavelengths from Bruzual & Charlot (2003) for a Chabrier (2003)
    # initial mass function with given metallicity
    
    metal = str(metal)[2:] # convert to string, remove leading zero and decimal
    table = Table.read('tools/bc03_lr_ch_z{}.ised_ASCII.fits'.format(metal))[0]
    
    waves = table['LAMBDA'] # (1221), [Angstrom]
    
    return waves/1e4

def get_filter_waves(filters) :
    # adapted from FAST++, and verified by comparing with FAST++ output files
    
    lambda_phots = np.full(len(filters), -1.)
    
    for i, filt in enumerate(filters) :
        lam, trans = np.loadtxt(
            'passbands/passbands_micron/' + filt + '.txt', unpack=True)
        
        # find the photon distribution-based effective wavelength
        lambda_phots[i] = np.trapezoid(lam*lam*trans)/np.trapezoid(lam*trans)
    
    return lambda_phots

def get_lum2fl(zz=0.5) :
    # adapted from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-sfh2sed.cpp#L174-L178
    
    # determine the conversion factor to go from SED units of Lsol AA^-1 to
    # 10^-19 ergs s^-1 cm^-2 AA^-1
    lum = (1*u.solLum/u.AA).to(u.erg/u.s/u.AA)
    d_l = cosmo.luminosity_distance(zz).to(u.cm)
    lum2fl = 1e19*lum/(4*np.pi*(1 + zz)*np.square(d_l))
    
    return lum2fl.value

def get_model_fluxes(filters, waves, flx) :
    # adapted from FAST++
    # https://github.com/cschreib/fastpp/tree/master#adding-new-filters
    
    models = np.full(len(filters), -1.)
    for i, filt in enumerate(filters) :
        lam, trans = np.loadtxt(
            'passbands/passbands_micron/' + filt + '.txt', unpack=True)
        
        # interpolate the transmission curve to match the wavelengths in waves
        trans = np.interp(waves, lam, trans)
        
        # find the model fluxes
        models[i] = np.trapezoid(
            waves*waves*trans*flx)/np.trapezoid(waves*waves*trans)
    
    return models

def get_model_fluxes_alt(filters, waves, flx) :
    # adapted from FAST++
    # https://github.com/cschreib/fastpp/tree/master#adding-new-filters
    
    models = np.full(len(filters), -1.)
    for i, filt in enumerate(filters) :
        lam, trans = np.loadtxt(
            'passbands/passbands_micron/' + filt + '.txt', unpack=True)
        
        # alternatively, interpolate the SED to match the wavelengths in lam
        flux = np.interp(lam, waves, flx.copy())
        
        # find the model fluxes
        models[i] = np.trapezoid(
            lam*lam*trans*flux)/np.trapezoid(lam*lam*trans)
    
    return models

def get_template_error(wphot, zz=0.5) :
    # adapted from FAST++
    
    # open template error function
    tpl_lam, tpl_err = np.loadtxt('fastpp/TEMPLATE_ERROR.fast.v0.2', unpack=True)
    
    # interpolate to match the restframe photometric wavelengths
    tpl_err = np.interp(wphot/(1 + zz), tpl_lam/1e4, tpl_err)
    
    return tpl_err

def get_times(zz=0.5) :
    
    # set the observation time and the time array
    # tobs = 8622.39 # matches with FAST++'s output, but isn't same as astropy
    tobs = cosmo.age(zz).to(u.Myr).value # Myr
    ts = np.array(np.arange(1, int(tobs) + 1).tolist() + [tobs]) # Myr
    
    return tobs, ts

def integrate(age, iage, isfr) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-ssp.hpp#L23-L47
    
    t2 = 0.0
    integrated = np.zeros_like(age)
    for i in range(len(age)) :
        t1 = t2
        if i < len(age) - 1 :
            t2 = 0.5*(age[i] + age[i + 1])
        else:
            t2 = age[-1]
        
        if t2 <= iage[0] :
            continue
        
        t1 = max(t1, iage[0])
        t2 = min(t2, iage[-1])
        
        integrated[i] = integrate_hinted(iage, isfr, t1, t2)
        
        if t2 >= iage[-1] :
            break
    
    return integrated

def integrate_func(ff, x0, x1, eps=np.finfo(float).eps) :
    # from FAST++/vif
    # https://github.com/cschreib/vif/blob/master/include/vif/math/reduce.hpp#L883-L916
    
    buffer = []
    buffer.append(0.5 * (x1 - x0) * (ff(x0) + ff(x1)))
    
    nn = 0
    oid = 0
    
    while True :
        nn += 1
        
        tr = 0
        tn = 1 << nn
        dd = (x1 - x0) / float(tn)
        for k in range(1, tn // 2 + 1) :
            tr += ff(x0 + (2 * k - 1) * dd)
        buffer.append(0.5 * buffer[oid] + dd * tr)
        
        for mm in range(1, nn + 1) :
            tt = 1 << (2 * mm)
            buffer.append((tt * buffer[-1] - buffer[oid + mm - 1]) / (tt - 1))
        
        oid += nn
        if abs((buffer[-1] - buffer[-2]) / buffer[-1]) <= eps :
            break
    
    return buffer[-1]

def integrate_hinted(xx, yy, x0, x1) :
    # from FAST++/vif
    # https://github.com/cschreib/vif/blob/master/include/vif/math/reduce.hpp#L798-L861
    
    i0 = next((i for i, val in enumerate(xx) if val > x0), None)
    if i0 is None :
        i1 = len(xx) - 1
    else :
        i1 = i0 - 1
        while i1 < len(xx) - 1 and xx[i1 + 1] <= x1 :
            i1 += 1
    
    if i0 > i1 :
        y0 = interpolate(yy[i1], yy[i0], xx[i1], xx[i0], x0)
        y1 = interpolate(yy[i1], yy[i0], xx[i1], xx[i0], x1)
        return 0.5*(y0 + y1)*(x1 - x0)
    else :
        rr = 0.0
        for i in range(i0, i1) :
            rr += 0.5*(yy[i + 1] + yy[i])*(xx[i + 1] - xx[i])
        
        if i0 > 0 :
            y0 = interpolate(yy[i0 - 1], yy[i0], xx[i0 - 1], xx[i0], x0)
            rr += 0.5*(yy[i0] + y0)*(xx[i0] - x0)
        
        if i1 < len(xx) - 1 :
            y1 = interpolate(yy[i1], yy[i1 + 1], xx[i1], xx[i1 + 1], x1)
            rr += 0.5*(y1 + yy[i1])*(x1 - xx[i1])
        
        return rr

def interpolate(y1, y2, x1, x2, xx) :
    # from FAST++/vif
    # https://github.com/cschreib/vif/blob/master/include/vif/math/interpolate.hpp#L11-L14
    return y1 + (y2 - y1)*(xx - x1)/(x2 - x1)

def kc2013(waves) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-gridder.cpp#L508-L509
    
    return noll2009(waves, 1.0, -0.1)

def lookback_time(zz) :
    # from FAST++/vif
    # https://github.com/cschreib/vif/blob/master/include/vif/astro/astro.hpp#L370-L375
    
    def integrand(tt) :
        return (np.power(np.power((1 + tt), 3) * 0.3089 + 0.6911 +
                         np.power(1 + tt, 2) * 0, -0.5) / (1 + tt))
    
    integral, _ = quad(integrand, 0, zz)
    return (3.09 / (67.74 * 3.155e-3)) * integral

def madau1995(zz, waves) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-sfh2sed.cpp#L180-L217
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-gridder.cpp#L460-L498
    
    corr = np.zeros_like(waves) # waves in microns
    
    # compute everything in microns as opposed to angstroms
    lam = np.arange(0.105*(1 + zz), 0.117*(1 + zz) + 1)
    da = np.mean(np.exp(-3.6e-3*np.power(lam/0.1216, 3.46)))
    
    lam = np.arange(0.092*(1 + zz), 0.1015*(1 + zz) + 1)
    db = np.mean(np.exp(-1.7e-3*np.power(lam/0.1026, 3.46)
                        -1.2e-3*np.power(lam/0.09725, 3.46)
                        -9.3e-4*np.power(lam/0.095, 3.46)))
    
    corr[(waves >= 0.0912) & (waves < 0.1026)] = db
    corr[(waves >= 0.1026) & (waves < 0.1216)] = da
    corr[waves >= 0.1216] = 1.0
    
    return corr

def noll2009(waves, eb, delta) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-gridder.cpp#L447-L456
    
    iRv = 1/4.05
    waves2 = np.square(waves) # waves in microns
    
    corr = (calzetti2000(waves) + iRv*eb*waves2*np.square(0.035)/(
        np.square(waves2 - np.square(0.2175)) + waves2*np.square(0.035))
        )*np.power(waves/0.55, delta)
    
    return corr

def resave_bc03_libraries_in_fastpp_format() :
    
    import h5py; from astropy.table import Table
    with h5py.File('tools/bc03_2016.hdf5', 'r') as hf :
        datacube = hf['datacube'][:]
        metallicities = hf['metallicities'][:]
        stellar_ages = hf['stellar_ages'][:]
        wavelengths = hf['wavelengths'][:]
        masses = hf['masses'][:]
    
    # loop every metallicity, saving the resultant table to file
    for i, metallicity in enumerate(metallicities) :
        t = Table([[stellar_ages], [masses[i]], [wavelengths], [datacube[i]]],
                  names=('AGE', 'MASS', 'LAMBDA', 'SED'))
        t.write('bc03_lr_ch_z{}.ised_ASCII.fits'.format(
            str(metallicity).split('.')[1]))
    
    return

def sfh2sed_fastpp(metal, tobs, ts, sfh) :
    # from FAST++
    # https://github.com/cschreib/fastpp/blob/master/src/fast%2B%2B-sfh2sed.cpp#L130-L142
    
    ages, masses, seds = get_bc03_library(metal)
    
    # account for lookback time
    ts = np.flip(tobs - ts)
    sfh = np.flip(sfh)
    
    formed = integrate(ages, ts, sfh)
    tpl_flux = np.zeros(seds.shape[1])
    # mstar = 0.
    for form, mass, sed in zip(formed, masses, seds) :
        tpl_flux += form*sed
        # mstar += form*mass
    
    return tpl_flux

def compare_dust_laws() :
    
    waves = np.linspace(0.1, 0.9, 10001) # [um]
    C00 = calzetti2000(waves)
    C89 = cardelli1989(waves)
    eb, delta = 2.0, 0
    N09 = noll2009(waves, eb, delta)
    KC13 = kc2013(waves)
    
    xs = [waves*1e4, waves*1e4, waves*1e4, waves*1e4]
    ys = [C00, C89, N09, KC13]
    labels = ['Calzetti et al. (200)', 'Cardelli et al. (1989)',
              r'Noll et al. (2009) $E_b =$ {}, $\delta = $ {}'.format(eb, delta),
              r'Kriek \& Conroy (2013) $E_b = 1.0$, $\delta = -0.1$']
    colors = ['k', 'r', 'b', 'm']
    markers = ['', '', '', '']
    styles = ['-', '-', '-', '-']
    alphas = [1, 1, 1, 1]
    
    from plotting import plot_simple_multi
    plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
        xmin=1196.708178394, xmax=9000, ymin=0, ymax=5,
        xlabel=r'$\lambda$ (${\rm \AA}$)',
        ylabel=r'${\rm A}(\lambda)/{\rm A}_{\rm V}$')
    
    return
