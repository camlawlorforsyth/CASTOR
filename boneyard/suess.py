
import numpy as np

import astropy.constants as c
from astropy.table import Table
import astropy.units as u

# test the method of Suess+ 2019a
fout = np.loadtxt('Suess+2019a_method/photometry_23November2024_subID_198186.fout',
                  dtype=str, skiprows=18)
avs = fout[:, 4].astype(float)
lages = fout[:, 3].astype(float)
ltaus = fout[:, 9].astype(float)
rs = fout[:, 8].astype(float)

import itertools

tt = Table.read('Suess+2019a_method/chi2.grid.fits')[0]
chi2 = tt['CHI2']
avGrid = tt['AV']
ageGrid = tt['LAGE']
tauGrid = tt['LOG_TAU']
rGrid = tt['R']

massGrid = tt['LMASS']
sfrGrid = tt['LSFR']

# find the minimum chi2 value for the integrated aperture
# print(np.min(chi2[:, 21])/3)

# check that the correct masses can be found in the chi2 grid
# for i in range(22) :
#     loc = np.argmin(chi2[:, i])
#     print(np.round(np.log10(massGrid[loc, i]), 2))

# get the existing positions of the annuli, as in Suess+ (2019a)
positions = np.array([avs[:20], lages[:20], ltaus[:20]]).T # rs[:20]
# positions = np.full((20, 3), -1.0)
# for i in range(20) :
#     loc = np.argmin(chi2[:, i])
#     positions[i] = [avGrid[loc, i], ageGrid[loc, i], tauGrid[loc, i]]
# positions = np.round(positions, 2) # account for floating point error


# get the integrated chi2 according to Suess et al. (2019a)
inDir = 'Suess+2019a_method/best_fits_free/'
intFile = 'photometry_23November2024_subID_198186_198186_bin_int.input_res.fit'
wl, modelFlux = np.loadtxt(inDir + intFile)[:, :2].T

# attach units to the output fluxes
wl *= u.Angstrom
modelFlux = modelFlux*1e-19*u.erg/u.s/u.cm/u.cm/u.Angstrom

# convert the model flux to janskies
modelFlux = (modelFlux*wl*wl/c.c).to(u.Jy).value

catFile = 'Suess+2019a_method/photometry_23November2024_subID_198186.cat'
cat = np.loadtxt(catFile)[21, 1:] # read the integrated values
catFlux = np.array([cat[i] for i in [0, 2, 4, 6, 8, 10, 12]])
catErr = np.array([cat[i + 1] for i in [0, 2, 4, 6, 8, 10, 12]])

intChi = np.sum(np.square((catFlux - modelFlux)/catErr))

def get_chisq(posList) :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # sum of FAST++ chi2 for each annulus
    chis = np.full(20, -1.)
    for i in range(20) :
        pos = posList[i]
        mask = ((np.abs(avGrid[:, i] - pos[0]) <= 0.001) &
                (np.abs(ageGrid[:, i] - pos[1]) <= 0.001) &
                (np.abs(tauGrid[:, i] - pos[2]) <= 0.001)) #&
                # (np.abs(rGrid[:, i] - pos[3]) <= 0.001))
        chis[i] = chi2[:, i][mask][0]
    fastChi = np.sum(chis)
    
    # reduce chi2 and return it, also using the chi2 for all integrated filters
    chi_red = fastChi/(7 - 4) + intChi/(7 - 4)
    
    return chi_red

def adjustPos(positions, ann) :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # make a list of the possible values our Av, age, tau, rr can now take,
    # moving up to +/- 3 steps in any variable
    newPos = np.array([tuple(map(sum, zip(positions[ann], i)))
        for i in itertools.product([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                                   repeat=3)])
    
    # initialize housekeeping variables
    newChi = np.zeros(len(newPos)) + 1e10
    tmpPos = np.array([i for i in positions])
    changed = False
    
    # for each possible new position, calculate the chi2
    for posIdx, pos in enumerate(newPos) :
        
        # make sure this position is actually allowed (e.g., doesn't hit the
        # edge of the grid)
        if ((0 <= pos[0] <= 0.1) and (9 <= pos[1] <= 9.4) and  # 784 valid
            (8.1 <= pos[2] <= 8.5)) : # and (0 <= pos[3] <= 1)) : # possible new
                                                             # positions
            # calculate chi2 for new position
            tmpPos[ann] = pos
            newChi[posIdx] = get_chisq(tmpPos)
    
    # locs = np.where(newChi == np.min(newChi))[0]
    # for loc in locs :
        # print(newPos[loc])
    
    # get lowest chi2 of new possible positions (if it didn't fail)
    if np.min(newChi) < 1e10 :
        # see if it changed
        if list(newPos[np.argmin(newChi)]) != list(positions[ann]) :
            changed = True
        
        # set new positions
        positions[ann] = newPos[np.argmin(newChi)]
    
    return positions, changed

def find_bestPos() :
    # from Suess
    # https://github.com/wrensuess/half-mass-radii/blob/master/photFuncs.py
    
    # set the maximum number of iterations
    maxIter = 500
    
    # initialize the counters for how many times we've updated, and how long
    # it's been stable at a chi2 minimum
    it, lastChanged = 0, 0
    
    # while we haven't coverged, update the positions
    while (it < maxIter and lastChanged < 20*3) :
        # find new positions
        ann = it % 20
        print(it, lastChanged, ann)
        bestPos, changed = adjustPos(positions, ann)
        
        # if unchanged, increment counter
        if not changed :
            lastChanged += 1
        it += 1
    
    print(positions)
    print(bestPos)
    
    return

# find_bestPos()
