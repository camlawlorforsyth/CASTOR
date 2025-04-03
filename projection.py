
import numpy as np

def calculate_MoI_tensor(gas_masses, gas_sfrs, gas_coords, star_ages,
                         star_masses, star_coords, Re, center) :
    # adapted from https://www.tng-project.org/data/forum/topic/223
    
    # check if sufficient gas cells are available within 2Re
    if gas_masses is not None :
        rad_gas = radial_distances(center, gas_coords) # get the radial distance
        
        wGas = np.where((rad_gas <= 2*Re) & (gas_sfrs > 0.0))[0]
        
        masses = gas_masses[wGas]
        xyz = gas_coords[wGas, :]
    else : # if gas cells aren't available, use stars within Re
        rad_stars = radial_distances(center, star_coords)
        
        wStar = np.where((rad_stars <= Re) & (star_ages > 0.0))[0] # stars
            # have ages > 0, whereas wind cells have ages < 0
        
        masses = star_masses[wStar]
        xyz = star_coords[wStar, :]
    
    # shift
    xyz = np.squeeze(xyz)
    
    if xyz.ndim == 1:
        xyz = np.reshape(xyz, (1, 3))
    
    for i in range(3):
        xyz[:, i] -= center[i]
    
    # construct moment of inertia tensor
    I_tensor = np.zeros((3, 3), dtype='float32')
    
    I_tensor[0, 0] = np.sum(masses*(xyz[:, 1]*xyz[:, 1] + xyz[:, 2]*xyz[:, 2]))
    I_tensor[1, 1] = np.sum(masses*(xyz[:, 0]*xyz[:, 0] + xyz[:, 2]*xyz[:, 2]))
    I_tensor[2, 2] = np.sum(masses*(xyz[:, 0]*xyz[:, 0] + xyz[:, 1]*xyz[:, 1]))
    I_tensor[0, 1] = -1*np.sum(masses*(xyz[:, 0]*xyz[:, 1]))
    I_tensor[0, 2] = -1*np.sum(masses*(xyz[:, 0]*xyz[:, 2]))
    I_tensor[1, 2] = -1*np.sum(masses*(xyz[:, 1]*xyz[:, 2]))
    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]
    
    return I_tensor

def radial_distances(center, coordinates) :
    
    dx = coordinates[:, 0] - center[0]
    dy = coordinates[:, 1] - center[1]
    dz = coordinates[:, 2] - center[2]
    
    rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
    
    return rs

def rotation_matrix_from_MoI_tensor(I_tensor) :
    # adapted from https://www.tng-project.org/data/forum/topic/223
    '''
    Calculate 3x3 rotation matrix by a diagonalization of the moment of
    inertia tensor. Note the resultant rotation matrices are hard-coded for
    projection with axes=[0, 1] e.g. along z.
    '''
    
    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(I_tensor)
    
    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]
    
    # permute the eigenvectors into this order, which is the rotation matrix
    # which orients the principal axes to the cartesian x, y, z axes, such that
    # if axes=[0, 1] we have face-on
    new_matrix = np.matrix((rotation_matrix[:, sort_inds[0]],
                            rotation_matrix[:, sort_inds[1]],
                            rotation_matrix[:, sort_inds[2]]))
    
    # make a random edge on view
    # phi = np.random.uniform(0, 2*np.pi)
    # theta = np.pi / 2
    # psi = 0
    
    # A_00 =  np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    # A_01 =  np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    # A_02 =  np.sin(psi)*np.sin(theta)
    # A_10 = -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    # A_11 = -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    # A_12 =  np.cos(psi)*np.sin(theta)
    # A_20 =  np.sin(theta)*np.sin(phi)
    # A_21 = -np.sin(theta)*np.cos(phi)
    # A_22 =  np.cos(theta)
    
    # random_edgeon_matrix = np.matrix(((A_00, A_01, A_02),
    #                                   (A_10, A_11, A_12),
    #                                   (A_20, A_21, A_22)))
    
    # prepare return with a few other useful versions of this rotation matrix
    rot = {}
    rot['face-on'] = new_matrix
    rot['edge-on'] = np.matrix([[1.,  0.,  0.],
                                [0.,  0.,  1.],
                                [0., -1.,  0.]])*rot['face-on'] # disk along x-hat
    rot['edge-on-smallest'] = np.matrix([[0.,  1.,  0.],
                                         [0.,  0.,  1.],
                                         [1.,  0.,  0.]])*rot['face-on']
    rot['edge-on-y'] = np.matrix([[0.,  0.,  1.],
                                  [1.,  0.,  0.],
                                  [0., -1.,  0.]])*rot['face-on'] # disk along y-hat
    # rot['edge-on-random'] = random_edgeon_matrix*rot['face-on']
    # rot['phi'] = phi
    # rot['identity'] = np.matrix(np.identity(3))
    
    return rot
