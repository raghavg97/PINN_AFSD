import numpy as np
from scipy.interpolate import RegularGridInterpolator


def fvm_centers(lb_xyz,ub_xyz):
    [x_min,y_min,z_min] = lb_xyz
    [x_max,y_max,z_max] = ub_xyz

    x = np.linspace(x_min,x_max,251)
    y = np.linspace(y_min,y_max,101)
    z = np.linspace(z_min,z_max,13)

    x_centers = (x[0:-1] + x[1:]).reshape(-1,)/2
    y_centers = (y[0:-1] + y[1:]).reshape(-1,)/2
    z_centers = (z[0:-1] + z[1:]).reshape(-1,)/2

    xyz_centers = {"x":x,"y":y,"z":z,"x_centers":x_centers,"y_centers":y_centers,"z_centers":z_centers}

    return xyz_centers


def quick_test_sampler(xyz_centers):


    x_centers = xyz_centers['x_centers']
    y_centers = xyz_centers['y_centers']
    z_centers = xyz_centers['z_centers']

    x_cmin = np.min(x_centers)
    x_cmax = np.max(x_centers)
    y_cmin = np.min(y_centers)
    y_cmax = np.max(y_centers)
    z_cmin = np.min(z_centers)
    z_cmax = np.max(z_centers)

    np.random.seed(1234)

    x_test = np.sort(np.random.uniform(x_cmin,x_cmax,(100,)))
    y_test = np.sort(np.random.uniform(y_cmin,y_cmax,(100,)))
    z_test = np.sort(np.random.uniform(z_cmin,z_cmax,(10,)))

    return x_test,y_test,z_test


def meshgrid_multipurpose(x,y,z):

    X,Y,Z = np.meshgrid(x,y,z)

    X = X.flatten('F').reshape(-1,1)
    Y = Y.flatten('F').reshape(-1,1)
    Z = Z.flatten('F').reshape(-1,1)

    xyz = np.hstack((X,Y,Z))

    return xyz


def interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s): #nu_s list of tuples

    interp_method = 'slinear'

    xyz_centers = fvm_centers(lb_xyz,ub_xyz)

    x = xyz_centers['x']
    y = xyz_centers['y']
    z = xyz_centers['z']
    x_centers = xyz_centers['x_centers']
    y_centers = xyz_centers['y_centers']
    z_centers = xyz_centers['z_centers']
    
    
    if(main_dim==1):
        interpolator= RegularGridInterpolator([x,y_centers,z_centers],data,method=interp_method)
    elif(main_dim ==2):
        interpolator= RegularGridInterpolator([x_centers,y,z_centers],data,method=interp_method) 
    elif(main_dim ==3):
        interpolator= RegularGridInterpolator([x_centers,y_centers,z],data,method=interp_method)
    elif(main_dim == 0):
        interpolator= RegularGridInterpolator([x_centers,y_centers,z_centers],data,method=interp_method)
    else: 
        print("Main dim must be 1,2,3, or 0. Currently it is ",main_dim)
        return
    
    print("Interpolation Done...")

    if(opt == 'grid'):
        x_test = x_centers
        y_test = y_centers
        z_test = z_centers
    elif(opt == 'neutral'):
        x_test,y_test,z_test= quick_test_sampler(xyz_centers)
    else:
        print("opt must be grid or neutral")
        return
    

    xyz_test = meshgrid_multipurpose(x_test,y_test,z_test)

    
    outputs = []
    for i in range(len(nu_s)):
        interp_values = interpolator(xyz_test,nu = nu_s[i])
        if(opt=='grid'):
            outputs.append(interp_values.reshape(250,100,12,order = 'F'))
        elif(opt == 'neutral'):
            outputs.append(interp_values.reshape(100,100,10,order = 'F'))


    return outputs




