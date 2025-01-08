import numpy as np
from smt.sampling_methods import LHS

from IPython import get_ipython
import copy
ipython = get_ipython()

global R0, A, Rs, pi, Omega, V, F, delta
delta = ipython.user_ns['delta']
R0 = ipython.user_ns['R0']
A = ipython.user_ns['A']
Rs = ipython.user_ns['Rs']
pi = ipython.user_ns['pi']
Omega = ipython.user_ns['Omega']
V = ipython.user_ns['V']
F = ipython.user_ns['F']



def trainingdata_uvw(N_B,N_f,lb_xyz,ub_xyz,seed):
    [x_min,y_min,z_min] = lb_xyz
    [x_max,y_max,z_max] = ub_xyz

    
    ub_xyz = np.array([x_max,y_max,z_max])
    #Boundary Top
    x_top = np.random.uniform(x_min,x_max,(2*N_B,1))
    y_top = np.random.uniform(y_min,y_max,(2*N_B,1))
    z_top = z_max*np.ones((2*N_B,1))
    xyz_top = np.hstack((x_top,y_top,z_top))
    
    #Additional Samples at top "Centre"
    x_c_top = np.random.normal(0,5,(N_B,1))
    y_c_top = np.random.normal(0,5,(N_B,1))    
    z_c_top = z_max*np.ones((N_B,1))
    xyz_c_top = np.hstack((x_c_top,y_c_top,z_c_top))
    
    xyz_top = np.vstack((xyz_top,xyz_c_top))
    

    #Boundary Bottom
    x_bot = np.random.uniform(x_min,x_max,(N_B,1))
    y_bot = np.random.uniform(y_min,y_max,(N_B,1))
    z_bot = z_min*np.ones((N_B,1))
    xyz_bot = np.hstack((x_bot,y_bot,z_bot))
    
    #Side1 
    x_1 = np.random.uniform(x_min,x_max,(N_B,1))
    y_1 = y_min*np.ones((N_B,1))
    z_1 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_1 = np.hstack((x_1,y_1,z_1))
    
    #Side2
    x_2 = x_max*np.ones((N_B,1))
    y_2 = np.random.uniform(z_min,z_max,(N_B,1))
    z_2 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_2 = np.hstack((x_2,y_2,z_2))
    
    #Side3
    x_3 = np.random.uniform(x_min,x_max,(N_B,1))
    y_3 = y_max*np.ones((N_B,1))
    z_3 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_3 = np.hstack((x_3,y_3,z_3))
    
     #Side4
    x_4 = x_min*np.ones((N_B,1))
    y_4 = np.random.uniform(z_min,z_max,(N_B,1))
    z_4 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_4 = np.hstack((x_4,y_4,z_4))


    x01 = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]])
    sampling = LHS(xlimits=x01,random_state =seed)
    samples = sampling(N_f)

    xyz_coll = lb_xyz + (ub_xyz - lb_xyz)*samples
    xyz_coll = np.vstack((xyz_coll,xyz_1,xyz_2,xyz_3,xyz_4,xyz_top,xyz_bot)) # append training points to collocation points


    #True Velocity Calculations
    r = np.sqrt(np.square(xyz_top[:,0]) + np.square(xyz_top[:,1]))
        
    r_fr = (r<R0).reshape(-1,1)
    r_ph = np.logical_and(r>=R0,r<=Rs).reshape(-1,1)
    r_out = np.logical_not(r<=Rs).reshape(-1,1)
    
    cos_theta = xyz_top[:,1]/r # From Y direction 6/4
    sin_theta = xyz_top[:,0]/r #
    
    r = r.reshape(-1,1)
    cos_theta = cos_theta.reshape(-1,1)
    sin_theta = sin_theta.reshape(-1,1)
    
    B = 1.38
    # delta = 1-np.exp(-A*(r-R0)/Rs)
    delta = B*(r-R0)/Rs

    
    #BC1 #Exponential 
    u_true_ph = (1-delta)*(2*pi/60)*Omega*r*cos_theta
    v_true_ph = (1-delta)*(2*pi/60)*Omega*r*sin_theta - V
    w_true_ph = 0.0

    
    #BC2
    u_true_fr = (2*pi/60)*Omega*r*cos_theta
    v_true_fr = (2*pi/60)*Omega*r*sin_theta - V
    w_true_fr = -F
    
    #OTHER
    u_true_out = 0.0
    v_true_out = -V
    w_true_out = 0.0

    u_true = u_true_fr*r_fr + u_true_ph*r_ph + u_true_out*r_out
    v_true = v_true_fr*r_fr + v_true_ph*r_ph + v_true_out*r_out
    w_true = w_true_fr*r_fr + w_true_ph*r_ph + w_true_out*r_out

    uvw_true_top = np.hstack((u_true,v_true,w_true))

    r_fr_ph_out = np.hstack((r_fr,r_ph,r_out))

    return xyz_coll, xyz_1, xyz_2, xyz_3, xyz_4, xyz_top, xyz_bot, uvw_true_top, r_fr_ph_out


def trainingdata_T(N_B,N_f,lb_xyz,ub_xyz,seed):
    [x_min,y_min,z_min] = lb_xyz
    [x_max,y_max,z_max] = ub_xyz
    #Boundary Top
    x_top = np.random.uniform(x_min,x_max,(N_B,1))
    y_top = np.random.uniform(y_min,y_max,(N_B,1))
    z_top = z_max*np.ones((N_B,1))
    xyz_top = np.hstack((x_top,y_top,z_top))
    
    #Additional Samples at top "Centre"
    x_c_top = np.random.normal(0,5,(N_B,1))
    y_c_top = np.random.normal(0,5,(N_B,1))
    z_c_top = z_max*np.ones((N_B,1))
    xyz_c_top = np.hstack((x_c_top,y_c_top,z_c_top))
    
    xyz_top = np.vstack((xyz_top,xyz_c_top))
    

    #Boundary Bottom
    x_bot = np.random.uniform(x_min,x_max,(N_B,1))
    y_bot = np.random.uniform(y_min,y_max,(N_B,1))
    z_bot = z_min*np.ones((N_B,1))
    xyz_bot = np.hstack((x_bot,y_bot,z_bot))
    
    #Side1
    x_1 = np.random.uniform(x_min,x_max,(N_B,1))
    y_1 = y_min*np.ones((N_B,1))
    z_1 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_1 = np.hstack((x_1,y_1,z_1))
    
    #Side2
    x_2 = x_max*np.ones((N_B,1))
    y_2 = np.random.uniform(z_min,z_max,(N_B,1))
    z_2 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_2 = np.hstack((x_2,y_2,z_2))
    
    #Side3
    x_3 = np.random.uniform(x_min,x_max,(N_B,1))
    y_3 = y_max*np.ones((N_B,1))
    z_3 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_3 = np.hstack((x_3,y_3,z_3))
    
     #Side4
    x_4 = x_min*np.ones((N_B,1))
    y_4 = np.random.uniform(z_min,z_max,(N_B,1))
    z_4 = np.random.uniform(z_min,z_max,(N_B,1))
    xyz_4 = np.hstack((x_4,y_4,z_4))


    x01 = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]])
    sampling = LHS(xlimits=x01,random_state =seed)
    samples = sampling(N_f)

    xyz_coll = lb_xyz + (ub_xyz - lb_xyz)*samples
    xyz_coll = np.vstack((xyz_coll,xyz_1,xyz_2,xyz_3,xyz_4,xyz_top,xyz_bot)) # append training points to collocation points

    return xyz_coll, xyz_1, xyz_2, xyz_3, xyz_4, xyz_top, xyz_bot