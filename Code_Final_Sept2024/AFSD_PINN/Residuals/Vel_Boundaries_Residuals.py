import numpy as np
import torch
# from Residual_helper import interpolator_fvm,xyz_test_sampler
from scipy.interpolate import RegularGridInterpolator

def rmse_helper_vel(mse1,s1,mse2,s2):

    mse1 = mse1.cpu().detach().numpy()
    mse2 = mse2.cpu().detach().numpy()

    mse_overall = mse1*s1 + mse2*s2

    return np.sqrt(mse_overall/(s1+s2))

def top_velocity(xy,process_conditions):
    r = np.sqrt(np.square(xy[:,0]) + np.square(xy[:,1]))

    R0 = process_conditions['R0']
    Rs = process_conditions['Rs']
    pi = np.pi
    Omega = process_conditions['Omega']
    A = process_conditions['A']
    V = process_conditions['V']
    F = process_conditions['F']
    slip_factor = process_conditions['slip_factor']


    r_fr = (r<R0).reshape(-1,1)
    r_ph = np.logical_and(r>=R0,r<=Rs).reshape(-1,1)
    r_out = np.logical_not(r<=Rs).reshape(-1,1)
    
    cos_theta = xy[:,1]/r # From Y direction 6/4
    sin_theta = xy[:,0]/r #
    
    r = r.reshape(-1,1)
    cos_theta = cos_theta.reshape(-1,1)
    sin_theta = sin_theta.reshape(-1,1)
    

    if(slip_factor =="Exp"):
        delta = 1-np.exp(-A*(r-R0)/Rs)
    elif(slip_factor == "Linear"):
        B = 1.38
        # delta = 1-np.exp(-A*(r-R0)/Rs)
        delta = B*(r-R0)/Rs
    elif(slip_factor == 'PureSlip'):
        delta = 1
    else:
        print("slip factor must be correct...")
        return

    
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

    u_true = (u_true_fr*r_fr + u_true_ph*r_ph + u_true_out*r_out)
    v_true = (v_true_fr*r_fr + v_true_ph*r_ph + v_true_out*r_out)
    w_true = (w_true_fr*r_fr + w_true_ph*r_ph + w_true_out*r_out)

    return u_true, v_true, w_true


def rmse_residuals_vel_top_FVM(xyz_test_top,uvw_true_top, lb_xyz,ub_xyz,process_conditions):
    [x_min,y_min,_] = lb_xyz
    [x_max,y_max,_] = ub_xyz

    x = np.linspace(x_min,x_max,251)
    y = np.linspace(y_min,y_max,101)

    x= (x[0:-1] + x[1:]).reshape(-1,)/2
    y = (y[0:-1] + y[1:]).reshape(-1,)/2

    X,Y = np.meshgrid(x,y)

    X = X.flatten('F').reshape(-1,1)
    Y = Y.flatten('F').reshape(-1,1)

    xy_grid = np.hstack((X,Y))

    u_true_grid, v_true_grid, w_true_grid = top_velocity(xy_grid,process_conditions)

    # print(np.max(u_true_grid))
    # print(np.max(v_true_grid))
    # print(np.max(w_true_grid))

    u_true_grid = u_true_grid.reshape(250,100,order = 'F')
    v_true_grid = v_true_grid.reshape(250,100,order = 'F')
    w_true_grid = w_true_grid.reshape(250,100,order = 'F')


    interp_method = 'cubic'
    interpolator_u = RegularGridInterpolator([x,y],u_true_grid,method=interp_method,bounds_error=False,fill_value=None)
    interpolator_v = RegularGridInterpolator([x,y],v_true_grid,method=interp_method,bounds_error=False,fill_value=None)
    interpolator_w = RegularGridInterpolator([x,y],w_true_grid,method=interp_method,bounds_error=False,fill_value=None)


    u_interp = interpolator_u(xyz_test_top[:,:-1])
    v_interp = interpolator_v(xyz_test_top[:,:-1])
    w_interp = interpolator_w(xyz_test_top[:,:-1])

    rmse_u = np.sqrt(np.mean(np.square(u_interp - uvw_true_top[:,0])))
    rmse_v = np.sqrt(np.mean(np.square(v_interp - uvw_true_top[:,1])))
    rmse_w = np.sqrt(np.mean(np.square(w_interp - uvw_true_top[:,2])))

    return  [rmse_u,rmse_v,rmse_w]


def detach_np(s):
    return s.cpu().detach().numpy()

def rmse_residuals_vel_PINN(model_PINN,xyz_test_top,xyz_test_5sides, uvw_true_top,device1):
    xyz_test_top_tensor = torch.from_numpy(xyz_test_top).float().to(device1)
    xyz_test_5sides_tensor = torch.from_numpy(xyz_test_5sides).float().to(device1)
    uvw_true_top_tensor = torch.from_numpy(uvw_true_top).float().to(device1)

    res_u_top,res_v_top,res_w_top,s_top = model_PINN.loss_D_top_uvw_residuals(xyz_test_top_tensor,uvw_true_top_tensor)
    res_u_5sides,res_v_5sides,res_w_5sides,s_5sides =model_PINN.loss_B_uvw_5sides_residuals(xyz_test_5sides_tensor)

    # rmse_u = rmse_helper_vel(res_u_top,s_top,res_u_5sides,s_5sides)
    # rmse_v = rmse_helper_vel(res_v_top,s_top,res_v_5sides,s_5sides)
    # rmse_w = rmse_helper_vel(res_w_top,s_top,res_w_5sides,s_5sides)

    return [detach_np(res_u_top),detach_np(res_v_top),detach_np(res_w_top)],[detach_np(res_u_5sides),detach_np(res_v_5sides),detach_np(res_w_5sides)]


def rmse_vel_boundaries(model_PINN,xyz_test_top,xyz_test_5sides, uvw_true_top,device1,lb_xyz,ub_xyz,process_conditions):

    PINN_top, PINN_5_sides = rmse_residuals_vel_PINN(model_PINN,xyz_test_top,xyz_test_5sides, uvw_true_top,device1)
    fvm_top = rmse_residuals_vel_top_FVM(xyz_test_top,uvw_true_top, lb_xyz,ub_xyz,process_conditions)

    return fvm_top,PINN_top,PINN_5_sides
