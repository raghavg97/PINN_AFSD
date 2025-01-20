import numpy as np
import torch
# from Residual_helper import interpolator_fvm,xyz_test_sampler
from scipy.interpolate import RegularGridInterpolator

def interpolator_T_boundaries(data,lb_xyz,ub_xyz):
    [x_min,y_min,z_min] = lb_xyz
    [x_max,y_max,z_max] = ub_xyz

    x = np.linspace(x_min,x_max,251)
    y = np.linspace(y_min,y_max,101)
    z = np.linspace(z_min,z_max,13)

    x_centers = (x[0:-1] + x[1:]).reshape(-1,)/2
    y_centers = (y[0:-1] + y[1:]).reshape(-1,)/2
    z_centers = (z[0:-1] + z[1:]).reshape(-1,)/2
 

    interp_method = 'cubic'

    interpolator= RegularGridInterpolator([x_centers,y_centers,z_centers],data,method=interp_method
                                          ,bounds_error=False,fill_value=None)
    
    return interpolator
    
def rmse_residuals_T_sides(heat_mat_props,T_a,xyz_sides,interpolator):

    k = heat_mat_props['k']
    h_sides = heat_mat_props['h_sides']
    C_bot = heat_mat_props['C_bot']

    xyz_1 = xyz_sides[0]
    xyz_2 = xyz_sides[1]
    xyz_3 = xyz_sides[2]
    xyz_4 = xyz_sides[3]
    xyz_bot = xyz_sides[4]


    #Side 1    y_min
    T = interpolator(xyz_1,nu = [0,0,0])
    T_y = interpolator(xyz_1,nu = [0,1,0])
    f1 = k*T_y + h_sides*(T_a - T)

    #Side 2    x_min
    T = interpolator(xyz_2,nu = [0,0,0])
    T_x = interpolator(xyz_2,nu = [1,0,0])
    f2 = -k*T_x + h_sides*(T_a - T)

    #Side 3   y_max
    T = interpolator(xyz_3,nu = [0,0,0])
    T_y = interpolator(xyz_3,nu = [0,1,0])
    f3 = -k*T_y + h_sides*(T_a - T)

    #Side 4    x_max
    T = interpolator(xyz_4,nu = [0,0,0])
    T_x = interpolator(xyz_4,nu = [1,0,0])
    f4 = k*T_x + h_sides*(T_a - T)

    #Side 5    z_min
    T = interpolator(xyz_bot,nu = [0,0,0])
    T_z = interpolator(xyz_bot,nu = [0,0,1])
    f5 = k*T_z + C_bot*np.power(T_a - T,3)

    f_conv = np.concatenate((f1,f2,f3,f4,f5))

    rmse_sides = np.sqrt(np.mean(np.square(f_conv)))

    return rmse_sides


def rmse_residuals_T_top(xyz_top,heat_mat_props,process_conditions,r_fr_ph_out, interpolator_T,interpolator_sigma):

    pi = np.pi

    k = heat_mat_props['k']
    eeta = heat_mat_props['eeta']
    
    Omega = process_conditions['Omega']
    A = process_conditions['A']
    R0 = process_conditions['R0']
    Rs = process_conditions['Rs']
    mu = process_conditions['mu']
    slip_factor =process_conditions['slip_factor']


    r = np.sqrt(np.square(xyz_top[:,0]) + np.square(xyz_top[:,1]))

    r_fr = r_fr_ph_out[:,0].reshape(-1,)
    r_ph = r_fr_ph_out[:,1].reshape(-1,)
    r_out = r_fr_ph_out[:,2].reshape(-1,)

    # print(sigma_e.shape)
    # print(r.shape)

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
    # delta = 0.5

    sigma_e = interpolator_sigma(xyz_top,nu=[0,0,0])/1e6

    q_ph = eeta*(0.9*(1-delta)*(sigma_e/np.sqrt(3)) + delta*mu*sigma_e)*(2*pi/60)*Omega*r
    q_fr = 0.5*(0.9*(sigma_e/np.sqrt(3)))*(2*pi/60)*Omega*r

    # print(sigma_e.shape)
    # print(q_ph.shape)
    # print(q_fr.shape)

    T_z = interpolator_T(xyz_top,nu = [0,0,1])
    q = (q_fr*r_fr + q_ph*r_ph + 0.0*r_out)

     
    f = k*T_z - q.reshape(-1,)

    return np.sqrt(np.mean(np.square(f)))

def rmse_residuals_T_FVM(lb_xyz,ub_xyz,fvm_data,xyz_sides,xyz_top,heat_mat_props,process_conditions,r_fr_ph_out):

    T_fvm = fvm_data['T_fvm']
    sigma_e_fvm = fvm_data['sigma_e_fvm']

    T_a = process_conditions['T_a']

    interpolator_T = interpolator_T_boundaries(T_fvm,lb_xyz,ub_xyz)
    interpolator_sigma = interpolator_T_boundaries(sigma_e_fvm,lb_xyz,ub_xyz)
    
    rmse_fvm_top = rmse_residuals_T_top(xyz_top,heat_mat_props,process_conditions,r_fr_ph_out, interpolator_T,interpolator_sigma)
    rmse_fvm_sides = rmse_residuals_T_sides(heat_mat_props,T_a,xyz_sides,interpolator_T)

    return rmse_fvm_top, rmse_fvm_sides
      
def rmse_residuals_T_PINN(model_PINN, xyz_sides, xyz_top, r_fr_ph_out,device2):
    xyz_top = torch.from_numpy(xyz_top).float().to(device2)
    xyz_1 = torch.from_numpy(xyz_sides[0]).float().to(device2)
    xyz_2 = torch.from_numpy(xyz_sides[1]).float().to(device2)
    xyz_3 = torch.from_numpy(xyz_sides[2]).float().to(device2)
    xyz_4 = torch.from_numpy(xyz_sides[3]).float().to(device2)
    xyz_bot = torch.from_numpy(xyz_sides[4]).float().to(device2)

    N_hat = torch.zeros(xyz_top.shape[0],1).to(device2)

    r_fr_ph_out =torch.from_numpy(r_fr_ph_out).float().to(device2)
    
    res_T_top = model_PINN.loss_N_top_T(xyz_top,N_hat,r_fr_ph_out)
    res_T_5sides =model_PINN.loss_NB_T_5sides_v2(xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot)

    # rmse_u = rmse_helper_vel(res_u_top,s_top,res_u_5sides,s_5sides)
    # rmse_v = rmse_helper_vel(res_v_top,s_top,res_v_5sides,s_5sides)
    # rmse_w = rmse_helper_vel(res_w_top,s_top,res_w_5sides,s_5sides)

    return res_T_top.cpu().detach().numpy(), res_T_5sides.cpu().detach().numpy()


def rmse_residuals_T_boundaries(model_PINN, xyz_sides, xyz_top, r_fr_ph_out,device2,
                                lb_xyz,ub_xyz,fvm_data,heat_mat_props,process_conditions):
    
    PINN_top_sides = rmse_residuals_T_PINN(model_PINN, xyz_sides, xyz_top, r_fr_ph_out,device2)
    FVM_top_sides = rmse_residuals_T_FVM(lb_xyz,ub_xyz,fvm_data,xyz_sides,xyz_top,heat_mat_props,process_conditions,r_fr_ph_out)

    return FVM_top_sides,PINN_top_sides