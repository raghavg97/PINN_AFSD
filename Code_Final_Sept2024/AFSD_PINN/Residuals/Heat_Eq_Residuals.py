import numpy as np
from Residual_helper import interpolator_fvm,xyz_test_sampler
import torch


def rmse_heat_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props,xyz_test,opt,N_xyz):
    main_dim = 0

    T_fvm = fvm_data['T_fvm']
    u_fvm = fvm_data['u_fvm']*1000 #converting to mm/s
    v_fvm = fvm_data['v_fvm']*1000 #converting to mm/s
    w_fvm = fvm_data['w_fvm']*1000 #converting to mm/s
    sigma_e_fvm = fvm_data['sigma_e_fvm']/1e6 #Correcting units
    eps_e_fvm = fvm_data['eps_e_fvm']/1e3 #Correcting units

    k = heat_mat_props['k']
    rho = heat_mat_props['rho']
    c_p = heat_mat_props['c_p']


    #LHS
    data = T_fvm
    nu_s = [[2,0,0],[0,2,0],[0,0,2]]    
    outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)

    T_xx_fvm = outputs[0]
    T_yy_fvm = outputs[1]
    T_zz_fvm = outputs[2]

    #Q_G 
    data_2 = [sigma_e_fvm,eps_e_fvm]
    terms2 = []
    for data in data_2:
        nu_s = [None]
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)
        terms2.append(outputs)

    sigma_e_interp = terms2[0][0]
    eps_e_interp = terms2[1][0]

    q_g_fvm = 0.9*sigma_e_interp*eps_e_interp

    #RHS
    u_cfvm = (u_fvm[0:-1,:,:] + u_fvm[1:,:,:])/2
    v_cfvm = (v_fvm[:,0:-1,:] + v_fvm[:,1:,:])/2
    w_cfvm = (w_fvm[:,:,0:-1] + w_fvm[:,:,1:])/2

    data_3 = [u_cfvm*T_fvm,v_cfvm*T_fvm,w_cfvm*T_fvm]

    nu_s_3 = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]
    terms_3 = []


    for i in range(3):
        data = data_3[i]   
        nu_s =nu_s_3[i]
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)
        terms_3.append(outputs)

    uT_x_fvm = terms_3[0][0]
    vT_y_fvm= terms_3[1][0]
    wT_z_fvm = terms_3[2][0]

    residuals_fvm = k*(T_xx_fvm + T_yy_fvm + T_zz_fvm) - q_g_fvm - rho*c_p*(uT_x_fvm + vT_y_fvm + wT_z_fvm)*1e-6

    return np.sqrt(np.mean(np.square(residuals_fvm)))

def rmse_heat_res_PINN(model_PINN,xyz_test):

    xyz_test_tensor = torch.from_numpy(xyz_test).float()

    s = xyz_test_tensor.size(dim = 0)
    s = int(s/10)
    # print(s)
    
    residuals_PINN = []
    #Looping for memory
    for i in range(10):
        g = xyz_test_tensor[i*s:(i+1)*s,:].clone()
        g.requires_grad = True
        f5 = model_PINN.helper_gradients(g,ind = 2)
        residuals_PINN.append(f5.cpu().detach().numpy())

    residuals_PINN = np.array(residuals_PINN)
 
    return np.sqrt(np.mean(np.square(residuals_PINN)))


def rmse_heat_res_fvm_pinn(lb_xyz,ub_xyz,fvm_data, heat_mat_props, model_PINN, samples_opt,N_xyz =None):
    
    xyz_test = xyz_test_sampler(lb_xyz,ub_xyz,samples_opt,N_xyz)

    rmse_res_fvm = rmse_heat_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props,xyz_test,samples_opt,N_xyz)
    rmse_res_PINN = rmse_heat_res_PINN(model_PINN,xyz_test)


    return rmse_res_fvm, rmse_res_PINN