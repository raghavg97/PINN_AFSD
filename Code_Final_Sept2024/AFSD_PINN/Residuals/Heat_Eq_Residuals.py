import numpy as np
from Residual_helper import interpolator_fvm, fvm_centers, quick_test_sampler, meshgrid_multipurpose
import torch


def rmse_heat_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props):
    main_dim = 0
    opt = 'neutral'

    T_fvm = fvm_data['T_fvm']
    u_fvm = fvm_data['u_fvm']
    v_fvm = fvm_data['v_fvm']
    w_fvm = fvm_data['w_fvm']
    sigma_e_fvm = fvm_data['sigma_e_fvm']
    eps_e_fvm = fvm_data['eps_e_fvm']

    k = heat_mat_props['k']
    rho = heat_mat_props['rho']
    c_p = heat_mat_props['c_p']


    #LHS
    data = T_fvm
    nu_s = [[2,0,0],[0,2,0],[0,0,2]]
    outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)

    T_xx_fvm = outputs[0]
    T_yy_fvm = outputs[1]
    T_zz_fvm = outputs[2]

    #Q_G 
    data_2 = [sigma_e_fvm,eps_e_fvm]
    terms2 = []
    for data in data_2:
        nu_s = [None]
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)
        terms2.append(outputs)

    sigma_e_interp = terms2[0][0]
    eps_e_interp = terms2[1][0]

    q_g_fvm = 0.9*sigma_e_interp*eps_e_interp

    #RHS

    u_cfvm = interpolator_fvm(u_fvm,main_dim = 1,opt = 'grid',lb_xyz=lb_xyz,ub_xyz=ub_xyz,nu_s=[[0,0,0]])[0]
    v_cfvm = interpolator_fvm(v_fvm,main_dim = 2,opt = 'grid',lb_xyz=lb_xyz,ub_xyz=ub_xyz,nu_s=[[0,0,0]])[0]
    w_cfvm = interpolator_fvm(w_fvm,main_dim = 3,opt = 'grid',lb_xyz=lb_xyz,ub_xyz=ub_xyz,nu_s=[[0,0,0]])[0]

    data_3 = [u_cfvm*T_fvm,v_cfvm*T_fvm,w_cfvm*T_fvm]

    nu_s_3 = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]
    terms_3 = []


    for i in range(3):
        data = data_3[i]   
        nu_s =nu_s_3[i]
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)
        terms_3.append(outputs)

    uT_x_fvm = terms_3[0][0]
    vT_y_fvm= terms_3[1][0]
    wT_z_fvm = terms_3[2][0]

    residuals_fvm = k*(T_xx_fvm + T_yy_fvm + T_zz_fvm) - q_g_fvm - rho*c_p*(uT_x_fvm + vT_y_fvm + wT_z_fvm)

    return np.sqrt(np.mean(np.square(residuals_fvm)))

def rmse_heat_res_PINN(model_PINN,lb_xyz,ub_xyz):
    xyz_centers = fvm_centers(lb_xyz,ub_xyz)
    x_test,y_test, z_test = quick_test_sampler(xyz_centers)
    xyz_test = meshgrid_multipurpose(x_test,y_test,z_test)

    xyz_test_tensor = torch.from_numpy(xyz_test).float()

    g = xyz_test_tensor.clone()
    g.requires_grad = True
    residuals_PINN = model_PINN.helper_gradients(g,ind = 2)
 
    return np.sqrt(np.mean(np.square(residuals_PINN.cpu().detach().numpy())))


def rmse_heat_res_fvm_pinn(lb_xyz,ub_xyz,fvm_data, heat_mat_props, model_PINN):
    
    rmse_res_fvm = rmse_heat_res_FVM(lb_xyz,ub_xyz,fvm_data,heat_mat_props)
    rmse_res_PINN = rmse_heat_res_PINN(model_PINN,lb_xyz,ub_xyz)


    return rmse_res_fvm, rmse_res_PINN