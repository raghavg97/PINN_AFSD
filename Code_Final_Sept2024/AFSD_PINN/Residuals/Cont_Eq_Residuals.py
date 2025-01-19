import numpy as np
from Residual_helper import interpolator_fvm, fvm_centers, quick_test_sampler, meshgrid_multipurpose
import torch


def rmse_cont_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props):
    
    opt = 'grid'

    u_fvm = fvm_data['u_fvm']
    v_fvm = fvm_data['v_fvm']
    w_fvm = fvm_data['w_fvm']

    #LHS - Term 1
    main_dim = 1
    data = u_fvm*1000 #convert to mm/s
    nu_s = [[1,0,0]]
    outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)
    du_dx = outputs[0]

    #LHS - Term 2
    main_dim = 2
    data = v_fvm*1000
    nu_s = [[0,1,0]]
    outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)
    dv_dy = outputs[0]

    #LHS - Term 3
    main_dim = 3
    data = w_fvm*1000
    nu_s = [[0,0,1]]
    outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s)
    dw_dz = outputs[0]

    #RHS - 0

    residuals_fvm = du_dx + dv_dy + dw_dz

    # print(residuals_fvm.shape)

    return np.sqrt(np.mean(np.square(residuals_fvm)))

def rmse_cont_res_PINN(model_PINN,lb_xyz,ub_xyz):
    xyz_centers = fvm_centers(lb_xyz,ub_xyz)
    x_test,y_test, z_test = quick_test_sampler(xyz_centers)
    xyz_test = meshgrid_multipurpose(x_test,y_test,z_test)

    xyz_test_tensor = torch.from_numpy(xyz_test).float()

    s = xyz_test_tensor.size(dim = 0)
    s = int(s/10)
    # print(s)
    
    residuals_PINN = []
    #Looping for memory
    for i in range(10):
        g = xyz_test_tensor[i*s:(i+1)*s,:].clone()
        g.requires_grad = True
        _,_,_,f4 = model_PINN.helper_gradients(g,ind = 0)
        residuals_PINN.append(f4.cpu().detach().numpy())

    residuals_PINN = np.array(residuals_PINN)
    return np.sqrt(np.mean(np.square(residuals_PINN)))


def rmse_cont_res_fvm_pinn(lb_xyz,ub_xyz,fvm_data, heat_mat_props, model_PINN):
    
    rmse_res_fvm = rmse_cont_res_FVM(lb_xyz,ub_xyz,fvm_data,heat_mat_props)
    rmse_res_PINN = rmse_cont_res_PINN(model_PINN,lb_xyz,ub_xyz)


    return rmse_res_fvm, rmse_res_PINN