import numpy as np
from Residual_helper import interpolator_fvm,xyz_test_sampler
import torch

def gradients_momentum_helper(data_n,nu_s_n,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz):

    terms_n = []
    for i in range(n):
        outputs = interpolator_fvm(data_n[i],main_dim,opt,lb_xyz,ub_xyz,nu_s_n[i],xyz_test,N_xyz)
        terms_n.append(outputs)

    return terms_n

def fvm_uvwp_xyz(lb_xyz,ub_xyz, fvm_data,xyz_test,opt,N_xyz):
    main_dim = 0

    u_fvm = fvm_data['u_fvm']*1000 #converting to mm/s
    v_fvm = fvm_data['v_fvm']*1000 #converting to mm/s
    w_fvm = fvm_data['w_fvm']*1000 #converting to mm/s
    p_fvm = fvm_data['p_fvm']/1e9

    u_cfvm = (u_fvm[0:-1,:,:] + u_fvm[1:,:,:])/2
    v_cfvm = (v_fvm[:,0:-1,:] + v_fvm[:,1:,:])/2
    w_cfvm = (w_fvm[:,:,0:-1] + w_fvm[:,:,1:])/2

    #Computing Derivs 
    n = 4
    data_4 = [u_cfvm,v_cfvm,w_cfvm,p_fvm]
    nu_s_4_x = [[[1,0,0]],[[1,0,0]],[[1,0,0]],[[1,0,0]]]
    nu_s_4_y = [[[0,1,0]],[[0,1,0]],[[0,1,0]],[[0,1,0]]]
    nu_s_4_z = [[[0,0,1]],[[0,0,1]],[[0,0,1]],[[0,0,1]]]

    terms_4_x = gradients_momentum_helper(data_4,nu_s_4_x,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    terms_4_y = gradients_momentum_helper(data_4,nu_s_4_y,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    terms_4_z = gradients_momentum_helper(data_4,nu_s_4_z,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    u_x = terms_4_x[0][0]
    v_x = terms_4_x[1][0]
    w_x = terms_4_x[2][0]
    p_x = terms_4_x[3][0]

    u_y = terms_4_y[0][0]
    v_y = terms_4_y[1][0]
    w_y = terms_4_y[2][0]
    p_y = terms_4_y[3][0]

    u_z = terms_4_z[0][0]
    v_z = terms_4_z[1][0]
    w_z = terms_4_z[2][0]
    p_z = terms_4_z[3][0]

    RHS_1st_derivs = {"u_x": u_x, "v_x":v_x,"w_x":w_x,"p_x":p_x,
                      "u_y": u_y, "v_y":v_y,"w_y":w_y,"p_y":p_y,
                      "u_z": u_z, "v_z":v_z,"w_z":w_z,"p_z":p_z}

    return RHS_1st_derivs


def fvm_uvw_xyz_combos(lb_xyz,ub_xyz, fvm_data,xyz_test,opt,N_xyz):
    main_dim = 0

    u_fvm = fvm_data['u_fvm']*1000 #converting to mm/s
    v_fvm = fvm_data['v_fvm']*1000 #converting to mm/s
    w_fvm = fvm_data['w_fvm']*1000 #converting to mm/s

    u_cfvm = (u_fvm[0:-1,:,:] + u_fvm[1:,:,:])/2
    v_cfvm = (v_fvm[:,0:-1,:] + v_fvm[:,1:,:])/2
    w_cfvm = (w_fvm[:,:,0:-1] + w_fvm[:,:,1:])/2

    u2_fvm = u_cfvm*u_cfvm
    v2_fvm = v_cfvm*v_cfvm
    w2_fvm = w_cfvm*w_cfvm

    uv_fvm = u_cfvm*v_cfvm
    uw_fvm = u_cfvm*w_cfvm
    vw_fvm = v_cfvm*w_cfvm


    #Computing Derivs 
    n = 3
    data_3_x_moment = [u2_fvm,uv_fvm,uw_fvm]
    data_3_y_moment = [uv_fvm,v2_fvm,vw_fvm]
    data_3_z_moment = [uw_fvm,vw_fvm,w2_fvm]

    nu_s_3_x = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]
    nu_s_3_y = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]
    nu_s_3_z = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]

    terms_3_x = gradients_momentum_helper(data_3_x_moment,nu_s_3_x,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    terms_3_y = gradients_momentum_helper(data_3_y_moment,nu_s_3_y,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    terms_3_z = gradients_momentum_helper(data_3_z_moment,nu_s_3_z,n,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    u2_x = terms_3_x[0][0]
    uv_y = terms_3_x[1][0]
    uw_z = terms_3_x[2][0]

    uv_x = terms_3_y[0][0]
    v2_y = terms_3_y[1][0]
    vw_z = terms_3_y[2][0]

    uw_x = terms_3_z[0][0]
    vw_y = terms_3_z[1][0]
    w2_z = terms_3_z[2][0]


    LHS_terms = {"u2_x":u2_x,"uv_y": uv_y, "uw_z": uw_z,
                 "uv_x":uv_x,"v2_y": v2_y, "vw_z": vw_z,
                 "uw_x":uw_x,"vw_y": vw_y, "w2_z": w2_z}

    return LHS_terms

# def x_momentum_res(lb_xyz,ub_xyz, fvm_data, heat_mat_props,xyz_test,opt,N_xyz):

#     main_dim = 0

#     u_fvm = fvm_data['u_fvm']*1000 #converting to mm/s
#     v_fvm = fvm_data['v_fvm']*1000 #converting to mm/s
#     w_fvm = fvm_data['w_fvm']*1000 #converting to mm/s
#     mu_vis_fvm = fvm_data['mu_vis_fvm']/1e9
#     p_fvm = fvm_data['p_fvm']

#     rho = heat_mat_props["rho"]

#     u_cfvm = (u_fvm[0:-1,:,:] + u_fvm[1:,:,:])/2
#     v_cfvm = (v_fvm[:,0:-1,:] + v_fvm[:,1:,:])/2
#     w_cfvm = (w_fvm[:,:,0:-1] + w_fvm[:,:,1:])/2

#     #LHS 
#     data_3 = [u_cfvm*u_cfvm,u_cfvm*v_cfvm,u_cfvm*w_cfvm]
#     nu_s_3 = [[[1,0,0]],[[0,1,0]],[[0,0,1]]]

#     terms_3 = gradients_momentum_helper(data_3,nu_s_3,3,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

#     u2_x_fvm = terms_3[0][0]
#     uv_y_fvm= terms_3[1][0]
#     uw_z_fvm = terms_3[2][0]

#     #RHS Assembly 1
#     data_6 = [p_fvm,u_cfvm,v_cfvm,w_cfvm,u_cfvm,u_cfvm]
#     nu_s_6 = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0]]  #Only x-derivatives for all; y and z only for u   
#     terms_6 = gradients_momentum_helper(data_6,nu_s_6,6,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

#     p_x_fvm = terms_6[0][0]
#     u_x_fvm = terms_6[1][0]
#     v_x_fvm = terms_6[2][0]
#     w_x_fvm = terms_6[3][0]
#     u_y_fvm = terms_6[4][0]
#     u_z_fvm = terms_6[5][0]

#     #RHS 2 Assembly 2
#     data_3 = [2*mu_vis_fvm*u_x_fvm,mu_vis_fvm*v_x_fvm + mu_vis_fvm*u_y_fvm, mu_vis_fvm*w_x_fvm + mu_vis_fvm*u_z_fvm]
#     nu_s_3 = [[1,0,0],[0,1,0],[0,0,1]]    
#     terms_3 = gradients_momentum_helper(data_3,nu_s_3,3,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

#     vis_term_1 = terms_3[0][0] 
#     vis_term_2 = terms_3[1][0] 
#     vis_term_3 = terms_3[2][0] 

#     #Entire Assembly
#     f = rho*1e-6*(u2_x_fvm + uv_y_fvm+uw_z_fvm) + p_x_fvm -vis_term_1 -vis_term_2 - vis_term_3

#     return f


def x_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz):
    main_dim = 0
    print("At X-momentum calculations... ")

    u2_x = LHS_terms["u2_x"]
    uv_y = LHS_terms["uv_y"]
    uw_z = LHS_terms["uw_z"]

    p_x = RHS_1st_derivs['p_x']
    u_x = RHS_1st_derivs['u_x']
    v_x = RHS_1st_derivs['v_x']
    w_x = RHS_1st_derivs['w_x']

    u_y = RHS_1st_derivs['u_y']
    u_z = RHS_1st_derivs['u_z']

    if(opt == 'neutral'):
        data = p_x
        nu_s = [[0,0,0]] #Simply interpolate
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)

        p_x = outputs[0]
    elif(opt == "grid"):
        p_x = p_x
    else:
        print("OPt must be grid or neutral")
        

    RHS_data_3 =[2*mu_vis_fvm*u_x,(mu_vis_fvm*v_x + mu_vis_fvm*u_y),(mu_vis_fvm*w_x + mu_vis_fvm*u_z)]
    nu_s_3 =[[[1,0,0]],[[0,1,0]],[[0,0,1]]]

    main_dim = 0

    terms_3 = gradients_momentum_helper(RHS_data_3,nu_s_3,3,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    vis_term_1 = terms_3[0][0] 
    vis_term_2 = terms_3[1][0] 
    vis_term_3 = terms_3[2][0]

    f = rho*1e-6*(u2_x + uv_y+uw_z) + p_x -vis_term_1 -vis_term_2 - vis_term_3

    return f

def y_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz):
    main_dim = 0
    print("At Y-momentum calculations... ")

    uv_x = LHS_terms["uv_x"]
    v2_y = LHS_terms["v2_y"]
    vw_z = LHS_terms["vw_z"]

    p_y = RHS_1st_derivs['p_y']
    u_y = RHS_1st_derivs['u_y']
    v_y = RHS_1st_derivs['v_y']
    w_y = RHS_1st_derivs['w_y']

    v_x = RHS_1st_derivs['v_y']
    v_z = RHS_1st_derivs['v_z']

    if(opt == 'neutral'):
        data = p_y
        nu_s = [[0,0,0]] #Simply interpolate
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)

        p_y = outputs[0]
    elif(opt == "grid"):
        p_y = p_y
    else:
        print("OPt must be grid or neutral")
        

    RHS_data_3 =[(mu_vis_fvm*u_y+mu_vis_fvm*v_x),(2*mu_vis_fvm*v_y),(mu_vis_fvm*w_y + mu_vis_fvm*v_z)]
    nu_s_3 =[[[1,0,0]],[[0,1,0]],[[0,0,1]]]

   

    terms_3 = gradients_momentum_helper(RHS_data_3,nu_s_3,3,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    vis_term_1 = terms_3[0][0] 
    vis_term_2 = terms_3[1][0] 
    vis_term_3 = terms_3[2][0]

    f = rho*1e-6*(uv_x + v2_y+vw_z) + p_y -vis_term_1 -vis_term_2 - vis_term_3


    return f


def z_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz):
    main_dim = 0
    print("At Z-momentum calculations... ")

    uw_x = LHS_terms["uw_x"]
    vw_y = LHS_terms["vw_y"]
    w2_z = LHS_terms["w2_z"]

    p_z = RHS_1st_derivs['p_z']
    u_z = RHS_1st_derivs['u_z']
    v_z = RHS_1st_derivs['v_z']
    w_z = RHS_1st_derivs['w_z']

    w_x = RHS_1st_derivs['w_x']
    w_y = RHS_1st_derivs['w_y']

    if(opt == 'neutral'):
        data = p_z
        nu_s = [[0,0,0]] #Simply interpolate
        outputs = interpolator_fvm(data,main_dim,opt,lb_xyz,ub_xyz,nu_s,xyz_test,N_xyz)

        p_z = outputs[0]
    elif(opt == "grid"):
        p_z = p_z
    else:
        print("OPt must be grid or neutral")
        

    RHS_data_3 =[(mu_vis_fvm*u_z+mu_vis_fvm*w_x),(mu_vis_fvm*v_z + mu_vis_fvm*w_y),(2*mu_vis_fvm*w_z)]
    nu_s_3 =[[[1,0,0]],[[0,1,0]],[[0,0,1]]]

    

    terms_3 = gradients_momentum_helper(RHS_data_3,nu_s_3,3,main_dim,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    vis_term_1 = terms_3[0][0] 
    vis_term_2 = terms_3[1][0] 
    vis_term_3 = terms_3[2][0]

    f = rho*1e-6*(uw_x + vw_y+w2_z) + p_z -vis_term_1 -vis_term_2 - vis_term_3
    

    return f


def rmse_momentum_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props,xyz_test,opt,N_xyz):


    rho = heat_mat_props["rho"]
    mu_vis_fvm = fvm_data["mu_vis_fvm"]
    
    mu_vis_fvm[mu_vis_fvm>1e8] = 1e8
    mu_vis_fvm = mu_vis_fvm/1e9


    opt_right = 'grid'#Special case just for this function because these terms typically need another derivative (except p)
    N_xyz_right = None
    xyz_test_right = xyz_test_sampler(lb_xyz,ub_xyz,opt_right,N_xyz_right)
    
    
    RHS_1st_derivs = fvm_uvwp_xyz(lb_xyz,ub_xyz, fvm_data,xyz_test_right,opt_right,N_xyz_right)
    LHS_terms = fvm_uvw_xyz_combos(lb_xyz,ub_xyz, fvm_data,xyz_test,opt,N_xyz)


    fx = x_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    fy = y_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)
    fz = z_momentum_res(RHS_1st_derivs,LHS_terms,rho,mu_vis_fvm,opt,lb_xyz,ub_xyz,xyz_test,N_xyz)

    

    return [np.sqrt(np.mean(np.square(fx))),np.sqrt(np.mean(np.square(fy))),np.sqrt(np.mean(np.square(fz)))]

def rmse_momentum_res_PINN(model_PINN,xyz_test):

    xyz_test_tensor = torch.from_numpy(xyz_test).float()

    s = xyz_test_tensor.size(dim = 0)
    s = int(s/20)
    # print(s)
    
    residuals_PINN_x = []
    residuals_PINN_y = []
    residuals_PINN_z = []

    #Looping for memory
    for i in range(20):
        g = xyz_test_tensor[i*s:(i+1)*s,:].clone()
        g.requires_grad = True
        fx,fy,fz,_ = model_PINN.helper_gradients(g,ind = 0)
        residuals_PINN_x.append(fx.cpu().detach().numpy())
        residuals_PINN_y.append(fy.cpu().detach().numpy())
        residuals_PINN_z.append(fz.cpu().detach().numpy())

    residuals_PINN_x = np.array(residuals_PINN_x)
    residuals_PINN_y = np.array(residuals_PINN_y)
    residuals_PINN_z = np.array(residuals_PINN_z)
 
    return [np.sqrt(np.mean(np.square(residuals_PINN_x))),np.sqrt(np.mean(np.square(residuals_PINN_y))),np.sqrt(np.mean(np.square(residuals_PINN_z)))]


def rmse_momentum_res_fvm_pinn(lb_xyz,ub_xyz,fvm_data, heat_mat_props, model_PINN, samples_opt,N_xyz =None):
    
    xyz_test = xyz_test_sampler(lb_xyz,ub_xyz,samples_opt,N_xyz)

    rmse_res_fvm = rmse_momentum_res_FVM(lb_xyz,ub_xyz, fvm_data, heat_mat_props,xyz_test,samples_opt,N_xyz)
    rmse_res_PINN = rmse_momentum_res_PINN(model_PINN,xyz_test)


    return rmse_res_fvm, rmse_res_PINN