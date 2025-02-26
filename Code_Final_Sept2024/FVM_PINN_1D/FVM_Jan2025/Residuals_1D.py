import numpy as np
from scipy.interpolate import CubicSpline 
import torch

def residuals_fvm_cubicspline(x_fvm,T_fvm,C_fvm,x_test,pde_related_funcs,problem_constants):
    
    alpha = problem_constants['alpha']
    beta = problem_constants['beta']

    Q = pde_related_funcs["heat_source_func"]
    S = pde_related_funcs["conc_force_func"]

    interp_T = CubicSpline(x_fvm, T_fvm, bc_type='not-a-knot', extrapolate=True)
    interp_C = CubicSpline(x_fvm, C_fvm, bc_type='not-a-knot', extrapolate=True)

    T_interp = interp_T(x_test)
    d2T_dx2_interp = interp_T(x_test,nu=2)

    C_interp = interp_C(x_test)
    d2C_dx2_interp = interp_C(x_test,nu=2) 

    residual_eq1 = alpha*d2T_dx2_interp - Q(C_interp)
    residual_eq2 = beta*d2C_dx2_interp - S(T_interp)

    return residual_eq1,residual_eq2


def pde_res_compare(opt,interp_method,
                pde_related_funcs,problem_constants, N_x_test,
                N_x,T_fvm,C_fvm,T_right_fvm,T_left,C_right,C_left,coPINN,device):
    
    L = problem_constants['L']

    del_x = L/N_x

    x = np.linspace(del_x/2,L-del_x/2,N_x).reshape(-1,)
    x = np.insert(x,0,0)
    x = np.append(x,L)

    T_fvm = np.insert(T_fvm,0,T_left)
    T_fvm = np.append(T_fvm,T_right_fvm)

    C_fvm = np.insert(C_fvm,0,C_left)
    C_fvm = np.append(C_fvm,C_right)

    if(opt == 'grid'):
        x_test = np.linspace(del_x/2,L-del_x/2,N_x_test)
        # x_test = np.linspace(0,L,N_x_test+1)
    elif(opt == 'neutral'):
        x_test = np.sort(np.random.uniform(0,L,(N_x_test,)))
    else:
        print("Give grid or neutral option")
        return 

    # x_pinn,t_pinn = np.meshgrid(x_test,t_test)

    x_pinn = x_test.reshape(-1,1)
    
    
    x_test_tensor = torch.from_numpy(x_pinn).float().to(device)

    f1_PINN,f2_PINN = coPINN.residuals_f1f2(x_test_tensor)
    f1_PINN = f1_PINN.reshape(N_x_test).cpu().detach().numpy()
    f2_PINN = f2_PINN.reshape(N_x_test).cpu().detach().numpy()


    if(interp_method == 'cubic spline'):
        f1_fvm,f2_fvm = residuals_fvm_cubicspline(x,T_fvm,C_fvm,x_test,pde_related_funcs,problem_constants)
    elif(interp_method == 'linear'):
            print("WRONG....")
    else:
         raise Exception("Interpolation must be spline or NN")


    RMS_Residuals_f1_fvm = np.sqrt(np.mean(np.square(f1_fvm)))
    RMS_Residuals_f1_PINN = np.sqrt(np.mean(np.square(f1_PINN)))


    RMS_Residuals_f2_fvm = np.sqrt(np.mean(np.square(f2_fvm)))
    RMS_Residuals_f2_PINN = np.sqrt(np.mean(np.square(f2_PINN)))

    
    print("PINN_f1: %.4f"%RMS_Residuals_f1_PINN)
    print("FVM_f1: %.4f"%RMS_Residuals_f1_fvm)
    print("PINN_f2: %.4f"%RMS_Residuals_f2_PINN)
    print("FVM_f2: %.4f"%RMS_Residuals_f2_fvm)

    return f1_PINN,f1_fvm,f2_PINN,f2_fvm, x_test


#------------------------------------------------------------------------
#------------------------------------------------------------------------

def residual_fvm_robin_cubicspline(x_fvm,T_fvm,x_right,T_right, problem_constants):
    
    # alpha = problem_constants['alpha']
    # beta = problem_constants['beta']

    h_c = problem_constants['h_c']
    T_a = problem_constants['T_a']

    interp_T = CubicSpline(x_fvm, T_fvm, bc_type='not-a-knot', extrapolate=True)

    # T_interp = interp_T(x_right)
    dT_dx_interp = interp_T(x_right,nu=1)

    robin_residual = dT_dx_interp - h_c*(T_right - T_a)
    return robin_residual


def robin_residual_compare(problem_constants, N_x,T_fvm,T_right_fvm,T_left,coPINN,device):
    L = problem_constants['L']


    x = np.linspace(0 + L/N_x,L - L/N_x,N_x).reshape(-1,)
    x = np.insert(x,0,0)
    x = np.append(x,L)

    T_fvm = np.insert(T_fvm,0,T_left)
    T_fvm = np.append(T_fvm,T_right_fvm)

    r_fvm = residual_fvm_robin_cubicspline(x.reshape(-1,),T_fvm,L,T_right_fvm,problem_constants)  


    x_test_tensor = torch.tensor(L).float().to(device)
    #f1_PINN,f2_PINN = coPINN.residuals_f1f2(x_test_tensor)
    r_PINN = coPINN.residuals_robin_BC(x_test_tensor.reshape(-1,1)).cpu().detach().numpy()

    return np.abs(r_fvm), np.abs(r_PINN)


def dirichlet_residual_compare(problem_constants, N_x,T_fvm,T_right_fvm,coPINN,device):
    L = problem_constants['L']


    x = np.linspace(0,L,N_x).reshape(-1,)
    r_fvm = residual_fvm_robin_cubicspline(x.reshape(-1,),T_fvm,L,T_right_fvm,problem_constants)  

    x_test_tensor = torch.from_numpy(L).float().to(device)
    #f1_PINN,f2_PINN = coPINN.residuals_f1f2(x_test_tensor)
    r_PINN = coPINN.residuals_robin_BC(x_test_tensor).cpu().detach().numpy()

    return np.abs(r_fvm), np.abs(r_PINN)