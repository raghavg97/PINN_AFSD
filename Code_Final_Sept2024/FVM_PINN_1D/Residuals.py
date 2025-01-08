from scipy.interpolate import RectBivariateSpline
import numpy as np
import torch

from NN_interpolator import NN_interpolator


def residuals_fvm_NN(x,t,T_matrix,C_matrix,x_test,t_test,
                         pde_related_funcs,problem_constants,NN_constants):
    
    alpha = problem_constants['alpha']
    beta = problem_constants['beta']
    gamma = problem_constants['gamma']
    D = problem_constants['D']
    

    forcing_func = pde_related_funcs["forcing_function"]


    
    T_interpolator = NN_interpolator(x,t,T_matrix,NN_constants)
    T_x_t,T_xx_tt = T_interpolator.derivatives(x_test,t_test)

    
    C_interpolator = NN_interpolator(x,t,C_matrix,NN_constants)
    C_x_t,C_xx_tt = C_interpolator.derivatives(x_test,t_test)

    test_size1 =np.shape(x_test)[0]
    test_size2 =np.shape(t_test)[0]

    T_matrix_test = T_interpolator.interpolate(x_test,t_test).reshape(test_size1,test_size2,order = 'F')
    dT_dt_test = T_x_t[:,[1]].reshape(test_size1,test_size2,order = 'F')
    d2T_dx2_test = T_xx_tt[:,[0]].reshape(test_size1,test_size2,order = 'F')

    C_matrix_test = C_interpolator.interpolate(x_test,t_test).reshape(test_size1,test_size2,order = 'F')
    dC_dt_test = C_x_t[:,[1]].reshape(test_size1,test_size2,order = 'F')
    d2C_dx2_test = C_xx_tt[:,[0]].reshape(test_size1,test_size2,order = 'F')

    # d2T_dx2_interpolator = T_interpolator.partial_derivative(dx=2,dy = 0)
    # dT_dt_interpolator = T_interpolator.partial_derivative(dx=0,dy = 1)

    # # T_test = T_interpolator(x_test,t_test)
    # dT_dt_test = dT_dt_interpolator(x_test,t_test)
    # d2T_dx2_test = d2T_dx2_interpolator(x_test,t_test)

    f = forcing_func(x_test).reshape(-1,1)

    # print(dT_dt_test.shape)
    # print(d2T_dx2_test.shape)
    # print(f.shape)

    f1_test =  dT_dt_test - alpha*d2T_dx2_test - beta*C_matrix_test - f


    # C_interpolator = RectBivariateSpline(x,t,C_matrix)
    # d2C_dx2_interpolator = C_interpolator.partial_derivative(dx=2,dy = 0)
    # dC_dt_interpolator = C_interpolator.partial_derivative(dx=0,dy = 1)

    # # C_test = C_interpolator(x_test,t_test)
    # dC_dt_test = dC_dt_interpolator(x_test,t_test)
    # d2C_dx2_test = d2C_dx2_interpolator(x_test,t_test)


    f2_test =  dC_dt_test - D*d2C_dx2_test - gamma*T_matrix_test 

    return f1_test,f2_test

def residuals_fvm_spline(x,t,T_matrix,C_matrix,x_test,t_test,
                         pde_related_funcs,problem_constants):
    
    alpha = problem_constants['alpha']
    beta = problem_constants['beta']
    gamma = problem_constants['gamma']
    D = problem_constants['D']
    
    forcing_func = pde_related_funcs["forcing_function"]
    
    T_interpolator = RectBivariateSpline(x,t,T_matrix)
    d2T_dx2_interpolator = T_interpolator.partial_derivative(dx=2,dy = 0)
    dT_dt_interpolator = T_interpolator.partial_derivative(dx=0,dy = 1)

    C_interpolator = RectBivariateSpline(x,t,C_matrix)
    d2C_dx2_interpolator = C_interpolator.partial_derivative(dx=2,dy = 0)
    dC_dt_interpolator = C_interpolator.partial_derivative(dx=0,dy = 1)

    C_matrix_test = C_interpolator(x_test,t_test)
    dC_dt_test = dC_dt_interpolator(x_test,t_test)
    d2C_dx2_test = d2C_dx2_interpolator(x_test,t_test)

    T_matrix_test = T_interpolator(x_test,t_test)
    dT_dt_test = dT_dt_interpolator(x_test,t_test)
    d2T_dx2_test = d2T_dx2_interpolator(x_test,t_test)



    f = forcing_func(x_test).reshape(-1,1)

    f1_test =  dT_dt_test - alpha*d2T_dx2_test - beta*C_matrix_test - f

    print(dC_dt_test.shape)
    print(d2C_dx2_test.shape)
    print(T_matrix_test.shape)

    f2_test =  dC_dt_test - D*d2C_dx2_test - gamma*T_matrix_test

    return f1_test,f2_test
    

def res_compare(opt,interp_method,
                pde_related_funcs,problem_constants, N_x_test, t_steps_test,
                N_x,t_steps,T_FVM,C_FVM,coPINN,device,
                NN_interp_constants=None):
    
    t_end = problem_constants['Max_time']
    L = problem_constants['L']


    x = np.linspace(0,L,N_x).reshape(-1,1)
    t = np.linspace(0,t_end,t_steps).reshape(-1,1)

    if(opt == 'grid'):
        x_test = x
        t_test = t
    elif(opt == 'neutral'):
        x_test = np.sort(np.random.uniform(L/10,L - L/10,(N_x_test,)))
        t_test = np.sort(np.random.uniform(t_end/10,t_end - t_end/10,(t_steps_test,)),axis = 0)
    else:
        print("Give grid or neutral option")
        return 

    x_pinn,t_pinn = np.meshgrid(x_test,t_test)

    x_pinn = x_pinn.flatten('F').reshape(-1,1)
    t_pinn = t_pinn.flatten('F').reshape(-1,1)
    
    xt_pinn_test= np.hstack((x_pinn,t_pinn))
    
    xt_test_tensor = torch.from_numpy(xt_pinn_test).float().to(device)

    # T_PINN = PINN.forward1(xt_test_tensor).cpu().detach().numpy()
    # C_PINN = PINN.forward2(xt_test_tensor).cpu().detach().numpy()

    # T_PINN_matrix = T_PINN.reshape(N_x,t_steps)
    T_FVM_matrix = T_FVM.reshape(N_x,t_steps)

    # C_PINN_matrix = C_PINN.reshape(N_x,t_steps)
    C_FVM_matrix = C_FVM.reshape(N_x,t_steps)

    f1_PINN,f2_PINN = coPINN.residuals_f1f2(xt_test_tensor)
    f1_PINN = f1_PINN.reshape(N_x_test,t_steps_test).cpu().detach().numpy()
    f2_PINN = f2_PINN.reshape(N_x_test,t_steps_test).cpu().detach().numpy()


    if(interp_method == 'spline'):
        f1_fvm,f2_fvm = residuals_fvm_spline(x,t,T_FVM_matrix,C_FVM_matrix,
                                         x_test,t_test,
                                         pde_related_funcs,problem_constants)
    elif(interp_method == 'NN'):
            f1_fvm,f2_fvm = residuals_fvm_NN(x,t,T_FVM_matrix,C_FVM_matrix,
                                         x_test,t_test,
                                         pde_related_funcs,problem_constants,
                                         NN_interp_constants)
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

    return f1_PINN,f1_fvm,f2_PINN,f2_fvm