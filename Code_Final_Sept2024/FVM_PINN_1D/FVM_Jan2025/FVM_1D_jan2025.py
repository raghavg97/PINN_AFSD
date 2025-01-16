import numpy as np
import time


def fvm_1D(pde_related_funcs,problem_constants,N_x,tol_T,tol_C,max_time):
        
    alpha = problem_constants['alpha']
    beta = problem_constants['beta']
    L = problem_constants['L']

    h_c = problem_constants['h_c']
    T_a = problem_constants['T_a']
    C_left = problem_constants['C_left']
    C_right = problem_constants['C_right']

    Q = pde_related_funcs["heat_source_func"]
    S = pde_related_funcs["conc_force_func"]


    del_x = L / N_x
    T_left = T_a


    T = 100*np.ones((N_x+1,),dtype='double')
    C = np.ones((N_x,),dtype='double')



    iters = 0
    eps_T = 50 #initializing eps to a large value
    eps_C = 50 #initializing eps to a large value
 


    n = N_x    

    start_time = time.time()

    while (eps_T > tol_T or eps_C > tol_C):

        #T_update
        T_old = T.copy()

        T[0] = (T[1] + 2*T_left - Q(C[0])*(del_x**2)/ alpha) / 3
        T[n-1] = h_c*(T[n] - T_a)*del_x + T[n-2] - (Q(C[n-1])/alpha)* (del_x**2)

        T[n] = (h_c*T_a - 2*T[n-1]/del_x)/(h_c - 2/del_x)
        T[1:n-1] = (T[0:n-2] + T[2:n])/2 - Q(C[1:n-1])*(del_x**2)/(2*alpha)


        #C_update
        C_old = C.copy()

        C[0] = (C[1] + 2*C_left - S(T[0])*(del_x**2)/beta)/3
        C[n-1] = (2*C_right + C[n-2] - S(T[n-1])*(del_x**2)/beta)/3
        C[1:n-1] = (C[2:n] + C[0:n-2])/2 - S(T[1:n-1])*(del_x**2)/(2*beta) 

        # C[0] = (C[1] + 2*C_left - S(T[1])*(del_x**2)/beta)/3
        # C[n-1] = (2*C_right + C[n-2] - S(T[n])*(del_x**2)/beta)/3
        # C[1:n-1] = (C[2:n] + C[0:n-2])/2 - S(T[2:n])*(del_x**2)/(2*beta) 


        # T_new = T.copy()
        eps_T = np.max(np.abs(T_old - T))
        eps_C = np.max(np.abs(C_old - C))
        # print(eps)
            

        iters += 1

        if(iters%10000==0):
            print("Iter: ", iters, "eps_T: ", eps_T, "eps_C: ", eps_C)

        # if (iters>5000000):
        if(time.time()-start_time>max_time):
            print("Max Time reached")
            print("eps T: ",eps_T)
            print("eps C: ",eps_C)
            break


    print("Elapsed Time %2f"%(time.time()-start_time))
    return T, C,iters, eps_T, eps_C
