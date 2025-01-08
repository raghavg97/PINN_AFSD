import numpy as np
import time


def fvm_1D(pde_related_funcs,problem_constants,N_x,t_steps):
        
    alpha = problem_constants['alpha']
    beta = problem_constants['beta']
    gamma = problem_constants['gamma']
    D = problem_constants['D']
    t_end = problem_constants['Max_time']
    L = problem_constants['L']

    initial_TC = pde_related_funcs["initial_condition"]
    boundary_TC = pde_related_funcs["boundary_conditions"]
    forcing_func = pde_related_funcs["forcing_function"]



    dx = L / N_x

    dt =  t_end/(t_steps-1) 

    x = np.linspace(dx/2, L - dx/2, N_x)  # Cell centers
    t = np.linspace(0,t_end,t_steps)

    

    T = []
    C =[]
    # Time loop

    # T.append(np.sin(np.pi*x))  # Initial temperature distribution
    # C.append(np.cos(np.pi*x))  # Initial concentration distribution

    # T.append(np.ones((N,)))
    # C.append(np.ones((N,)))
    T_init,C_init = initial_TC(x)
    T1,T2,C1,C2 = boundary_TC(x)

    T.append(T_init)
    C.append(C_init)

    

    f = forcing_func(x)


    t = 0.0
    time_step = 0

    start_time = time.time()

    while t < t_end:
        # Compute fluxes
        T_flux = alpha * (np.roll(T[time_step], -1) - 2 * T[time_step] + np.roll(T[time_step], 1)) / dx**2
        C_flux = D * (np.roll(C[time_step], -1) - 2 * C[time_step] + np.roll(C[time_step], 1)) / dx**2

        # Update equations
        T_new = T[time_step] + dt * (T_flux + beta * C[time_step] + f)
        C_new = C[time_step] + dt * (C_flux + gamma * T[time_step])

        flux_left = 0.01*(T_new[1] - T_new[0]) / dx  # Approximate derivative term
        T_new[0] = (1-flux_left) / (1)

        flux_right = 0.01*(T_new[-1] - T_new[-2]) / dx # Approximate derivative term
        T_new[-1] = (1-flux_right) / (1)

        # Apply boundary conditions (Dirichlet: T = 0, C = 0 at boundaries)
        # T_new[0] = T1
        # T_new[-1] = T2
        C_new[0] = C1
        C_new[-1] = C2

        
        # errorT = np.sqrt(np.mean(np.square(T[time_step]-T_new)))
        # errorC = np.sqrt(np.mean(np.square(C[time_step]-C_new)))


        time_step+= 1

        # Update variables
        T.append(T_new)
        C.append(C_new)
        t += dt

        T_FVM = np.transpose(np.array(T))
        C_FVM = np.transpose(np.array(C))


    print("Elapsed Time %2f"%(time.time()-start_time))
    return T_FVM, C_FVM
