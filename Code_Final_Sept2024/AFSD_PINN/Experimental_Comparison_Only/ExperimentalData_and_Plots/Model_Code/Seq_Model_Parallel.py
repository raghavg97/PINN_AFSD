import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from IPython import get_ipython
ipython = get_ipython()


# V = 1 #mm/s
# F = 0.67 #mm
# rho = 2700 * 1e-6 #g/mm3
# k_B = 1.380649*1e-23 #J/K
# R = 8.314 #J/(K.mol)
# E_a = 205000 #J/mol #Q
# alpha_sig = 52 #mm^2/N

global E_a, R, R0, Rs, A, Omega, log_A,eeta, n, k, c_p, alpha_sig, V, rho, k_B, pi, mu, T_a, h_sides, C_bot, delta  
E_a= ipython.user_ns['E_a']
R= ipython.user_ns['R']
R0= ipython.user_ns['R0']
Rs= ipython.user_ns['Rs']
A= ipython.user_ns['A']
Omega= ipython.user_ns['Omega']
F= ipython.user_ns['F']
log_A= ipython.user_ns['log_A']
eeta= ipython.user_ns['eeta']
n= ipython.user_ns['n']
k= ipython.user_ns['k']
c_p= ipython.user_ns['c_p']
alpha_sig= ipython.user_ns['alpha_sig']
V= ipython.user_ns['V']
rho = ipython.user_ns['rho']
k_B = ipython.user_ns['k_B']
pi =  ipython.user_ns['pi']
mu = ipython.user_ns['mu']
T_a = ipython.user_ns['T_a']
h_sides = ipython.user_ns['h_sides']
C_bot = ipython.user_ns['C_bot']
delta = ipython.user_ns['delta']
# global device


class Sequentialmodel(nn.Module):
    
    def __init__(self,layers,device,lb_xyz,ub_xyz):
        super().__init__() #call __init__ from parent class 
              
    
        self.activation = nn.Tanh()
        self.device = device
        
        self.layers = layers
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data) 
        
        self.beta = Parameter(torch.ones((50,len(layers)-2)))
        self.beta.requiresGrad = True

        self.ub = torch.from_numpy(ub_xyz).float().to(self.device)
        self.lb = torch.from_numpy(lb_xyz).float().to(self.device)
    
    
    'forward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                

        #preprocessing input 
        x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0 #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z) + self.beta[:,i]*z*self.activation(z)
            
        a = self.linears[-1](a) 
         
        return a



class coupled_PINN(nn.Module):

    def __init__(self,layers1,layers2,device1,device2,lb_xyz,ub_xyz):
        super().__init__() #call __init__ from parent class 

        self.loss_function = nn.MSELoss(reduction ='mean')

        self.PINN_uvw = Sequentialmodel(layers1,device1,lb_xyz,ub_xyz).to(device1)
        self.PINN_T = Sequentialmodel(layers2,device2,lb_xyz,ub_xyz).to(device2)

        self.device1 = device1
        self.device2 = device2

        print(self.PINN_uvw)
        print(self.PINN_T)
    

    def helper_gradients(self,g,ind):

        #indicator takes 0, 1, or 2
        g.requires_grad = True
        g = g.to(self.device1)
        out_uvw = self.PINN_uvw.forward(g)
        
        u = out_uvw[:,0:1]
        v = out_uvw[:,1:2]
        w = out_uvw[:,2:3]
        p = out_uvw[:,3:4]
        
        u_xyz = autograd.grad(u,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_xyz = autograd.grad(v,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w_xyz = autograd.grad(w,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]


        if(ind==0):    

            p_xyz = autograd.grad(p,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]

            # u2 = u*u
            # v2 = v*v
            # w2 = w*w
            # uv = u*v
            # uw = u*w
            # vw = v*w
            
            u2_x =  autograd.grad(u*u,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0][:,0]
            v2_y =  autograd.grad(v*v,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0][:,1]
            w2_z =  autograd.grad(w*w,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0][:,2]
            uv_xyz =  autograd.grad(u*v,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            uw_xyz =  autograd.grad(u*w,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            vw_xyz =  autograd.grad(v*w,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]     
            
            T = self.PINN_T.forward(g.to(self.device2)).to(self.device1)
            sigma_e,eps_e = self.helper_sigma_eff(u_xyz,v_xyz, w_xyz,T)
           

            mu_vis = sigma_e/(3*eps_e)
            
            u_x = mu_vis*u_xyz[:,0:1]
            u_y = mu_vis*u_xyz[:,1:2]
            u_z = mu_vis*u_xyz[:,2:3]
            
            v_x = mu_vis*v_xyz[:,0:1]
            v_y = mu_vis*v_xyz[:,1:2]
            v_z = mu_vis*v_xyz[:,2:3]
            
            w_x = mu_vis*w_xyz[:,0:1]
            w_y = mu_vis*w_xyz[:,1:2]
            w_z = mu_vis*w_xyz[:,2:3]
                    
            u_x_xyz = autograd.grad(u_x,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            u_y_xyz = autograd.grad(u_y,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            u_z_xyz = autograd.grad(u_z,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            
            
            v_x_xyz = autograd.grad(v_x,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            v_y_xyz = autograd.grad(v_y,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            v_z_xyz = autograd.grad(v_z,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            
            
            
            w_x_xyz = autograd.grad(w_x,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            w_y_xyz = autograd.grad(w_y,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
            w_z_xyz = autograd.grad(w_z,g,torch.ones([g.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]


            f1 = rho*1e-6*(u2_x + uv_xyz[:,1] + uw_xyz[:,2]) + p_xyz[:,0] - 2*u_x_xyz[:,0] - (v_x_xyz[:,1] + u_y_xyz[:,1]) - (w_x_xyz[:,2] + u_z_xyz[:,2]) 
            f2 = rho*1e-6*(uv_xyz[:,0] + v2_y + vw_xyz[:,2]) + p_xyz[:,1] - (u_y_xyz[:,0] + v_x_xyz[:,0]) - 2*v_y_xyz[:,1]  - (w_y_xyz[:,2] + v_z_xyz[:,2])
            f3 = rho*1e-6*(uw_xyz[:,0] + vw_xyz[:,1] + w2_z) + p_xyz[:,2] - (u_z_xyz[:,0] + w_x_xyz[:,0]) - (v_z_xyz[:,1] + w_y_xyz[:,1]) - 2*w_z_xyz[:,2]
            
            f4 = u_xyz[:,0] + v_xyz[:,1] + w_xyz[:,2]

            return f1,f2,f3,f4

        else:
            g = g.to(self.device2)
            T = self.PINN_T.forward(g)

            sigma_e,eps_e = self.helper_sigma_eff(u_xyz.to(self.device2),v_xyz.to(self.device2), w_xyz.to(self.device2),T)

            if(ind==1):
                T_xyz = autograd.grad(T,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
                return T_xyz,sigma_e,eps_e
        
            elif(ind==2):
        
                q_g = 0.9*sigma_e*eps_e

                T_xyz = autograd.grad(T,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
                
                T_x = T_xyz[:,0:1]
                T_y = T_xyz[:,1:2]
                T_z = T_xyz[:,2:3]
                
                T_x_xyz = autograd.grad(T_x,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
                T_y_xyz = autograd.grad(T_y,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
                T_z_xyz = autograd.grad(T_z,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
                
                kT_xx = k*T_x_xyz[:,0]
                kT_yy = k*T_y_xyz[:,1]
                kT_zz = k*T_z_xyz[:,2]

                uT_x = autograd.grad(u.to(self.device2)*T,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0][:,0]
                vT_y = autograd.grad(v.to(self.device2)*T,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0][:,1]
                wT_z = autograd.grad(w.to(self.device2)*T,g,torch.ones([g.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0][:,2]
                
                
                # f5 = kT_xx + kT_yy + kT_zz + q_g.reshape(-1,)/(rho*c_p) - (rho*c_p)*(uT_x + vT_y + wT_z)
                # q_g = q_g/1e6
                f5 = kT_xx + kT_yy + kT_zz - q_g.reshape(-1,) - (rho*c_p)*(uT_x + vT_y + wT_z)*1e-6

                return f5

    def helper_sigma_eff(self,u_xyz,v_xyz,w_xyz,T):

        eps2_11 = torch.square(1/2*(2*u_xyz[:,0]))
        eps2_12 = torch.square(1/2*(u_xyz[:,1] + v_xyz[:,0]))
        eps2_13 = torch.square(1/2*(u_xyz[:,2] + w_xyz[:,0]))
        
        eps2_21 = eps2_12
        eps2_22 = torch.square(1/2*(2*v_xyz[:,1])) 
        eps2_23 = torch.square(1/2*(v_xyz[:,2] + w_xyz[:,1]))
        
        eps2_31 = eps2_13
        eps2_32 = eps2_23 
        eps2_33 = torch.square(1/2*(2*w_xyz[:,2]))
        
        eps_e = torch.sqrt((2/3)*(eps2_11 + eps2_12 + eps2_13 + eps2_21 + eps2_22 + eps2_23 + eps2_31 + eps2_32 + eps2_33)).reshape(-1,1)

        # Z = eps_e*torch.exp(E_a/(R*T))
        # log_Z = torch.log(eps_e) + E_a/(R*T)
        if(torch.mean(T.detach())<200.0):
            log_Z = torch.log(eps_e) + E_a/(R*300.0) #Simplification
        else:
            log_Z = torch.log(eps_e) + E_a/(R*T)
        
        W = (log_Z - log_A)/n        
        # sigma_e =  (1/alpha_sig)*torch.asinh(torch.pow(Z/A,1/n)) 
        sigma_e = (1/alpha_sig)*(np.log(2)/n + W) #Approximation
        

        return sigma_e,eps_e

    
    def loss_D_top_uvw(self,xyz_top,uvw_true_top):
        g = xyz_top.clone().to(self.device1)
        g.requires_grad = True
        
        u_true_top = uvw_true_top[:,0].reshape(-1,1)#.to(self.device1)
        v_true_top = uvw_true_top[:,1].reshape(-1,1)#.to(self.device1)
        w_true_top = uvw_true_top[:,2].reshape(-1,1)#.to(self.device1)
        
        out_top = self.PINN_uvw.forward(g)
        
        u = out_top[:,0:1]
        v = out_top[:,1:2]
        w = out_top[:,2:3]
        
        loss_top_D = self.loss_function(u,u_true_top) + self.loss_function(v,v_true_top) + self.loss_function(w,w_true_top)
        
        return loss_top_D  
    
    def loss_N_top_T(self,xyz_top,N_hat,r_fr_ph_out):
          #Neumann T at top
        g = xyz_top.clone()

        T_xyz,sigma_e,_ = self.helper_gradients(g,1)

        r = (torch.sqrt(torch.square(xyz_top[:,0]) + torch.square(xyz_top[:,1]))).reshape(-1,1).to(self.device2)

        r_fr = r_fr_ph_out[:,0].reshape(-1,1).to(self.device2)
        r_ph = r_fr_ph_out[:,1].reshape(-1,1).to(self.device2)
        r_out = r_fr_ph_out[:,2].reshape(-1,1).to(self.device2)

        # print(sigma_e.shape)
        # print(r.shape)

        q_ph = eeta*(0.9*(1-delta)*(sigma_e/np.sqrt(3)) + delta*mu*sigma_e)*(2*pi/60)*Omega*r
        q_fr = 0.5*(0.9*(sigma_e/np.sqrt(3)))*(2*pi/60)*Omega*r
        
        q = (q_fr*r_fr + q_ph*r_ph + 0*r_out)
        
        f = k*T_xyz[:,2] - q.reshape(-1,)

        loss_top_N = self.loss_function(f.reshape(-1,1),N_hat)

        return loss_top_N
    
    def loss_B_uvw_5sides(self,xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot):
        xyz_B = torch.vstack((xyz_1,xyz_2,xyz_3,xyz_4,xyz_bot)).to(self.device1)
        
        out_B = self.PINN_uvw.forward(xyz_B)
        
        #OTHER
        u_true = 0.0*torch.ones((out_B.shape[0],1),device = self.device1)
        v_true = -V*torch.ones((out_B.shape[0],1),device = self.device1)
        w_true = 0.0*torch.ones((out_B.shape[0],1),device = self.device1)
        
        u = out_B[:,0:1]
        v = out_B[:,1:2]
        w = out_B[:,2:3]
        
        loss_B = self.loss_function(u,u_true) + self.loss_function(v,v_true) + self.loss_function(w,w_true)
        
        
        return loss_B 
    
    
    def loss_NB_T_5sides(self,xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot):
            
        #Side 1    y_min
        g = xyz_1.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_1.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = k*T_y + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_1 = self.loss_function(f,f_true)
        
        #Side 2    x_max
        g = xyz_2.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_2.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = -k*T_x + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_2 = self.loss_function(f,f_true)
        
        #Side 3    y_max
        g = xyz_3.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_3.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = -k*T_y + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_3 = self.loss_function(f,f_true)
        
        #Side 4    x_min
        g = xyz_4.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_4.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = k*T_x + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_4 = self.loss_function(f,f_true)
        
        #Side 5    
        g = xyz_bot.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_bot.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        T_z = T_xyz[:,2:3]
        
        f = k*T_z + C_bot*torch.pow(T_a - T,3)
        f_true = 0.0*f.detach()
        
        loss_5 = self.loss_function(f,f_true)
        
        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5

    def loss_NB_T_5sides_v2(self,xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot):
            
        #Side 1    y_min
        g = xyz_1.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_1.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f1 = k*T_y + h_sides*(T_a - T)
        
        #Side 2    x_max
        g = xyz_2.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_2.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f2 = -k*T_x + h_sides*(T_a - T)
        
        #Side 3    y_max
        g = xyz_3.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_3.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f3 = -k*T_y + h_sides*(T_a - T)
       
        
        #Side 4    x_min
        g = xyz_4.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_4.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f4 = k*T_x + h_sides*(T_a - T)
    
        
        #Side 5    
        g = xyz_bot.clone().to(self.device2)
        g.requires_grad = True
        T = self.PINN_T.forward(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_bot.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        T_z = T_xyz[:,2:3]
        
        f5 = k*T_z + C_bot*torch.pow(T_a - T,3)
        
        
        f_conv = torch.cat((f1,f2,f3,f4,f5))
        f_true = 0.0*f_conv.detach()

        loss_conv = self.loss_function(f_conv,f_true)

        
        return loss_conv
    
    def loss_PDE_uvw(self, xyz_coll_batch, f_hat_batch):        

        #Batching Checks
        # print(i)
        # i += 1
        # print("#elements in batch ", xyz_coll_batch.shape[0])
        g = xyz_coll_batch.clone()
    
        f1,f2,f3,f4 = self.helper_gradients(g,0)

        
        loss_f1 = self.loss_function(f1.reshape(-1,1),f_hat_batch)
        loss_f2 = self.loss_function(f2.reshape(-1,1),f_hat_batch)
        loss_f3 =  self.loss_function(f3.reshape(-1,1),f_hat_batch)
        loss_f4 = self.loss_function(f4.reshape(-1,1),f_hat_batch)
    
        
        loss_f =  loss_f1+loss_f2 +loss_f3+loss_f4 #+loss_f5
        
        return loss_f
        
    def loss_PDE_T(self,xyz_coll_batch, f_hat_batch):
        #Heat Transfer 
        g = xyz_coll_batch.clone()

        f5 = self.helper_gradients(g,2)

        loss_f5 =  self.loss_function(f5.reshape(-1,1),f_hat_batch)

        return loss_f5


    def loss(self,xyz_coll,xyz_1, xyz_2, xyz_3, xyz_4,xyz_top,xyz_bot,f_hat,N_hat,uvw_true_top,r_fr_ph_out):

        loss_uvw = self.loss_PDE_uvw(xyz_coll, f_hat.to(self.device1)) + self.loss_D_top_uvw(xyz_top,uvw_true_top) + self.loss_B_uvw_5sides(xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot)
        loss_T = self.loss_PDE_T(xyz_coll, f_hat.to(self.device2))  + self.loss_N_top_T(xyz_top,N_hat,r_fr_ph_out) + self.loss_NB_T_5sides_v2(xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot)
        
#         print(loss_PDE.cpu().detach().numpy())
#         print(loss_B_top.cpu().detach().numpy())
#         print(loss_B.cpu().detach().numpy())
        
        # loss_val = loss_PDE + loss_B_top + loss_B + loss_NB_T
        # print((loss_uvw.cpu().detach().numpy()/loss_T.cpu().detach().numpy()))

        return loss_uvw,loss_T

    def pretrain_T_loss(self,xyz_coll):

        T_amb = 300.0*torch.ones([xyz_coll.shape[0], 1]).to(self.device2)
        return self.loss_function(self.PINN_T.forward(xyz_coll.to(self.device2)),T_amb)


    'test neural network'
    def test(self):
     
        u_pred= 0

        return u_pred

    def test_loss(self):
        u_pred = self.test()

        # test_mse = np.mean(np.square(u_pred.reshape(-1,1) - u_true.reshape(-1,1)))
        # test_re = np.linalg.norm(u_pred.reshape(-1,1) - u_true.reshape(-1,1),2)/u_true_norm
        test_mse = 0
        test_re = 0

        return test_mse, test_re
    

   