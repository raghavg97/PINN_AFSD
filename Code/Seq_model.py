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

global device, E_a, R, R0, Rs, A, Omega, log_A,eeta, n, k, c_p, alpha_sig, V, rho, k_B, pi, mu, T_a, h_sides, C_bot  
device= ipython.user_ns['device']
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
# global device

class Sequentialmodel(nn.Module):

    def __init__(self,layers1,layers2,lb_xyz,ub_xyz):
        super().__init__() #call __init__ from parent class

        print(device)
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.layers1 = layers1
        self.layers2 = layers2
        
        'Initialise neural network as a list using nn.Modulelist'
        self.linears1 = nn.ModuleList([nn.Linear(layers1[i], layers1[i+1]) for i in range(len(layers1)-1)])

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers1)-1):
            nn.init.xavier_normal_(self.linears1[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears1[i].bias.data)

        self.beta1 = Parameter(torch.ones((50,len(layers1)-2)))
        self.beta1.requiresGrad = True
        
        self.linears2 = nn.ModuleList([nn.Linear(layers2[i], layers2[i+1]) for i in range(len(layers2)-1)])

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers2)-1):
            nn.init.xavier_normal_(self.linears2[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears2[i].bias.data)

        self.beta2 = Parameter(torch.ones((50,len(layers2)-2)))
        self.beta2.requiresGrad = True
            
        self.lb_xyz =  torch.from_numpy(lb_xyz).float().to(device)    
        self.ub_xyz =  torch.from_numpy(ub_xyz).float().to(device)
        
        
        
    'foward pass'
    def forward1(self,xyz):
        if torch.is_tensor(xyz) != True:
            xyz = torch.from_numpy(xyz)


        #preprocessing input
        mp = (self.lb_xyz+self.ub_xyz)/2
        xyz = 2*(xyz - mp)/(self.ub_xyz - self.lb_xyz)

        #convert to float
        a = xyz.float()

        for i in range(len(self.layers1)-2):
            z = self.linears1[i](a)
            # a = self.activation(z)
            z1 = self.activation(z)
            a = z1 + self.beta1[:,i]*z*z1


        a = self.linears1[-1](a)    
            
        return a
    
    def forward2(self,xyz):
        if torch.is_tensor(xyz) != True:
            xyz = torch.from_numpy(xyz)


        #preprocessing input
        mp = (self.lb_xyz+self.ub_xyz)/2
        xyz = 2*(xyz - mp)/(self.ub_xyz - self.lb_xyz)

        #convert to float
        a = xyz.float()

        for i in range(len(self.layers2)-2):
            z = self.linears2[i](a)
            # a = self.activation(z)
            z1 = self.activation(z)
            a = z1 + self.beta2[:,i]*z*z1

        a = self.linears2[-1](a)    
            
        return a
    
    
    def loss_B_top(self,xyz_top,N_hat):
        g = xyz_top.clone()
        g.requires_grad = True
        
        #Inefficient Code
        r = torch.sqrt(torch.square(xyz_top[:,0]) + torch.square(xyz_top[:,1]))
        
        r_fr = (r<R0).reshape(-1,1)
        r_ph = torch.logical_and(r>=R0,r<=Rs).reshape(-1,1)
        r_out = torch.logical_not(r<=Rs).reshape(-1,1)
        
        cos_theta = xyz_top[:,1]/r #
        sin_theta = xyz_top[:,0]/r #
        
        r = r.reshape(-1,1)
        cos_theta = cos_theta.reshape(-1,1)
        sin_theta = sin_theta.reshape(-1,1)
         
        out_top = self.forward1(g)
        
        delta = 1-torch.exp(-A*(r-R0)/Rs)
        
        #BC1 #Exponential 
        u_true_ph = (1-delta)*(2*pi/60)*Omega*r*cos_theta
        v_true_ph = (1-delta)*(2*pi/60)*Omega*r*sin_theta - V
        w_true_ph = 0.0
    
        
        #BC2
        u_true_fr = (2*pi/60)*Omega*r*cos_theta
        v_true_fr = (2*pi/60)*Omega*r*sin_theta - V
        w_true_fr = -F
        
        #OTHER
        u_true_out = 0.0
        v_true_out = -V
        w_true_out = 0.0
    
        u_true = u_true_fr*r_fr + u_true_ph*r_ph + u_true_out*r_out
        v_true = v_true_fr*r_fr + v_true_ph*r_ph + v_true_out*r_out
        w_true = w_true_fr*r_fr + w_true_ph*r_ph + w_true_out*r_out
        
        u = out_top[:,0:1]
        v = out_top[:,1:2]
        w = out_top[:,2:3]
        
        loss_top_D = self.loss_function(u,u_true) + self.loss_function(v,v_true) + self.loss_function(w,w_true)
        
        u_xyz = autograd.grad(u,g,torch.ones([xyz_top.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_xyz = autograd.grad(v,g,torch.ones([xyz_top.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w_xyz = autograd.grad(w,g,torch.ones([xyz_top.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
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
        
          #Neumann T at top
        T = self.forward2(g)
        # print(T.shape)
        # print(eps_e.shape)
        # Z = eps_e*torch.exp(E_a/(R*T))
        # log_Z = torch.log(eps_e) + E_a/(R*T)
        if(torch.mean(T.detach())<200):
            log_Z = torch.log(eps_e) + E_a/(R*500.0) #Simplification
        else:
            log_Z = torch.log(eps_e) + E_a/(R*T)
        
        W = (log_Z - log_A)/n        
        # sigma_e =  (1/alpha_sig)*torch.asinh(torch.pow(Z/A,1/n)) 
        sigma_e = (1/alpha_sig)*(np.log(2)/n + W) #Approximation
        
        q_ph = eeta*(0.9*(1-delta)*(sigma_e/np.sqrt(3)) + delta*mu*sigma_e)*(2*pi/60)*Omega*r
        q_fr = 0.5*(0.9*(sigma_e/np.sqrt(3)))*(2*pi/60)*Omega*r
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_top.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        q = q_fr*r_fr + q_ph*r_ph + 0*r_out
       
        
        f = k*T_xyz[:,2] - q.reshape(-1,)

        
        loss_top_N = self.loss_function(f.reshape(-1,1),N_hat)
        
        return loss_top_D + loss_top_N  
    
    
    def loss_B_uvw(self,xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot):
        xyz_B = torch.vstack((xyz_1,xyz_2,xyz_3,xyz_4,xyz_bot))
        
        out_B = self.forward1(xyz_B)
        
        #OTHER
        u_true = 0.0*torch.ones((out_B.shape[0],1),device = device)
        v_true = -V*torch.ones((out_B.shape[0],1),device = device)
        w_true = 0.0*torch.ones((out_B.shape[0],1),device = device)
        
        u = out_B[:,0:1]
        v = out_B[:,1:2]
        w = out_B[:,2:3]
        
        loss_B = self.loss_function(u,u_true) + self.loss_function(v,v_true) + self.loss_function(w,w_true)
        
        
        return loss_B 
    
    
    def loss_NB_T(self,xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot):
            
        #Side 1    y_min
        g = xyz_1.clone()
        g.requires_grad = True
        T = self.forward2(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_1.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = k*T_y + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_1 = self.loss_function(f,f_true)
        
        #Side 2    x_max
        g = xyz_2.clone()
        g.requires_grad = True
        T = self.forward2(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_2.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = -k*T_x + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_2 = self.loss_function(f,f_true)
        
        #Side 3    y_max
        g = xyz_3.clone()
        g.requires_grad = True
        T = self.forward2(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_3.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = -k*T_y + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_3 = self.loss_function(f,f_true)
        
        #Side 4    x_min
        g = xyz_4.clone()
        g.requires_grad = True
        T = self.forward2(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_4.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        # T_z = T_xyz[:,2:3]
        
        f = k*T_x + h_sides*(T_a - T)
        f_true = 0.0*f.detach()
        
        loss_4 = self.loss_function(f,f_true)
        
        #Side 5    
        g = xyz_bot.clone()
        g.requires_grad = True
        T = self.forward2(g)
        
        T_xyz = autograd.grad(T,g,torch.ones([xyz_bot.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        # T_x = T_xyz[:,0:1]
        # T_y = T_xyz[:,1:2]
        T_z = T_xyz[:,2:3]
        
        f = k*T_z + C_bot*torch.pow(T_a - T,3)
        f_true = 0.0*f.detach()
        
        loss_5 = self.loss_function(f,f_true)
        
        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5
    
    def loss_PDE(self, xyz_coll_batch, f_hat_batch):
        
        

        #Batching Checks
        # print(i)
        # i += 1
        # print("#elements in batch ", xyz_coll_batch.shape[0])

        g = xyz_coll_batch.clone()
        g.requires_grad = True
        out_full = self.forward1(g)
        
        
        u = out_full[:,0:1]
        v = out_full[:,1:2]
        w = out_full[:,2:3]
        p = out_full[:,3:4]
        T = self.forward2(g)
        
        # print(T.shape)
                    
        p_xyz = autograd.grad(p,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        u_xyz = autograd.grad(u,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_xyz = autograd.grad(v,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w_xyz = autograd.grad(w,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
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
        # log_Z = torch.log(eps_e) + E_a/(R*500.0) #Simplification
        
        if(torch.mean(T.detach())<200):
            log_Z = torch.log(eps_e) + E_a/(R*500.0) #Simplification
        else:
            log_Z = torch.log(eps_e) + E_a/(R*T)
    
        W = (log_Z - log_A)/n
        
        
        
        # sigma_e =  (1/alpha_sig)*torch.asinh(W) 
        sigma_e = (1/alpha_sig)*(np.log(2)/n + W) #Approximation
        
        #____________________________#
        mu_vis = sigma_e/(3*eps_e)
        
        q_g = 0.9*sigma_e*eps_e
        #____________________________#      
        # print(torch.mean(sigma_e).cpu().detach().numpy())
    
        
        u2 = u*u
        v2 = v*v
        w2 = w*w
        uv = u*v
        uw = u*w
        vw = v*w
        
        u2_xyz =  autograd.grad(u2,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v2_xyz =  autograd.grad(u2,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w2_xyz =  autograd.grad(u2,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        uv_xyz =  autograd.grad(uv,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        uw_xyz =  autograd.grad(uw,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        vw_xyz =  autograd.grad(vw,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]     

        
        u_x = mu_vis*u_xyz[:,0:1]
        u_y = mu_vis*u_xyz[:,1:2]
        u_z = mu_vis*u_xyz[:,2:3]
        
        v_x = mu_vis*v_xyz[:,0:1]
        v_y = mu_vis*v_xyz[:,1:2]
        v_z = mu_vis*v_xyz[:,2:3]
        
        w_x = mu_vis*w_xyz[:,0:1]
        w_y = mu_vis*w_xyz[:,1:2]
        w_z = mu_vis*w_xyz[:,2:3]
                
        u_x_xyz = autograd.grad(u_x,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        u_y_xyz = autograd.grad(u_y,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        u_z_xyz = autograd.grad(u_z,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        
        v_x_xyz = autograd.grad(v_x,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_y_xyz = autograd.grad(v_y,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_z_xyz = autograd.grad(v_z,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        
        
        w_x_xyz = autograd.grad(w_x,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w_y_xyz = autograd.grad(w_y,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        w_z_xyz = autograd.grad(w_z,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        
        #Navier Stokes
        f1 = rho*(u2_xyz[:,0] + uv_xyz[:,1] + vw_xyz[:,2]) + p_xyz[:,0] - 2*u_x_xyz[:,0] - (v_x_xyz[:,1] + u_y_xyz[:,1]) - (w_x_xyz[:,2] + u_z_xyz[:,2]) 
        f2 = rho*(uv_xyz[:,0] + v2_xyz[:,1] + vw_xyz[:,2]) + p_xyz[:,1] - (u_y_xyz[:,0] + v_x_xyz[:,0]) - 2*v_y_xyz[:,1]  - (w_y_xyz[:,2] + v_z_xyz[:,2])
        f3 = rho*(uw_xyz[:,0] + vw_xyz[:,1] + w2_xyz[:,2]) + p_xyz[:,2] - (u_z_xyz[:,0] + w_x_xyz[:,0]) - (v_z_xyz[:,1] + w_y_xyz[:,1]) - 2*w_z_xyz[:,2]
        
        f4 = u_xyz[:,0] + v_xyz[:,1] + w_xyz[:,2]
        
        #Heat Transfer 
        T_xyz = autograd.grad(T,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        T_x = T_xyz[:,0:1]
        T_y = T_xyz[:,1:2]
        T_z = T_xyz[:,2:3]
        
        T_x_xyz = autograd.grad(T_x,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        T_y_xyz = autograd.grad(T_y,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        T_z_xyz = autograd.grad(T_z,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        kT_xx = k*T_x_xyz[:,0]
        kT_yy = k*T_y_xyz[:,1]
        kT_zz = k*T_z_xyz[:,2]
        
        uT_x = autograd.grad(u*T,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0][:,0]
        vT_y = autograd.grad(v*T,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0][:,1]
        wT_z = autograd.grad(w*T,g,torch.ones([xyz_coll_batch.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0][:,2]
        
        # f5 = kT_xx + kT_yy + kT_zz + q_g.reshape(-1,)/(rho*c_p) - (rho*c_p)*(uT_x + vT_y + wT_z)
        
        f5 = kT_xx + kT_yy + kT_zz - q_g.reshape(-1,) - (rho*c_p)*(uT_x + vT_y + wT_z)
        
        loss_f1 = self.loss_function(f1.reshape(-1,1),f_hat_batch)
        loss_f2 = self.loss_function(f2.reshape(-1,1),f_hat_batch)
        loss_f3 =  self.loss_function(f3.reshape(-1,1),f_hat_batch)
        loss_f4 = self.loss_function(f4.reshape(-1,1),f_hat_batch)
        loss_f5 =  self.loss_function(f5.reshape(-1,1),f_hat_batch)
        
        
        loss_f =  loss_f1+loss_f2 +loss_f3+loss_f4 +loss_f5
        
        return loss_f
        

    def loss(self,xyz_coll,xyz_1, xyz_2, xyz_3, xyz_4,xyz_top,xyz_bot,f_hat,N_hat):

        loss_PDE = self.loss_PDE(xyz_coll, f_hat)
        loss_B_top = self.loss_B_top(xyz_top,N_hat)
        loss_B = self.loss_B_uvw(xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot)
        loss_NB_T = self.loss_NB_T(xyz_1, xyz_2, xyz_3, xyz_4, xyz_bot)
            
        
#         print(loss_PDE.cpu().detach().numpy())
#         print(loss_B_top.cpu().detach().numpy())
#         print(loss_B.cpu().detach().numpy())
        
        loss_val = loss_PDE + loss_B_top + loss_B + loss_NB_T

        return loss_val

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