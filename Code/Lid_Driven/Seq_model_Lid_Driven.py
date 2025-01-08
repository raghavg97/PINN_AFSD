import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from IPython import get_ipython
ipython = get_ipython()

global Re
Re = ipython.user_ns['Re']


class Sequentialmodel(nn.Module):
    
    def __init__(self,layers,ub_xyz,lb_xyz,device):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
        
        beta_mean = 1.0*torch.ones((50,len(layers)-2))
        beta_std = 0.01*torch.ones((50,len(layers)-2))
        
        self.beta = Parameter(torch.normal(beta_mean,beta_std))
        self.beta.requiresGrad = True

        self.ub = torch.from_numpy(ub_xyz).float().to(device)
        self.lb = torch.from_numpy(lb_xyz).float().to(device)

        self.layers = layers

            
    'foward pass'
    def forward(self,xyt):
        if torch.is_tensor(xyt) != True:         
            xyt = torch.from_numpy(xyt)                
        
        # ubxyt = torch.from_numpy(ub_xyt).float().to(device)
        # lbxyt = torch.from_numpy(lb_xyt).float().to(device)
    
                      
        #preprocessing input 
        # xyt = 2.0*(xyt - lbxyt)/(ubxyt - lbxyt) - 1.0
        xyt = 2.0*(xyt - self.lb)/(self.ub - self.lb) - 1.0 #feature scaling
        
        #convert to float
        a = xyt.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            z1 = self.activation(z)
            a = z1 + self.beta[:,i]*z*z1            
        
        a = self.linears[-1](a) 
         
        return a
    

class PIdNN_lid_cavity(nn.Module):
    
    def __init__(self,layers1,layers2,layers3,lb_xyz,ub_xyz,device):
        super().__init__() #call __init__ from parent class 

        self.loss_function = nn.MSELoss(reduction ='mean')

        self.device = device

        self.PINN_u = Sequentialmodel(layers1,lb_xyz,ub_xyz,device).to(device)
        self.PINN_v = Sequentialmodel(layers2,lb_xyz,ub_xyz,device).to(device)
        self.PINN_p = Sequentialmodel(layers3,lb_xyz,ub_xyz,device).to(device)

        print(self.PINN_u)
        print(self.PINN_v)
        print(self.PINN_p)

    def loss_BC(self,xyt_BC,u_BC,v_BC):
                
        u = self.PINN_u.forward(xyt_BC)
        v = self.PINN_v.forward(xyt_BC)

        
        loss_bc_u = self.loss_function(u,u_BC)
        loss_bc_v = self.loss_function(v,v_BC)
        
        # psi_p_pred = self.forward(torch.cat((x1,y1,t1),dim =1))
        # psi = psi_p_pred[:,0:1]
        # p_pred = psi_p_pred[:,1:2]
                
        return loss_bc_u + loss_bc_v
    
    def loss_Ip(self,xyt_Ip,u_Ip,v_Ip,p_Ip):
                
        # uvp = self.forward(xyt_Ip)

        u = self.PINN_u.forward(xyt_Ip)
        v = self.PINN_v.forward(xyt_Ip)
        p = self.PINN_p.forward(xyt_Ip)
        
        loss_ip_u = self.loss_function(u,u_Ip)
        loss_ip_v = self.loss_function(v,v_Ip)
        loss_ip_p = self.loss_function(p,p_Ip)
        
        # psi_p_pred = self.forward(torch.cat((x1,y1,t1),dim =1))
        # psi = psi_p_pred[:,0:1]
        # p_pred = psi_p_pred[:,1:2]
                
        return loss_ip_u + loss_ip_v+loss_ip_p
    
    def loss_PDE(self, xyt_coll, f_hat):
        
        g = xyt_coll.clone()             
        g.requires_grad = True
        # uvp = self.forward(g) 
        
        # u = uvp[:,0:1]
        # v = uvp[:,1:2]
        # p = uvp[:,2:3]

        u = self.PINN_u.forward(g)
        v = self.PINN_v.forward(g)
        p = self.PINN_p.forward(g)
        
        #u
        u_xyt = autograd.grad(u,g,torch.ones([xyt_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
    
        u_xx_yy_tt = autograd.grad(u_xyt,g,torch.ones(xyt_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]

        du_dx = u_xyt[:,[0]]
        du_dy = u_xyt[:,[1]]
        du_dt = u_xyt[:,[2]]
        
        d2u_dx2 = u_xx_yy_tt[:,[0]]
        d2u_dy2 = u_xx_yy_tt[:,[1]]
        
        #v
        v_xyt = autograd.grad(v,g,torch.ones([xyt_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]

        v_xx_yy_tt = autograd.grad(v_xyt,g,torch.ones(xyt_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]

        dv_dx = v_xyt[:,[0]]
        dv_dy = v_xyt[:,[1]]
        dv_dt = v_xyt[:,[2]]
        
        d2v_dx2 = v_xx_yy_tt[:,[0]]
        d2v_dy2 = v_xx_yy_tt[:,[1]]
        
        #p
        p_xyt = autograd.grad(p,g,torch.ones([xyt_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]

        dp_dx = p_xyt[:,[0]]
        dp_dy = p_xyt[:,[1]]
    
        
        
        f_u = du_dt + u*du_dx + v*du_dy + dp_dx - (d2u_dx2 + d2u_dy2)/Re
        f_v = dv_dt + u*dv_dx + v*dv_dy + dp_dy - (d2v_dx2 + d2v_dy2)/Re
        
        
        loss_f_u = self.loss_function(f_u,f_hat)
        loss_f_v = self.loss_function(f_v,f_hat)
                
        return loss_f_u + loss_f_v
    
    
    def loss_NBC(self,xyt_coll,N_hat):
        g = xyt_coll.clone()             
        g.requires_grad = True
        # uvp = self.forward(g)
        
        u = self.PINN_u.forward(g)
        v = self.PINN_v.forward(g)
        # p = self.PINN_p.forward(g)
        # u = uvp[:,0:1]
        # v = uvp[:,1:2]
        
        u_xyt = autograd.grad(u,g,torch.ones([xyt_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        v_xyt = autograd.grad(v,g,torch.ones([xyt_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
        du_dx = u_xyt[:,[0]]
        dv_dy = v_xyt[:,[1]]
        
        loss_nbc = self.loss_function(du_dx + dv_dy, N_hat)
                
        return loss_nbc
    
    
    def loss(self,xyt_coll, xyt_BC,u_BC,v_BC, xyt_Ip, u_Ip,v_Ip, p_Ip,f_hat,N_hat):

        loss_BC = self.loss_BC(xyt_BC,u_BC,v_BC)
        loss_Ip = self.loss_Ip(xyt_Ip,u_Ip,v_Ip,p_Ip)
        loss_NBC = self.loss_NBC(xyt_coll,N_hat)
        loss_f = self.loss_PDE(xyt_coll,f_hat)
        
        loss_val = loss_BC + loss_f + loss_NBC +loss_Ip
        
        return loss_val
    