import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from IPython import get_ipython
import copy
ipython = get_ipython()


global mu, ub, lb
mu = ipython.user_ns['mu']
ub = ipython.user_ns['ub']
lb = ipython.user_ns['lb']


class Sequentialmodel(nn.Module):
    
    def __init__(self,layers,device):
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

        self.ub = torch.from_numpy(ub).float().to(self.device)
        self.lb = torch.from_numpy(lb).float().to(self.device)
    
    
    'forward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        x = x.to(self.device)
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

    def __init__(self,layers1,layers2,device1,device2):
        super().__init__() #call __init__ from parent class 

        self.loss_function = nn.MSELoss(reduction ='mean')
        self.PINN_y1 = Sequentialmodel(layers1,device1).to(device1)
        self.PINN_y2 = Sequentialmodel(layers2,device2).to(device2)

        self.device1 = device1
        self.device2 = device2

        print(self.PINN_y1)
        print(self.PINN_y2)



    def loss_BC1(self,x,y):
            
        loss_bc1 = self.loss_function(self.PINN_y1.forward(x.to(self.device1)), y.to(self.device1))
                
        return loss_bc1
    
    def loss_BC2(self,x,y):
                
        loss_bc2 = self.loss_function(self.PINN_y2.forward(x.to(self.device2)), y.to(self.device2))
                
        return loss_bc2
    

    def loss_PDE1(self, x_coll,f_hat1):
             
        g1 = x_coll.clone().to(self.device1)      
        g1.requires_grad = True
  
        # print(g.get_device())
        y1 = self.PINN_y1.forward(g1)
        y1_x = autograd.grad(y1,g1,torch.ones([x_coll.shape[0], 1]).to(self.device1), retain_graph=True, create_graph=True,allow_unused = True)[0]
        dy1_dx = y1_x[:,[0]]

        # print(y1.get_device()) 

        g2 = x_coll.clone().to(self.device2)   
        y2 = self.PINN_y2.forward(g2).to(self.device1)
        # print(y2.get_device())
        f = dy1_dx - y2 
        
        loss_f = self.loss_function(f,f_hat1)
                
        return loss_f
    
    def loss_PDE2(self, x_coll,f_hat):
             
        g2 = x_coll.clone().to(self.device2)            
        g2.requires_grad = True
  
        y2 = self.PINN_y2.forward(g2)

        y2_x = autograd.grad(y2,g2,torch.ones([x_coll.shape[0], 1]).to(self.device2), retain_graph=True, create_graph=True,allow_unused = True)[0]
       
        dy2_dx = y2_x[:,[0]]
        
        g1 = x_coll.clone().to(self.device1)
        y1 = self.PINN_y1.forward(g1).to(self.device2)
        f = dy2_dx - mu*(1-torch.square(y1))*y2 + y1
        
        loss_f = self.loss_function(f,f_hat)
                
        return loss_f

    
    def loss_y1(self,x_bc1,y_bc1,x_coll,f_hat):

        loss_bc1 = self.loss_BC1(x_bc1,y_bc1)
        loss_f = self.loss_PDE1(x_coll,f_hat)
        
        loss_val = loss_bc1 + loss_f
        
        return loss_val
    
    def loss_y2(self,x_bc2,y_bc2,x_coll,f_hat):

        loss_bc2 = self.loss_BC2(x_bc2,y_bc2)
        loss_f = self.loss_PDE2(x_coll,f_hat)
        
        loss_val = loss_bc2 + loss_f
        
        return loss_val
    
    'test neural network'
    
    def test(self,x_test_tensor):
        PINN_test = copy.deepcopy(self.PINN_y1).to('cpu')
        PINN_test.ub = PINN_test.ub.to('cpu')
        PINN_test.lb = PINN_test.lb.to('cpu')

        y_pred = PINN_test.forward(x_test_tensor)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred
    
    def test_loss(self,x_test_tensor,y_true):
        y_pred = self.test(x_test_tensor)


        test_mse = np.mean(np.square(y_pred.reshape(-1,1) - y_true.reshape(-1,1)))
        test_re = np.linalg.norm(y_pred.reshape(-1,1) - y_true.reshape(-1,1),2)/np.linalg.norm(y_true,2)
    
        return test_mse, test_re
    

    



    

