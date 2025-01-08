import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.nn.parameter import Parameter

from IPython import get_ipython
ipython = get_ipython()

global ub, lb, mu
# device = ipython.user_ns['device']
ub= ipython.user_ns['ub']
lb= ipython.user_ns['lb']
mu= ipython.user_ns['mu']



class Sequentialmodel_coPINN(nn.Module):
    
    def __init__(self,layers,device):
        super().__init__() #call __init__ from parent class 
              
    
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears1 = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears1[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears1[i].bias.data) 
        
        self.beta1 = Parameter(torch.ones((50,len(layers)-2)))
        self.beta1.requiresGrad = True
        
        self.linears2 = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears2[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears2[i].bias.data) 
        
        self.beta2 = Parameter(torch.ones((50,len(layers)-2)))
        self.beta2.requiresGrad = True

        self.layers = layers
        self.device = device
    
    'forward pass'
    def forward1(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float().to(self.device)
        l_b = torch.from_numpy(lb).float().to(self.device)
                      
        #preprocessing input 
        x = 2.0*(x - l_b)/(u_b - l_b) - 1.0 #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears1[i](a)
            a = self.activation(z) + self.beta1[:,i]*z*self.activation(z)
            
        a = self.linears1[-1](a) 
         
        return a
    
    def forward2(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float().to(self.device)
        l_b = torch.from_numpy(lb).float().to(self.device)
                      
        #preprocessing input 
        x = 2.0*(x - l_b)/(u_b - l_b) - 1.0 #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears2[i](a)
            a = self.activation(z) + self.beta2[:,i]*z*self.activation(z)
            
        a = self.linears2[-1](a) 
         
        return a
                        
    def loss_BC1(self,x,y):
                
        loss_bc1 = self.loss_function(self.forward1(x), y)
                
        return loss_bc1
    
    def loss_BC2(self,x,y):
                
        loss_bc2 = self.loss_function(self.forward2(x), y)
                
        return loss_bc2
    
    def loss_PDE1(self, x_coll,f_hat):
             
        g = x_coll.clone()             
        g.requires_grad = True
  
        y1 = self.forward1(g)
        y2 = self.forward2(g)

        y1_x = autograd.grad(y1,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
       
        dy1_dx = y1_x[:,[0]]
        
        f = dy1_dx - y2 
        
        loss_f = self.loss_function(f,f_hat)
                
        return loss_f
    
    def loss_PDE2(self, x_coll,f_hat):
             
        g = x_coll.clone()             
        g.requires_grad = True
  
        y1 = self.forward1(g)
        y2 = self.forward2(g)

        y2_x = autograd.grad(y2,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
       
        dy2_dx = y2_x[:,[0]]
        
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
    
    def test(self):
        y_pred = self.forward1(x_test_tensor)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred
    
    def test_loss(self):
        y_pred = self.test()
        
        # test_mse = np.mean(np.square(y_pred.reshape(-1,1) - y_true.reshape(-1,1)))
        # test_re = np.linalg.norm(y_pred.reshape(-1,1) - y_true.reshape(-1,1),2)/y_true_norm
        test_mse = 0
        test_re = 0
        
        return test_mse, test_re
    
class Sequentialmodel_vPINN(nn.Module):
    
    def __init__(self,layers,device):
        super().__init__() #call __init__ from parent class 
              
    
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data) 
        
        self.beta = Parameter(torch.ones((100,len(layers)-2)))
        self.beta.requiresGrad = True

        self.layers = layers
        self.device = device
    
    'forward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float().to(self.device)
        l_b = torch.from_numpy(lb).float().to(self.device)
                      
        #preprocessing input 
        x = 2.0*(x - l_b)/(u_b - l_b) - 1.0 #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z) #+ self.beta[:,i]*z*self.activation(z)
            
        a = self.linears[-1](a) 
         
        return a
                        
    def loss_BC1(self,x,y):
                
        loss_bc1 = self.loss_function(self.forward(x)[:,0:1], y)
                
        return loss_bc1
    
    def loss_BC2(self,x_bc2,bc2_val):
#         g = x_bc2.clone()             
#         g.requires_grad = True
#         y = self.forward(g)    
            
#         y_x = autograd.grad(y,g,torch.ones([x_bc2.shape[0], 1]).to(device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        
#         dy_dx = y_x[:,[0]]
        
#         bc2 = dy_dx
        
        loss_bc2= self.loss_function(self.forward(x_bc2)[:,1:2],bc2_val)

        return loss_bc2
    
    def loss_PDE(self, x_coll,f_hat):
             
        g = x_coll.clone()             
        g.requires_grad = True
  
        y1 = self.forward(g)[:,0:1]
        y2 = self.forward(g)[:,1:2]

        y1_x = autograd.grad(y1,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        y2_x = autograd.grad(y2,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]

        dy1_dx = y1_x[:,[0]]
        dy2_dx = y2_x[:,[0]]
        
        f1 = dy1_dx - y2
        f2 = dy2_dx - mu*(1-torch.square(y1))*y2 + y1 
        
        # f = dy2_d2x + dy_dx - u_coeff*y
        # f = dy2_d2x - mu*(1-torch.square(y))*dy_dx + y 
        
        loss_f1 = self.loss_function(f1,f_hat)
        loss_f2 = self.loss_function(f2,f_hat)
                
        return loss_f1 + loss_f2
    
    
    def loss(self,x_bc1,y_bc1,x_bc2,bc2_val,x_coll,f_hat):

        loss_bc1 = self.loss_BC1(x_bc1,y_bc1)
        loss_bc2 = self.loss_BC2(x_bc2,bc2_val)
        loss_f = self.loss_PDE(x_coll,f_hat)
        
        loss_val = loss_bc1 + loss_bc2 + loss_f
        
        return loss_val
          
    'test neural network'
    
    def test(self):
        y_pred = self.forward(x_test_tensor)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred
    
    def test_loss(self):
        # y_pred = self.test()
        y_pred= 0
        
        # test_mse = np.mean(np.square(y_pred.reshape(-1,1) - y_true.reshape(-1,1)))
        # test_re = np.linalg.norm(y_pred.reshape(-1,1) - y_true.reshape(-1,1),2)/y_true_norm
        test_mse = 0
        test_re = 0
        
        return test_mse, test_re