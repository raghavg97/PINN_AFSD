import numpy as np
import torch
import torch.nn as nn  
from torch.nn.parameter import Parameter
import torch.autograd as autograd 

class NN_interpolator():

    def __init__(self,x,t,values_matrix,NN_constants):
        super().__init__() #call __init__ from parent class 
              
        layers = NN_constants['layers']
        device = NN_constants['device']
        max_iter = NN_constants['max_iter']

        self.NN = Sequentialmodel(layers)
        self.NN.to(device)

        self.device = device

        self.max_iter = max_iter

        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=0.008)

        xt = self.meshgrid_tensor(x,t)

        out = values_matrix.reshape(-1,1,order = 'F')
        out = torch.from_numpy(out).float().to(device)

        self.train(xt,out)

    def meshgrid_tensor(self,x_1D,t_1D):
        x,t = np.meshgrid(x_1D,t_1D)

        x = x.flatten('F').reshape(-1,1)
        t = t.flatten('F').reshape(-1,1)
        
        xt_test= np.hstack((x,t))

        xt_test_tensor = torch.from_numpy(xt_test).float().to(self.device)

        return xt_test_tensor


    def train(self,xt,out):
    
        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            loss = self.NN.loss(xt,out)
            loss.backward()
            self.optimizer.step()

        print("Loss after (interpolation) training: %.2f"%loss.cpu().detach().numpy())



    def interpolate(self,x_test,t_test):
        xt_test_tensor = self.meshgrid_tensor(x_test,t_test)

        return self.predict(xt_test_tensor)

    def predict(self,xt_test):
        return self.NN.forward(xt_test).cpu().detach().numpy()
    
    def derivatives(self,x_test,t_test):
        xt_test_tensor = self.meshgrid_tensor(x_test,t_test)
    
        g = xt_test_tensor.clone()             
        g.requires_grad = True
        # uT = self.forward(g) 
        
        out = self.NN.forward(g)
        # _times_T = u.detach()*T
        
        out_x_t = autograd.grad(out,g,torch.ones([xt_test_tensor.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        out_xx_tt = autograd.grad(out_x_t,g,torch.ones(xt_test_tensor.shape).to(self.device), create_graph=True,allow_unused = True)[0]

        return out_x_t.cpu().detach().numpy(),out_xx_tt.cpu().detach().numpy()

class Sequentialmodel(nn.Module):
    
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        self.layers = layers

        'activation function'
        self.activation = nn.Tanh()
     
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) 
                                      for i in range(len(layers)-1)])
        
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data) 
    
            
    'foward pass'
    def forward(self,xy):
        if torch.is_tensor(xy) != True:         
            xy = torch.from_numpy(xy)                
        
        # ubxy = torch.from_numpy(self.ub_xt).float().to(self.device)
        # lbxy = torch.from_numpy(self.lb_xt).float().to(self.device)

                      
        #preprocessing input 
        # xy = 2.0*(xy - lbxy)/(ubxy - lbxy) - 1.0
        
        #convert to float
        a = xy.float()
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a =self.activation(z)
            
        a = self.linears[-1](a) 
         
        return a
    
    def loss(self,xt,outs):
        pred = self.forward(xt)
 
        loss = self.loss_function(pred, outs)
    
        return loss
    
    def forward_np(self,xy):
        out = self.forward(xy).cpu().detach().numpy()
        return out

    


        
