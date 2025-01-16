import numpy as np
from smt.sampling_methods import LHS
import torch
import time
import torch.nn as nn  
from torch.nn.parameter import Parameter
import torch.autograd as autograd   



class coPINN_1D_Solver():
    def __init__(self,coPINN_constants,pde_related_funcs, problem_constants,
                 N_x,):
        
        
        L = problem_constants['L']
        self.T_left = problem_constants['T_a']
        self.C_left = problem_constants['C_left']
        self.C_right = problem_constants['C_right']

        self.C_left_right = np.array([self.C_left,self.C_right]).reshape(-1,1)

        self.x = np.linspace(0,L,N_x).reshape(-1,1)

        self.lb_x = np.array(0.0)
        self.ub_x = np.array(L)

        coPINN_constants['lb_x'] = self.lb_x
        coPINN_constants['ub_x'] = self.ub_x

        self.device = coPINN_constants['device']

        coPINN_constants['x_test_tensor'] = torch.from_numpy(self.x).float().to(self.device)

        problem_constants['heat_source_func_torch'] = pde_related_funcs["heat_source_func_torch"]
        problem_constants['conc_force_func_torch'] = pde_related_funcs["conc_force_func_torch"]


        self.N_f = coPINN_constants['N_f']

        self.coPINN = Sequentialmodel(coPINN_constants,problem_constants)

        self.optimizer_algo = coPINN_constants["optimizer_algo"]
        if(self.optimizer_algo=="LBFGS"):
            self.optimizer = torch.optim.LBFGS(self.coPINN.parameters(), lr=0.75, 
                              max_iter = 30, 
                              max_eval = 30, 
                              tolerance_grad = 1e-08, 
                              tolerance_change = 1e-08, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
            
        elif(self.optimizer_algo=="Adam"):
            self.optimizer = torch.optim.Adam(self.coPINN.parameters(), lr=0.01)
        else:
            raise Exception("Optimizer Algo must be LBFGS or Adam") 

        self.coPINN.to(self.device)
        print(self.coPINN)

        self.max_iter = coPINN_constants["max_iter"]

    def trainingdata(self,N_f,seed):
        '''Boundary Conditions''' 
        x = self.x

        np.random.seed(seed)
        
        x_l = x[0].reshape(-1,1)

        
        x_r = x[-1].reshape(-1,1)
    

        '''Collocation Points'''

        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        x01 = np.array([[0.0,1.0]])
        sampling = LHS(xlimits=x01,random_state =seed)
        samples = sampling(N_f)
        
        x_coll = self.lb_x + (self.ub_x - self.lb_x)*samples

        x_coll = np.vstack((x_coll, x_l.reshape(-1,1),x_r.reshape(-1,1))) # append training points to collocation points 

        return x_coll, x_l, x_r
    
    def train_step_lbfgs(self,x_left,x_right,x_left_right,x_coll,T_left,C_left_right,f_hat,R_hat):

        def closure():
            self.optimizer.zero_grad()
            loss = self.coPINN.loss(x_left,x_right,x_left_right,x_coll,T_left,C_left_right,f_hat,R_hat)
            loss.backward()
            #print(loss.cpu().detach().numpy())
            return loss

        self.optimizer.step(closure)
    
    def train_step_adam(self,x_left,x_right,x_left_right,x_coll,T_left,C_left_right,f_hat,R_hat):

     
        self.optimizer.zero_grad()
        loss = self.coPINN.loss(x_left,x_right,x_left_right,x_coll,T_left,C_left_right,f_hat,R_hat)
        loss.backward()
        self.optimizer.step()
            #print(loss.cpu().detach().numpy())
        # return loss


    def train_model(self,rep): 
        print(rep) 
        torch.manual_seed(rep*9)
        start_time = time.time() 
        thresh_flag = 0

        test_mse_loss = []
        test_re_loss = []
        
        x_coll_np, x_l_np, x_r_np = self.trainingdata(self.N_f,rep*22)

        x_l_r_np = np.vstack([x_l_np,x_r_np]).reshape(-1,1)
        
            
        x_coll = torch.from_numpy(x_coll_np).float().to(self.device)
        x_l = torch.from_numpy(x_l_np).float().to(self.device)
        x_r = torch.from_numpy(x_r_np).float().to(self.device)
        x_l_r = torch.from_numpy(x_l_r_np).float().to(self.device)
            
        f_hat = torch.zeros(x_coll.shape[0],1).to(self.device)
        R_hat = torch.zeros(x_r.shape[0],1).to(self.device)
        T_l= torch.tensor(self.T_left).float().to(self.device)
        C_l_r = torch.tensor(self.C_left_right).float().to(self.device)

        for i in range(self.max_iter):
            if(self.optimizer_algo=="LBFGS"):
                self.train_step_lbfgs(x_l,x_r,x_l_r,x_coll,T_l,C_l_r,f_hat,R_hat)          
            else:
                self.train_step_adam(x_l,x_r,x_l_r,x_coll,T_l,C_l_r,f_hat,R_hat)
        
            
            # self.train_step(xy_BC,uT_BC,xy_coll,f_hat,i)
            loss_np = self.coPINN.loss(x_l,x_r,x_l_r,x_coll,T_l,C_l_r,f_hat,R_hat).cpu().detach().numpy()
    
        
            test_mse, test_re = self.coPINN.test_deviation()
            test_mse_loss.append(test_mse)
            test_re_loss.append(test_re)
            
            print(i,"Train Loss",loss_np,"RD T:",test_mse_loss[-1],"RD C:",test_re_loss[-1])   
            
        print('Training time: %.2f' %(time.time()-start_time))
        time_stamp = time.time()
        torch.save(self.coPINN.state_dict(),'./Saved_Models/coPINN_1D_%4d'%time_stamp+'.pt')

        return self.coPINN
        
    
#---------------------------------------------------------------------------------
#-================================================================================
#---------------------------------------------------------------------------------
#=================================================================================

class Sequentialmodel(nn.Module):
    
    def __init__(self,coPINN_constants,problem_constants):
        super().__init__() #call __init__ from parent class 
              
        layers1 = coPINN_constants['layers1']
        layers2 = coPINN_constants['layers2']

        self.problem_constants = problem_constants
        'activation function'
        self.activation = nn.Tanh()

     
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears1 = nn.ModuleList([nn.Linear(layers1[i], layers1[i+1]) for i in range(len(layers1)-1)])
        
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers1)-1):
            nn.init.xavier_normal_(self.linears1[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears1[i].bias.data)   
        
        beta_mean1 = 1.0*torch.ones((50,len(layers1)-2))
        beta_std1 = 0.1*torch.ones((50,len(layers1)-2))
        
        self.beta1 = Parameter(torch.normal(beta_mean1,beta_std1))
        self.beta1.requiresGrad = True
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears2 = nn.ModuleList([nn.Linear(layers2[i], layers2[i+1]) for i in range(len(layers2)-1)])
        
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers2)-1):
            nn.init.xavier_normal_(self.linears2[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears2[i].bias.data)   
        
        beta_mean2 = 1.0*torch.ones((50,len(layers2)-2))
        beta_std2 = 0.1*torch.ones((50,len(layers2)-2))
        
        self.beta2 = Parameter(torch.normal(beta_mean2,beta_std2))
        self.beta2.requiresGrad = True

        self.lb_x = coPINN_constants['lb_x']
        self.ub_x = coPINN_constants['ub_x']

        self.x_test_tensor = coPINN_constants['x_test_tensor']

        self.device = coPINN_constants['device']


        self.layers1 = layers1
        self.layers2 = layers2

        #Problem Constants
        self.h_c = problem_constants['h_c']

        self.alpha = self.problem_constants['alpha']
        self.beta = self.problem_constants['beta']
        
        self.T_a = self.problem_constants['T_a']
        self.C_left = self.problem_constants['C_left']
        self.C_right = self.problem_constants['C_right']


        self.heat_source_func_torch = self.problem_constants["heat_source_func_torch"]
        self.conc_force_func_torch = self.problem_constants["conc_force_func_torch"]

        fvm_results = coPINN_constants['fvm_results']

        self.T_FVM = fvm_results['T_FVM']
        self.T_FVM_norm = fvm_results['T_FVM_norm']
        self.C_FVM = fvm_results['C_FVM']
        self.C_FVM_norm = fvm_results['C_FVM_norm']


        
            
    'foward pass'
    def forward1(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        ubx = torch.from_numpy(self.ub_x).float().to(self.device)
        lbx = torch.from_numpy(self.lb_x).float().to(self.device)
    
                      
        #preprocessing input 
        x = 2.0*(x - lbx)/(ubx - lbx) - 1.0
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers1)-2):
            z = self.linears1[i](a)
            z1 =self.activation(z)
            a = z1 + self.beta1[:,i]*z*z1
       
            
        a = self.linears1[-1](a) 
         
        return a
    
    def forward2(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        ubx = torch.from_numpy(self.ub_x).float().to(self.device)
        lbx = torch.from_numpy(self.lb_x).float().to(self.device)
    
                      
        #preprocessing input 
        x = 2.0*(x - lbx)/(ubx - lbx) - 1.0
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers2)-2):
            z = self.linears2[i](a)
            z1 =self.activation(z)
            a = z1 + self.beta2[:,i]*z*z1
       
            
        a = self.linears2[-1](a) 
         
        return a
    
    def loss_BC_T(self,x,T_left): #loss function 1
        
        T_pred = self.forward1(x)
        
        loss_bc_T = self.loss_function(T_pred, T_left.reshape(-1,1))
        
        return loss_bc_T
    
    def loss_BC_C(self,x,C_left_right): #loss function 2
        
        C_pred = self.forward2(x)
        
        loss_bc_C = self.loss_function(C_pred, C_left_right.reshape(-1,1))
        
        return loss_bc_C
                        
    
    def residuals_robin_BC(self,x): #Robin BC on right
        g = x.clone()             
        g.requires_grad = True

        T = self.forward1(g)

        T_x = autograd.grad(T,g,torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]

        f = T_x - self.h_c*(T - self.T_a)

        return f

    def loss_Robin_BC(self,x,R_hat): #loss function 3
        f = self.residuals_robin_BC(x)

        loss_robin = self.loss_function(f,R_hat)

        return loss_robin 

    def residuals_f1f2(self, x_coll):
        
        g = x_coll.clone()             
        g.requires_grad = True
        # uT = self.forward(g) 
        
        T = self.forward1(g)
        C = self.forward2(g)
        # _times_T = u.detach()*T
        
        T_x = autograd.grad(T,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        T_xx = autograd.grad(T_x,g,torch.ones(x_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]

        # dT_dx = T_x_t[:,[0]]
        # dT_dt = T_x_t[:,[1]]
        d2T_dx2 = T_xx[:,[0]]
        
        C_x = autograd.grad(C,g,torch.ones([x_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        C_xx = autograd.grad(C_x,g,torch.ones(x_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]
        
        # dC_dx = C_x_t[:,[0]]
        # dC_dt = C_x_t[:,[1]]
        d2C_dx2 = C_xx[:,[0]]

        # print(dT_dt.shape)
        # print(forcing_func_torch(g[:,0].reshape(-1,1)).shape)

        f1 = self.alpha*d2T_dx2 - self.heat_source_func_torch(C)
        f2 = self.beta*d2C_dx2 - self.conc_force_func_torch(T) 
      

        return f1,f2
        
    
    def loss_PDE(self, x_coll, f_hat): #loss function 4
       
        f1,f2 = self.residuals_f1f2(x_coll)
        
        loss_f1 = self.loss_function(f1,f_hat)
        loss_f2 = self.loss_function(f2,f_hat)
        
        
        # print(loss_f2.cpu().detach().numpy()/loss_f1.cpu().detach().numpy())
        # w = loss_f1.cpu().detach().numpy()/loss_f2.cpu().detach().numpy()
                
        return loss_f1 + loss_f2
    
    def loss(self,x_left,x_right,x_left_right,x_coll,T_left,C_left_right,f_hat,R_hat): #The OVERALL loss function

        loss_f = self.loss_PDE(x_coll,f_hat)
        loss_T_BC = self.loss_BC_T(x_left,T_left)
        loss_C_BC = self.loss_BC_C(x_left_right,C_left_right)
        loss_Robin = self.loss_Robin_BC(x_right,R_hat)
        
        # print("Losses: ", loss_f.cpu().detach().numpy(),loss_T_BC.cpu().detach().numpy(),loss_C_BC.cpu().detach().numpy(),loss_Robin.cpu().detach().numpy())

        loss_val = loss_f + loss_T_BC + 5.0*loss_C_BC + loss_Robin
        
        return loss_val
         
    'test neural network'
    def test(self):
        T = self.forward1(self.x_test_tensor)
        C = self.forward2(self.x_test_tensor)
   
        return T.cpu().detach().numpy(), C.cpu().detach().numpy()

    def test_deviation(self):
        T_pred,C_pred = self.test()
        # u_pred = uT_pred[:,0]
        # T_pred = uT_pred[:,1]
        
        # test_mse = np.mean(np.square(u_pred.reshape(-1,1) - u_true.reshape(-1,1)))
        test_re_T = np.linalg.norm(T_pred.reshape(-1,1) - self.T_FVM.reshape(-1,1),2)/self.T_FVM_norm
        test_re_C = np.linalg.norm(C_pred.reshape(-1,1) - self.C_FVM.reshape(-1,1),2)/self.C_FVM_norm
        
        
        return test_re_T, test_re_C 

