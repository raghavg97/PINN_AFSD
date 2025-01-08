import numpy as np
from smt.sampling_methods import LHS
import torch
import time
import torch.nn as nn  
from torch.nn.parameter import Parameter
import torch.autograd as autograd   



class coPINN_1D_Solver():
    def __init__(self,coPINN_constants,pde_related_funcs, problem_constants,
                 N_x,t_steps):
        
        
        t_end = problem_constants['Max_time']
        L = problem_constants['L']

        self.initial_TC = pde_related_funcs["initial_condition"]
        self.boundary_TC = pde_related_funcs["boundary_conditions"]

        self.x_discrete = np.linspace(0,L,N_x).reshape(-1,1)
        self.t_discrete = np.linspace(0,t_end,t_steps).reshape(-1,1)

        X,T = np.meshgrid(self.x_discrete,self.t_discrete)

        X = X.flatten('F').reshape(-1,1)
        T = T.flatten('F').reshape(-1,1)
        
        xt = np.hstack((X,T))

        self.lb_xt = xt[0]
        self.ub_xt = xt[-1]

        coPINN_constants['lb_xt'] = self.lb_xt
        coPINN_constants['ub_xt'] = self.ub_xt

        self.device = coPINN_constants['device']

        coPINN_constants['xt_test_tensor'] = torch.from_numpy(xt).float().to(self.device)

        problem_constants['forcing_func_torch'] = pde_related_funcs["forcing_function_torch"]
        

        self.N_T = coPINN_constants['N_T']
        self.N_f = coPINN_constants['N_f']

        self.coPINN = Sequentialmodel(coPINN_constants,problem_constants)

        self.optimizer_algo = coPINN_constants["optimizer_algo"]
        if(self.optimizer_algo=="LBFGS"):
            self.optimizer = torch.optim.LBFGS(self.coPINN.parameters(), lr=1, 
                              max_iter = 20, 
                              max_eval = 30, 
                              tolerance_grad = 1e-08, 
                              tolerance_change = 1e-08, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
            
        elif(self.optimizer_algo=="Adam"):
            self.optimizer = torch.optim.Adam(self.coPINN.parameters(), lr=0.008)
        else:
            raise Exception("Optimizer Algo must be LBFGS or Adam") 

        self.coPINN.to(self.device)
        print(self.coPINN)

        self.max_iter = coPINN_constants["max_iter"]

    def trainingdata(self,N_T,N_f,seed):
        '''Boundary Conditions''' 
        x = self.x_discrete
        t = self.t_discrete
        T1,T2,C1,C2 = self.boundary_TC(x)

        np.random.seed(seed)
        
        x_l = x[0]*np.ones((N_T,1)) 
        t_l = np.random.uniform(t[0],t[-1],(N_T,1))
        xt_l = np.hstack((x_l,t_l))
        T_l = T1*np.ones((N_T,1))
        C_l = C1*np.ones((N_T,1))
        
        x_r = x[-1]*np.ones((N_T,1))
        t_r = np.random.uniform(t[0],t[-1],(N_T,1))
        xt_r = np.hstack((x_r,t_r))
        T_r = T2*np.ones((N_T,1))
        C_r = C2*np.ones((N_T,1))
        
        
        x_0 = np.random.uniform(x[0],x[-1],(N_T,1))
        t_0 = t[0]*np.ones((N_T,1))
        xt_0 = np.hstack((x_0,t_0))
        T_init,C_init = self.initial_TC(x_0)
        T_0 = T_init
        C_0 = C_init
        
        xt_BC = np.vstack((xt_l,xt_r,xt_0)) #choose indices from  set 'idx' (x,t)
        T_BC = np.vstack((T_l,T_r,T_0))
        C_BC = np.vstack((C_l,C_r,C_0))
        
        TC_BC = np.hstack((T_BC,C_BC))
        
        '''Collocation Points'''

        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        x01 = np.array([[0.0,1.0],[0.0,1.0]])
        sampling = LHS(xlimits=x01,random_state =seed)
        samples = sampling(N_f)
        
        xt_coll = self.lb_xt + (self.ub_xt - self.lb_xt)*samples
        
        xt_coll = np.vstack((xt_coll, xt_BC)) # append training points to collocation points 

        return xt_coll, xt_BC, TC_BC
    
    def train_step_lbfgs(self,xy_BC,uT_BC,xy_coll,f_hat):

        def closure():
            self.optimizer.zero_grad()
            loss = self.coPINN.loss(xy_BC,uT_BC,xy_coll,f_hat)
            loss.backward()
            #print(loss.cpu().detach().numpy())
            return loss

        self.optimizer.step(closure)
    
    def train_step_adam(self,xy_BC,uT_BC,xy_coll,f_hat):

     
        self.optimizer.zero_grad()
        loss = self.coPINN.loss(xy_BC,uT_BC,xy_coll,f_hat)
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
        
        xy_coll_np_array, xy_BC_np_array, uT_BC_np_array = self.trainingdata(self.N_T,self.N_f,rep*22)
            
        xy_coll = torch.from_numpy(xy_coll_np_array).float().to(self.device)
        xy_BC = torch.from_numpy(xy_BC_np_array).float().to(self.device)
        uT_BC = torch.from_numpy(uT_BC_np_array).float().to(self.device)
            
        f_hat = torch.zeros(xy_coll.shape[0],1).to(self.device)
        

        for i in range(self.max_iter):
            if(self.optimizer_algo=="LBFGS"):
                self.train_step_lbfgs(xy_BC,uT_BC,xy_coll,f_hat)          
            else:
                self.train_step_adam(xy_BC,uT_BC,xy_coll,f_hat)
        
            
            # self.train_step(xy_BC,uT_BC,xy_coll,f_hat,i)
            loss_np = self.coPINN.loss(xy_BC,uT_BC,xy_coll,f_hat).cpu().detach().numpy()
    
        
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

        self.lb_xt = coPINN_constants['lb_xt']
        self.ub_xt = coPINN_constants['ub_xt']

        self.xt_test_tensor = coPINN_constants['xt_test_tensor']

        self.device = coPINN_constants['device']
        self.layers1 = layers1
        self.layers2 = layers2

        fvm_results = coPINN_constants['fvm_results']

        self.T_FVM = fvm_results['T_FVM']
        self.T_FVM_norm = fvm_results['T_FVM_norm']
        self.C_FVM = fvm_results['C_FVM']
        self.C_FVM_norm = fvm_results['C_FVM_norm']
            
    'foward pass'
    def forward1(self,xy):
        if torch.is_tensor(xy) != True:         
            xy = torch.from_numpy(xy)                
        
        ubxy = torch.from_numpy(self.ub_xt).float().to(self.device)
        lbxy = torch.from_numpy(self.lb_xt).float().to(self.device)
    
                      
        #preprocessing input 
        xy = 2.0*(xy - lbxy)/(ubxy - lbxy) - 1.0
        
        #convert to float
        a = xy.float()
        
        for i in range(len(self.layers1)-2):
            z = self.linears1[i](a)
            z1 =self.activation(z)
            a = z1 + self.beta1[:,i]*z*z1
       
            
        a = self.linears1[-1](a) 
         
        return a
    
    def forward2(self,xy):
        if torch.is_tensor(xy) != True:         
            xy = torch.from_numpy(xy)                
        
        ubxy = torch.from_numpy(self.ub_xt).float().to(self.device)
        lbxy = torch.from_numpy(self.lb_xt).float().to(self.device)
    
                      
        #preprocessing input 
        xy = 2.0*(xy - lbxy)/(ubxy - lbxy) - 1.0
        
        #convert to float
        a = xy.float()
        
        for i in range(len(self.layers2)-2):
            z = self.linears2[i](a)
            z1 =self.activation(z)
            a = z1 + self.beta2[:,i]*z*z1
       
            
        a = self.linears2[-1](a) 
         
        return a
                        
    def loss_BC(self,xy,TC):
        
        # uT_pred = self.forward(xy)
        T_pred = self.forward1(xy)
        C_pred = self.forward2(xy)
     
        
        loss_bc_T = self.loss_function(T_pred, TC[:,0].reshape(-1,1))
        loss_bc_C = self.loss_function(C_pred, TC[:,1].reshape(-1,1))
        
        # print(loss_bc_T.cpu().detach().numpy()/loss_bc_u.cpu().detach().numpy())
                
        return loss_bc_T + loss_bc_C
    
    def loss_NBC(self,
    

    def residuals_f1f2(self, xy_coll):
        
        g = xy_coll.clone()             
        g.requires_grad = True
        # uT = self.forward(g) 
        
        T = self.forward1(g)
        C = self.forward2(g)
        # _times_T = u.detach()*T
        
        T_x_t = autograd.grad(T,g,torch.ones([xy_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        T_xx_tt = autograd.grad(T_x_t,g,torch.ones(xy_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]

        # dT_dx = T_x_t[:,[0]]
        dT_dt = T_x_t[:,[1]]
        d2T_dx2 = T_xx_tt[:,[0]]
        
        C_x_t = autograd.grad(C,g,torch.ones([xy_coll.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True,allow_unused = True)[0]
        C_xx_tt = autograd.grad(C_x_t,g,torch.ones(xy_coll.shape).to(self.device), create_graph=True,allow_unused = True)[0]
        
        # dC_dx = C_x_t[:,[0]]
        dC_dt = C_x_t[:,[1]]
        d2C_dx2 = C_xx_tt[:,[0]]

        # print(dT_dt.shape)
        # print(forcing_func_torch(g[:,0].reshape(-1,1)).shape)

        alpha = self.problem_constants['alpha']
        beta = self.problem_constants['beta']
        gamma = self.problem_constants['gamma']
        D = self.problem_constants['D']
        forcing_func_torch = self.problem_constants["forcing_func_torch"]

      
        f1 = dT_dt - alpha*d2T_dx2 - beta*C - forcing_func_torch(g[:,0].reshape(-1,1))
        # u = u.detach()    

        
       
        f2 = dC_dt - D*d2C_dx2 - gamma*T

        return f1,f2
        
    
    def loss_PDE(self, xy_coll, f_hat):
       
        f1,f2 = self.residuals_f1f2(xy_coll)
        
        loss_f1 = self.loss_function(f1,f_hat)
        loss_f2 = self.loss_function(f2,f_hat)

        
        
        # print(loss_f2.cpu().detach().numpy()/loss_f1.cpu().detach().numpy())
        # w = loss_f1.cpu().detach().numpy()/loss_f2.cpu().detach().numpy()
                
        return loss_f1 + loss_f2
    
    def loss(self,xy_BC,TC_BC,xy_coll,f_hat):

        loss_BC = self.loss_BC(xy_BC,TC_BC)
        loss_f = self.loss_PDE(xy_coll,f_hat)
        
        loss_val = loss_BC + loss_f
        
        return loss_val
         
    'test neural network'
    def test(self):
        T = self.forward1(self.xt_test_tensor)
        C = self.forward2(self.xt_test_tensor)
   
        return T.cpu().detach().numpy(), C.cpu().detach().numpy()

    def test_deviation(self):
        T_pred,C_pred = self.test()
        # u_pred = uT_pred[:,0]
        # T_pred = uT_pred[:,1]
        
        # test_mse = np.mean(np.square(u_pred.reshape(-1,1) - u_true.reshape(-1,1)))
        test_re_T = np.linalg.norm(T_pred.reshape(-1,1) - self.T_FVM.reshape(-1,1),2)/self.T_FVM_norm
        test_re_C = np.linalg.norm(C_pred.reshape(-1,1) - self.C_FVM.reshape(-1,1),2)/self.C_FVM_norm
        
        
        return test_re_T, test_re_C 

