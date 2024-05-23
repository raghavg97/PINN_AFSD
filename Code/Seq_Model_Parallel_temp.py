 class Sequentialmodel_T(nn.Module):
    def __init__(self,layers,lb_xyz,ub_xyz,device):
        super().__init__() #call __init__ from parent class

        print(device)
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.layers = layers
        
        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

        self.beta = Parameter(torch.ones((50,len(layers)-2)))
        self.beta.requiresGrad = True
            
        self.lb_xyz =  torch.from_numpy(lb_xyz).float().to(device)    
        self.ub_xyz =  torch.from_numpy(ub_xyz).float().to(device)

    def forward(self,xyz):
        if torch.is_tensor(xyz) != True:
            xyz = torch.from_numpy(xyz)


        #preprocessing input
        mp = (self.lb_xyz+self.ub_xyz)/2
        xyz = 2*(xyz - mp)/(self.ub_xyz - self.lb_xyz)

        #convert to float
        a = xyz.float()

        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            # a = self.activation(z)
            z1 = self.activation(z)
            a = z1 + self.beta[:,i]*z*z1


        a = self.linears[-1](a)    
            
        return a