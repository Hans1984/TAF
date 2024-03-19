import torch
from torch import nn
import numpy as np
import math 


class SineLayer(nn.Module):

    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 

        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


class MyTanh(nn.Module):
    def __init__(self):
        super(MyTanh, self).__init__()

    def forward(self, x):
        # Apply the tanh activation function
        x = torch.tanh(2.0*x)
        return x


class TMO(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 1
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.LeakyReLU(inplace=True))#(nn.ReLU(inplace=True))

        self.net.append(nn.Linear(hidden_features, out_features)) 
        self.net.append(MyTanh())    
        self.net = nn.Sequential(*self.net)
        

        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.Tensor([0.5]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output


