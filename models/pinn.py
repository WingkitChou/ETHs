import torch
import torch.nn as nn
import torch.nn.functional as F
from .ffn import MLP

class PINN(nn.Module):
    """
    Physics-Informed Neural Network model
    
    Parameters:
    - input_size: input dimension, typically includes time and spatial coordinates
    - output_size: output dimension, typically 1 (solution function)
    - hidden_size: number of neurons in each hidden layer
    - num_layers: number of hidden layers
    - activation: activation function, default is "tanh" (recommended for PINNs)
    """
    def __init__(self, 
                 input_size,
                 output_size=1,
                 hidden_size=64,
                 num_layers=4,
                 activation="tanh"):
        super().__init__()
        
        # Use MLP as the base network
        self.network = MLP(input_size, output_size, hidden_size, num_layers, activation=activation)
        
        # Store model information
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        
    def reset_parameters(self):
        """Reset network parameters"""
        self.network.reset_parameters()
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: input tensor of shape [batch_size, input_size]
        
        Returns:
        - output tensor of shape [batch_size, output_size]
        """
        return self.network(x)