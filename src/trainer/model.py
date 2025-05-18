import torch
import torch.nn as nn

from trainer.architectures.architecture import Architecture
from trainer.architectures.mlp_architecture import MLPArchitecture

def create_model(model_class, archi_info, num_classes):
    architecture = MLPArchitecture(archi_info)
    model = model_class(architecture) 
    return model
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)