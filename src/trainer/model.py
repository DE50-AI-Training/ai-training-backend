import torch
import torch.nn as nn
import torchprofile

from architectures import Architecture, MLPArchitecture
from mlp_model import MLP

class Model(nn.Module):
    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_activation(self) -> None | nn.ReLU | nn.Sigmoid | nn.Tanh | nn.Softmax:
        if self.arch.activation is None:
            return None
        if self.arch.activation == "relu":
            return nn.ReLU()
        elif self.arch.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.arch.activation == "tanh":
            return nn.Tanh()
        elif self.arch.activation == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation function: {self.arch.activation}")
    
    def save(self, path: str) -> None:
        # To be implemented with the DB
        pass
        # torch.save(self.state_dict(), path)
        # self.arch.set_model_weights_path(path)
        # self.arch.save(path.replace('.pth', '_arch.json'))

    
    def size(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def compute(self, input_shape: tuple) -> int:
        self.eval()
        dummy_input = torch.randn(input_shape)
        return torchprofile.profile_macs(self.cpu(), dummy_input)
    

def create_model(arch_dict: dict) -> Model:
    if 'architecture' not in arch_dict:
        raise ValueError("Architecture type is required")
    if arch_dict['architecture'] == 'MLP':
        return MLP(MLPArchitecture(arch_dict))
    else:
        raise ValueError(f"Unsupported architecture: {arch_dict['architecture']}. Supported architectures are Currently MLP.")
