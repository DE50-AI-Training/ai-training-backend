import torch
import torch.nn as nn
import torchprofile

from trainer.architectures import Architecture

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
        torch.save(self.state_dict(), path)
        self.arch.set_model_weights_path(path)
        self.arch.save(path.replace('.pth', '_arch.json'))

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def size(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def compute(self, input_shape: tuple) -> int:
        self.eval()
        dummy_input = torch.randn(input_shape)
        return torchprofile.profile_macs(self.cpu(), dummy_input)
