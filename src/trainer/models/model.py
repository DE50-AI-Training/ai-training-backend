import torch
import torch.nn as nn
import torchprofile
import os

from trainer.architectures import Architecture

class Model(nn.Module):
    """
    Base class for defining neural network models.
    This class provides a structure for defining various models and includes methods
    for handling activation functions, saving/loading model weights, and computing model size.
    It ensures that the model is initialized with a valid architecture.
    Attributes:
        arch (Architecture): An instance of the Architecture class defining the model architecture.
    """

    def __init__(self, arch: Architecture):
        """
        Initializes the model with the provided architecture.
        :param arch: An instance of the Architecture class defining the model architecture.
        """

        super().__init__()
        self.arch = arch

    def forward(self, x):
        """
        Defines the forward pass of the model.
        :param x: Input tensor to the model.
        :return: Output tensor after passing through the model.
        """
        # This method should be implemented in subclasses
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_activation(self) -> None | nn.ReLU | nn.Sigmoid | nn.Tanh:
        """
        Returns the activation function based on the architecture configuration.
        :return: An instance of the activation function or None if no activation is specified.
        """

        if self.arch.activation is None:
            return None
        if self.arch.activation == "relu":
            return nn.ReLU()
        elif self.arch.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.arch.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {self.arch.activation}")
    
    def save(self, path: str) -> None:
        """
        Saves the model's state dictionary and architecture configuration to a file.
        :param path: Path to the file where the model weights and architecture will be saved.
        """

        # To be implemented with the DB
        torch.save(self.state_dict(), path)
        self.arch.set_model_weights_path(path)
        json_path = os.path.splitext(path)[0] + '.json'
        self.arch.save(json_path)

    def load(self, path: str) -> None:
        """
        Loads the model's state dictionary from a file.
        :param path: Path to the file from which the model weights and architecture will be loaded.
        """

        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def size(self) -> int:
        """
        Computes the size of the model in terms of the number of parameters.
        :return: Total number of parameters in the model.
        """

        return sum(p.numel() for p in self.parameters())
    
    def compute(self, input_shape: tuple) -> int:
        """
        Computes the number of multiply-accumulate operations (MACs) for the model given an input shape.
        :param input_shape: Shape of the input tensor (excluding batch size).
        :return: Total number of MACs for the model.
        """
        
        self.eval()
        dummy_input = torch.randn(input_shape)
        return torchprofile.profile_macs(self.cpu(), dummy_input)
