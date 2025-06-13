from trainer.models import Model, Architecture

import torch.nn as nn

class MLP(Model):
    """
    Multi-Layer Perceptron (MLP) model implementation.
    Inherits from the base Model class and constructs a sequential model
    based on the architecture defined in the MLPArchitecture class.
    This model is designed for tasks where the input is a fixed-size vector
    and the output is a vector of class probabilities or regression values.
    """

    def __init__(self, arch: Architecture):
        """
        Initializes the MLP model with the provided architecture.
        :param arch: An instance of the MLPArchitecture class defining the model architecture.
        """

        super().__init__(arch)
        assert self.arch.architecture == 'MLP', "Architecture must be MLP"

        self.model = nn.Sequential()
        in_features = self.arch.input_size
        activation = self.get_activation()

        for i, layer in enumerate(self.arch.layers[1:]):
            self.model.add_module(f"layer_{i}", nn.Linear(in_features, layer))
            if i < len(self.arch.layers) - 2:
                self.model.add_module(f"relu_{i}", activation)
            in_features = layer

    def forward(self, x):
        """
        Defines the forward pass of the MLP model.
        :param x: Input tensor to the model.
        :return: Output tensor after passing through the model.
        """
        
        return self.model(x)