from model import Model, Architecture

import torch.nn as nn

class MLP(Model):
    def __init__(self, arch: Architecture):
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
        return self.model(x)
