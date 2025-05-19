from trainer.architectures import Architecture, ALL_ACTIVATIONS 

class MLPArchitecture(Architecture):
    def __init__(self, arch_info: dict) -> None:
        super().__init__(arch_info)
        assert 'layers' in arch_info, "Layers information is required"
        assert 'activation' in arch_info, "Activation function is required"
        assert arch_info['activation'] in ALL_ACTIVATIONS, "Unsupported activation function"

        self.layers = arch_info['layers']
        self.activation = arch_info['activation']

        assert len(self.layers) >= 2, "MLP must have at least input and output layers"
        assert all(isinstance(layer, int) for layer in self.layers), "All layers must be integers in MLP"

        assert self.layers[0] == self.input_size, "First layer must match input size"
        assert self.layers[-1] == self.output_size, "Last layer must match number of classes/outputs"
