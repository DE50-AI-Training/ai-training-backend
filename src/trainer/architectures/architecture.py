import json

ALL_ARCHS = ['MLP', 'CNN', 'ResNet']
ALL_ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'softmax']

class Architecture:
    def __init__(self, arch_info: dict) -> None:
        self.arch_info = arch_info
        assert 'architecture' in arch_info, "Architecture type is required"
        assert arch_info['architecture'] in ALL_ARCHS, "Unsupported architecture type"

        assert 'output_size' in arch_info, "Output size is required"
        assert 'input_size' in arch_info, "Input size is required"

        self.architecture = arch_info['architecture']
        self.output_size = arch_info['output_size']
        self.input_size = arch_info['input_size']

    def set_model_weights_path(self, path: str) -> None:
        self.model_weights_path = path

    def get_model_weights_path(self) -> str:
        return self.model_weights_path

    def save(self, path: str) -> None:
        del self.__dict__['arch_info']
        json.dump(self.__dict__, open(path, 'w'), indent=4)