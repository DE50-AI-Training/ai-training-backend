import json

ALL_ARCHS = ["MLP", "CNN", "ResNet"]
ALL_ACTIVATIONS = ["relu", "sigmoid", "tanh"]


class Architecture:
    """Base class for defining neural network architectures.
    This class provides a structure for defining various architectures
    and includes methods for setting model weights path and saving the architecture configuration.
    It ensures that the architecture type, input size, and output size are specified.
    Attributes:
        arch_info (dict): Dictionary containing architecture information.
        model_weights_path (str): Path to the model weights file.
    """

    def __init__(self, arch_info: dict) -> None:
        """
        Initializes the architecture with the provided architecture information.
        :param arch_info: Dictionary containing architecture information including:
                          - architecture: Type of architecture (e.g., "MLP", "CNN", "ResNet").
                          - output_size: Number of output classes or size.
                          - input_size: Size of the input features.
        """

        self.arch_info = arch_info
        assert "architecture" in arch_info, "Architecture type is required"
        assert arch_info["architecture"] in ALL_ARCHS, "Unsupported architecture type"

        assert "output_size" in arch_info, "Output size is required"
        assert "input_size" in arch_info, "Input size is required"

        self.architecture = arch_info["architecture"]
        self.output_size = arch_info["output_size"]
        self.input_size = arch_info["input_size"]

    def set_model_weights_path(self, path: str) -> None:
        """
        Sets the path for saving model weights.
        :param path: Path to the model weights file.
        """

        self.model_weights_path = path

    def get_model_weights_path(self) -> str:
        """
        Returns the path where model weights are saved.
        :return: Path to the model weights file.
        """

        return self.model_weights_path

    def save(self, path: str) -> None:
        """
        Saves the architecture configuration to a JSON file.
        :param path: Path to the JSON file where the architecture configuration will be saved.
        """
        
        # Deletion of arch_info from __dict__ to avoid circular reference
        del self.__dict__["arch_info"]
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
