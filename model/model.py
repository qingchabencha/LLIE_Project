import torch.nn


class Model(torch.nn.Module):
    """
    A class for the model.
    """

    def __init__(self):
        """
        Initialize the model with the given model and device.

        Args:
            model (torch.nn.Module): The model to be used.
            device (str): The device to be used (e.g., 'cpu', 'cuda').
        """
        super(Model, self).__init__()

    
    def forward(self, low, degree_of_brightness=-1):
        """
        Forward pass of the model.

        Args:
            low (torch.Tensor): The low-light image.
            degree_of_brightness (int, optional): The degree of brightness adjustment. Defaults to -1.

        """
        return low