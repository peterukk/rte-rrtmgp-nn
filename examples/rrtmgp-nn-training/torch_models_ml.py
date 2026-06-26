import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def _activation_from_name(name):
    name = str(name).lower()
    if name == "linear":
        return lambda x: x
    if name == "relu":
        return torch.relu
    if name == "softsign":
        return torch.nn.functional.softsign
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "elu":
        return torch.nn.functional.elu
    if name == "softplus":
        return torch.nn.functional.softplus
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear(linear_layer, kernel_init):
    kernel_init = str(kernel_init).lower()
    if kernel_init in {"glorot_uniform", "xavier_uniform"}:
        nn.init.xavier_uniform_(linear_layer.weight)
    elif kernel_init in {"he_uniform", "kaiming_uniform"}:
        nn.init.kaiming_uniform_(linear_layer.weight, a=np.sqrt(5))
    elif kernel_init == "lecun_uniform":
        nn.init.kaiming_uniform_(linear_layer.weight, a=np.sqrt(3))
    else:
        # Reasonable default.
        nn.init.xavier_uniform_(linear_layer.weight)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


# class GasOpticsMLP(nn.Module):
#     def __init__(self, nx, ny, neurons=(40, 40), activ=("softsign", "softsign", "linear"), kernel_init="glorot_uniform"):
#         super().__init__()

#         self.mlp1 = nn.Linear(nx, neurons[0])
#         self.mlp2 = nn.Linear(neurons[0], neurons[1])
#         self.mlp3 = nn.Linear(neurons[1], ny)

#         _init_linear(self.mlp1, kernel_init)
#         _init_linear(self.mlp2, kernel_init)
#         _init_linear(self.mlp3, kernel_init)

#         self.softsign = nn.Softsign()

#     @torch.compile
#     def forward(self, x):
#         x = self.softsign(self.mlp1(x))
#         x = self.softsign(self.mlp2(x))
#         x = self.mlp3(x)
#         return x

# ---------------------------------

class GasOpticsMLP(nn.Module):
    def __init__(self, nx, ny, 
                 neurons=(40, 40), 
                 activ=("softsign", "softsign", "linear"), 
                 kernel_init="glorot_uniform"):
        super().__init__()

        neurons = list(neurons)
        activ = list(activ)
        if len(activ) != len(neurons) + 1:
            raise ValueError("Number of activations must be number of hidden layers + 1")

        self.nx = int(nx)
        self.ny = int(ny)
        self.neurons = neurons
        self.activation_names = activ
        self.kernel_init = kernel_init

        self.linear_layers = nn.ModuleList()
        prev = self.nx
        for i, units in enumerate(neurons):
            layer = nn.Linear(prev, int(units))
            _init_linear(layer, kernel_init)
            self.linear_layers.append(layer)
            prev = int(units)

        out_layer = nn.Linear(prev, self.ny)
        _init_linear(out_layer, kernel_init)
        self.linear_layers.append(out_layer)

        self.activation_fns = [_activation_from_name(a) for a in activ]

    @torch.compile
    def forward(self, x):
        for i, layer in enumerate(self.linear_layers[:-1]):
            x = layer(x)
            x = self.activation_fns[i](x)
        x = self.linear_layers[-1](x)
        x = self.activation_fns[-1](x)
        return x
    
class GasOpticsSwBothMLP(nn.Module):
    """Wrapper that nests separate SW absorption and SW rayleigh models.

    The forward pass concatenates outputs so the training target is a single
    tensor with absorption outputs followed by Rayleigh outputs.
    """

    def __init__(self, abs_model, ray_model):
        super().__init__()
        self.abs_model = abs_model
        self.ray_model = ray_model
        self.activation_names = list(getattr(abs_model, "activation_names", [])) + list(
            getattr(ray_model, "activation_names", [])
        )

    def forward(self, x):
        y_abs = self.abs_model(x)
        y_ray = self.ray_model(x)
        return torch.cat((y_abs, y_ray), dim=-1)