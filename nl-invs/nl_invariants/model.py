
import math
import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class Orthogonal(nn.Module):
    
    def __init__(self, n: int, bias: bool = True) -> None:
        super().__init__()
        self.q_params = nn.Parameter(torch.Tensor(n, n))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self) -> None:
        
        # Init rotation parameters
        nn.init.uniform_(self.q_params, -math.pi, math.pi)
        
        self.q_params[:] = torch.triu(self.q_params, diagonal=1)
        
        # Init bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.q_params)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    @property
    def log_rotation(self):
        triu = torch.triu(self.q_params, diagonal=1)
        return triu - triu.T
    
    @property
    def q(self):
        return torch.matrix_exp(self.log_rotation)
    
    def log_det(self, x):
        return torch.full(x.shape[0], 1.0)
    
    def forward(self, x):
        x = nn.functional.linear(x, self.q, self.bias)
        return x
    
    def reverse(self, x):
        if self.bias is not None:
            x = x - self.bias
        x = nn.functional.linear(x, self.q.T)
        return x


class ConditionalMLP(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, channel_mults: list) -> None:
        super().__init__()
        n = max(in_features, out_features)
        self.pre_layers = nn.Sequential(
            nn.Linear(in_features, n*4),
            nn.Sigmoid(),
            nn.Linear(n*4, n*4),
        )

        self.post_layers = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(n*4, n*4),
            nn.Sigmoid(),
            nn.Linear(n*4, out_features),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n*4, n*4),
        )
    
    def forward(self, x, condition):
        x = self.pre_layers(x)
        condition = self.emb_layers(condition)

        x = x + condition
        x = self.post_layers(x)

        return x
    

class ConditionalCouplingLayer(nn.Module):
    
    def __init__(self, dim: int, channel_mults: list) -> None:
        super().__init__()
        
        self.transformed_features = dim//2
        self.num_params = dim - self.transformed_features
        self.mlp = ConditionalMLP(self.num_params, self.transformed_features, channel_mults)
    
    def forward(self, x, condition):
        x1, x2 = x[:, :self.transformed_features], x[:, self.transformed_features:]
        x1 = x1 + self.mlp(x2, condition)
        return torch.cat([x1, x2], dim=1)
    
    def reverse(self, x, condition):
        x1, x2 = x[:, :self.transformed_features], x[:, self.transformed_features:]
        x1 = x1 - self.mlp(x2, condition)
        return torch.cat([x1, x2], dim=1)


class ConditionalVolumePreservingNet(nn.Module):
    
    def __init__(self, dim: int, num_layers: int = 4, num_classes = 2, channel_mults = [1, 1, 1, 1]) -> None:
        super().__init__()
        layers = [i for _ in range(num_layers) for i in (Orthogonal(dim), ConditionalCouplingLayer(dim, channel_mults))] + [Orthogonal(dim)]
        self.layers = nn.Sequential(*layers)

        self.class_embedding = nn.Embedding(num_classes, (dim // 2 + dim % 2) * 4)
    
    def forward(self, x, condition):
        condition = self.class_embedding(condition)

        for layer in self.layers:
            x = layer(x, condition) if layer.__class__.__name__ == 'ConditionalCouplingLayer' else layer(x)

        return x
    
    def reverse(self, x, condition):
        condition = self.class_embedding(condition)

        for layer in reversed(self.layers):
            x = layer.reverse(x, condition) if layer.__class__.__name__ == 'ConditionalCouplingLayer' else layer.reverse(x)
        return x


def build_model(dim: int, num_layers: int, num_classes: int, channel_mults: list):
    model = ConditionalVolumePreservingNet(dim, num_layers, num_classes, channel_mults)
    
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("Trainable params: %d", num_of_parameters)

    return model