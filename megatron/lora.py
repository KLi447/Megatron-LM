import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 1.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        # precompute scaling factor
        self.scaling = self.alpha / self.r
        # low-rank factors
        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))
        # optional dropout
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        W_delta = (self.A @ self.B) * self.scaling
        # apply update
        return torch.matmul(x, W_delta)
