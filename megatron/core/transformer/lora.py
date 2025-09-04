import torch
from torch import nn
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

class LoraAdapter(nn.Module):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        self.adapter_name = adapter_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        if r > 0:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = 1.0

        self.lora_A = nn.Parameter(torch.Tensor(r, in_features))
        self.lora_B = nn.Parameter(torch.Tensor(out_features, r))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora_dropout(x)
        x = F.linear(x, self.lora_A)
        x = F.linear(x, self.lora_B)
        return x * self.scaling

class LoraLayer(nn.Module):
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.original_layer = original_layer

        if not isinstance(self.original_layer, (ColumnParallelLinear, RowParallelLinear)):
            raise TypeError(f"Lora can only be applied to ColumnParallelLinear or RowParallelLinear, but got {type(original_layer)}")

        self.in_features = self.original_layer.input_size
        self.out_features = self.original_layer.output_size

        self.original_layer.weight.requires_grad = False
        if hasattr(self.original_layer, 'bias') and self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self.adapters: MutableMapping[str, LoraAdapter] = nn.ModuleDict({})

    def add_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float = 0.0):
        if adapter_name in self.adapters:
            print(f"Warning: Adapter '{adapter_name}' already exists. Overwriting.")

        out_features_lora = self.original_layer.output_size_per_partition if isinstance(self.original_layer, ColumnParallelLinear) else self.out_features
        adapter = LoraAdapter(adapter_name, self.in_features, out_features_lora, r, lora_alpha, lora_dropout)
        self.adapters[adapter_name] = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.original_layer(x)
        if isinstance(base_result, tuple):
            output_tensor = base_result[0]
            other_outputs = base_result[1:]
        else:
            output_tensor = base_result
            other_outputs = None

        if len(self.adapters) > 0:
            lora_sum = output_tensor
            for adapter in self.adapters.values():
                lora_output = adapter(x)
                lora_sum = lora_sum + lora_output

            if other_outputs is not None:
                return (lora_sum,) + other_outputs
            else:
                return lora_sum

        return base_result

def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    adapter_name: str = "default"
):
    for name, module in model.named_children():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)) and any(target in name for target in target_modules):
            wrapped_layer = LoraLayer(module)
            wrapped_layer.add_adapter(
                adapter_name=adapter_name,
                r=r,
                lora_alpha=lora_alpha
            )
            setattr(model, name, wrapped_layer)
        else:
            apply_lora(module, target_modules, r, lora_alpha, adapter_name)

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}
