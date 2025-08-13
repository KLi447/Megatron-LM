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
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        # low-rank factors
        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))
        # optional dropout
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.r > 0:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return (x @ self.A @ self.B) * self.scaling

# -------- Utilities for freezing/verification ----------

def _is_lora_param(name: str, p: torch.Tensor) -> bool:
    lname = name.lower()
    return (
        "lora" in lname
        or getattr(p, "is_lora_param", False)
        or lname.endswith(".a") or lname.endswith(".b")
        or lname.endswith(".lora_a") or lname.endswith(".lora_b")
    )

def freeze_non_lora_params(model: nn.Module):
    """
    Freeze *everything* except parameters tagged as LoRA.
    Returns a small stats dict so you can log it.
    """
    total, frozen, trainable = 0, 0, 0
    for n, p in model.named_parameters():
        total += p.numel()
        if _is_lora_param(n, p):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    return {"total": total, "frozen": frozen, "trainable": trainable}

def report_trainable_params(model: nn.Module):
    t, f = 0, 0
    for _, p in model.named_parameters():
        if p.requires_grad: t += p.numel()
        else: f += p.numel()
    pct = 100.0 * t / (t + f + 1e-9)
    print(f"[LoRA] trainable params: {t:,} | frozen params: {f:,} | % trainable: {pct:.4f}%")

@torch.no_grad()
def layerwise_weight_norms(model: nn.Module, top_k: int = 10):
    """
    Quick sanity: list top-k parameter L2 norms. If loads are broken,
    you often see many zeros or weirdly uniform tiny values.
    """
    norms = []
    for n, p in model.named_parameters():
        # skip non-floating tensors
        if p.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) and p.ndim >= 1:
            norms.append((n, p.detach().float().norm().item()))
    norms.sort(key=lambda x: -x[1])
    print("[Sanity] Top weight norms:")
    for n, v in norms[:top_k]:
        print(f"  {v:12.6f}  {n}")
    zeros = [n for n, v in norms if v == 0.0]
    if zeros:
        print(f"[Sanity] Found {len(zeros)} params with ZERO norm! (likely a bad load)")
    return norms