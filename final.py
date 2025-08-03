# final.py
import sys, os, importlib
import torch

# 0) make sure your repo root is on the path
sys.path.insert(0, os.getcwd())

# 1) import and reload to bust any stale caches
import megatron.lora
importlib.reload(megatron.lora)

# 2) show exactly where Python loaded it from
print(">>>> LoRALinear loaded from:", megatron.lora.__file__)

from megatron.lora import LoRALinear

# 3) fix the RNG so every run is identical
torch.manual_seed(0)

# 4) build the module & parameters
in_dim, out_dim, r, alpha = 16, 16, 8, 8
m = LoRALinear(in_dim, out_dim, r=r, alpha=alpha, dropout_p=0.0)
m.reset_parameters()

# 5) give A and B nonzero values
with torch.no_grad():
    torch.manual_seed(1)
    m.A.uniform_(-0.1, 0.1)
    m.B.uniform_(-0.1, 0.1)

# 6) random input
x = torch.randn(4, in_dim)

# 7) adapter output
y1 = m(x)

# 8) reference low‑rank output
W = m.A @ m.B                          # shape (in_dim, out_dim)
y2 = torch.matmul(x, W) * (alpha / r)

# 9) compare, but if it fails, report the maximum error
try:
    torch.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-6)
    print("✅ final.py test passed!")
except AssertionError as e:
    diff = (y1 - y2).abs().max().item()
    print(f"final.py test FAILED!  max |y1−y2| = {diff:.3e}")
    raise
