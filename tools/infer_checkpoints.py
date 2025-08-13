import os
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")       # satisfies Megatron assert
os.environ.setdefault("MEGATRON_NO_GRAD_ACCUM_FUSION", "1")     # avoids Apex fused ext on build

import argparse, torch, json
import numpy as np

import sys
sys.path.append(os.path.abspath("."))

from megatron.training import get_model, get_args
from megatron.training.global_vars import set_args
from megatron.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.text_generation.forward_step import forward_step as generate_forward  # if present in your tree
from megatron.text_generation_utils import generate_and_post_process  # legacy util
from megatron.training import get_tokenizer

from megatron.lora import (
    freeze_non_lora_params,
    report_trainable_params,
    layerwise_weight_norms,
)

def set_determinism(seed=17):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', required=True)                # checkpoint root (same as training --load/--save)
    ap.add_argument('--ckpt-iters', nargs='+', required=True, type=int)  # e.g., 0 500 1000
    ap.add_argument('--prompt', default="The quick brown fox")
    ap.add_argument('--out', default='inference_outputs.jsonl')
    # standard Megatron args needed to build model/tokenizer (match training flags!)
    ap.add_argument('--tensor-model-parallel-size', type=int, required=True)
    ap.add_argument('--pipeline-model-parallel-size', type=int, required=True)
    ap.add_argument('--tokenizer-type', default='GPT2BPETokenizer')
    ap.add_argument('--vocab-file', required=True)
    ap.add_argument('--merge-file', required=True)
    args, unknown = ap.parse_known_args()

    # Initialize runtime (single node assumed; will work under srun as well)
    set_determinism(1234)
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    initialize_megatron(extra_args_provider=lambda: None,
                        args_defaults={'no_load_optim': True,
                                       'no_load_rng': True})

    meg_args = get_args()
    tok = get_tokenizer()

    results = []
    for it in args.ckpt_iters:
        # Megatron will load latest by default; force a specific iteration by pointing to the file
        ckpt_dir = os.path.join(args.load, f'iter_{it:07d}')
        # or for iter 0 from convert script: pass the convert dir directly
        target = ckpt_dir if os.path.isdir(ckpt_dir) else args.load

        meg_args.load = target
        # Rebuild model per iteration load to be safe
        from megatron.training.checkpointing import load_checkpoint
        model = get_model(lambda: GPTModel(config=core_transformer_config_from_args(meg_args),
                                           num_tokentypes=0,
                                           parallel_output=False))[0]
        iteration, _ = load_checkpoint(model, None, None, strictness=meg_args.dist_ckpt_strictness)

        stats = freeze_non_lora_params(model)
        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            print(f"[LoRA] freeze stats: {stats}")
            report_trainable_params(model)
            print("[Sanity] Top parameter norms (pre-inference):")
            layerwise_weight_norms(model, top_k=10)

        # ------------------ NEW: one explicit forward() with proper masks ------------------
        tok = get_tokenizer()
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        def _tokenize(prompt: str) -> torch.Tensor:
            ids = tok.tokenize(prompt) if hasattr(tok, "tokenize") else tok.encode(prompt)
            if isinstance(ids, list) and len(ids) and isinstance(ids[0], list):
                ids = ids[0]
            return torch.tensor([ids], dtype=torch.long, device=device)  # [1, S]

        def _build_masks(tokens: torch.Tensor):
            S = tokens.size(1)
            position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
            # additive causal mask: 0 keeps; -inf masks future positions
            attn = torch.zeros((1, 1, S, S), device=device, dtype=torch.float32)
            upper = torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1)
            attn.masked_fill_(upper, float("-inf"))
            return position_ids, attn  # [1,S], [1,1,S,S]

        model.eval()
        with torch.no_grad():
            _tokens = _tokenize(args.prompt)
            _pos, _mask = _build_masks(_tokens)
            _ = model(_tokens, _pos, _mask, labels=None)  # logits exist only on last PP stage

        # Greedy generation (temp=0)
        from megatron.text_generation.generation import generate_tokens_probs_and_return_on_first_stage as generate
        text = args.prompt
        output = generate(text, max_new_tokens=64, temperature=0.0, top_k=0, top_p=0.0)
        results.append({'iter': int(it), 'text': text, 'output': output})

    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(f"Wrote {len(results)} generations to {args.out}")

if __name__ == "__main__":
    main()