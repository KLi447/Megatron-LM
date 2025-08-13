# tools/convert_hf_gpt2_to_megatron.py
from pathlib import Path
import os, sys, torch, argparse
from transformers import AutoModelForCausalLM, AutoConfig

# Make repo imports work from any cwd
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def hf_to_megatron_map(hf_sd, n_layers):
    out = {}
    # Embeddings
    out['language_model.embedding.word_embeddings.weight'] = hf_sd['transformer.wte.weight']
    out['language_model.embedding.position_embeddings.weight'] = hf_sd['transformer.wpe.weight']
    # Layers
    for i in range(n_layers):
        out[f'language_model.encoder.layers.{i}.input_layernorm.weight']  = hf_sd[f'transformer.h.{i}.ln_1.weight']
        out[f'language_model.encoder.layers.{i}.input_layernorm.bias']    = hf_sd[f'transformer.h.{i}.ln_1.bias']
        out[f'language_model.encoder.layers.{i}.post_attention_layernorm.weight'] = hf_sd[f'transformer.h.{i}.ln_2.weight']
        out[f'language_model.encoder.layers.{i}.post_attention_layernorm.bias']   = hf_sd[f'transformer.h.{i}.ln_2.bias']

        # Attention (HF fused qkv -> Megatron QKV)
        Wqkv_hf = hf_sd[f'transformer.h.{i}.attn.c_attn.weight']   # [hidden, 3*hidden]
        bqkv_hf = hf_sd[f'transformer.h.{i}.attn.c_attn.bias']     # [3*hidden]
        out[f'language_model.encoder.layers.{i}.self_attention.query_key_value.weight'] = Wqkv_hf.t().contiguous()
        out[f'language_model.encoder.layers.{i}.self_attention.query_key_value.bias']   = bqkv_hf

        # Attention out proj
        W_o = hf_sd[f'transformer.h.{i}.attn.c_proj.weight']       # [hidden, hidden]
        b_o = hf_sd[f'transformer.h.{i}.attn.c_proj.bias']         # [hidden]
        out[f'language_model.encoder.layers.{i}.self_attention.dense.weight'] = W_o.t().contiguous()
        out[f'language_model.encoder.layers.{i}.self_attention.dense.bias']   = b_o

        # MLP
        W_fc   = hf_sd[f'transformer.h.{i}.mlp.c_fc.weight']       # [hidden, 4*hidden]
        b_fc   = hf_sd[f'transformer.h.{i}.mlp.c_fc.bias']         # [4*hidden]
        W_proj = hf_sd[f'transformer.h.{i}.mlp.c_proj.weight']     # [4*hidden, hidden]
        b_proj = hf_sd[f'transformer.h.{i}.mlp.c_proj.bias']       # [hidden]
        out[f'language_model.encoder.layers.{i}.mlp.dense_h_to_4h.weight'] = W_fc.t().contiguous()
        out[f'language_model.encoder.layers.{i}.mlp.dense_h_to_4h.bias']   = b_fc
        out[f'language_model.encoder.layers.{i}.mlp.dense_4h_to_h.weight'] = W_proj.t().contiguous()
        out[f'language_model.encoder.layers.{i}.mlp.dense_4h_to_h.bias']   = b_proj

    # Final norm
    out['language_model.encoder.final_layernorm.weight'] = hf_sd['transformer.ln_f.weight']
    out['language_model.encoder.final_layernorm.bias']   = hf_sd['transformer.ln_f.bias']

    # If untied heads were ever needed (not for GPT-2 default):
    if 'lm_head.weight' in hf_sd:
        out['word_embeddings_for_head'] = hf_sd['lm_head.weight']
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf_name', default='gpt2')
    ap.add_argument('--save-dir', required=True)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.hf_name)
    hf  = AutoModelForCausalLM.from_pretrained(args.hf_name, torch_dtype=torch.float32)
    sd  = hf.state_dict()

    mapped = hf_to_megatron_map(sd, cfg.n_layer)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(
        {
            'model': mapped,            # <-- directly use mapped tensors
            'iteration': 0,
            'args': {
                'num_layers': cfg.n_layer,
                'hidden_size': cfg.n_embd,
                'num_attention_heads': cfg.n_head,
                'max_position_embeddings': cfg.n_positions,
                'padded_vocab_size': cfg.vocab_size,
            },
            'optimizer': None,
            'checkpoint_version': 3.0,
        },
        os.path.join(args.save_dir, 'mp_rank_00_model_states.pt')
    )
    print(f"Saved Megatron init checkpoint (iter=0) to {args.save_dir}")

if __name__ == "__main__":
    main()