# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Finetune GPT."""

import datetime
import os
import torch

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.datasets.sft_dataset import SFTDataset

import megatron.legacy.model  # isort: skip
from megatron.core.transformer.lora import apply_lora, count_parameters

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from transformers import AutoModelForCausalLM, AutoTokenizer

##FIXME needs to work with tensor/pipeline sharded model
def convert_hf_tinyllama_to_megatron(hf_model, target_vocab_size=None):
    """
    Converts a Hugging Face TinyLlama model to a Megatron-LM compatible state dictionary.
    This function includes the correct logic for handling Grouped-Query Attention (GQA)
    and SwiGLU MLP weights.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    # Load the state dict and get model configuration
    hf_state_dict = hf_model.state_dict()
    config = hf_model.config
    megatron_state_dict = {}

    # 1. Convert and pad Word Embeddings
    original_embedding_tensor = hf_state_dict['model.embed_tokens.weight']
    original_vocab_size, hidden_size = original_embedding_tensor.shape
    
    padded_embedding_tensor = original_embedding_tensor
    if target_vocab_size and target_vocab_size > original_vocab_size:
        padding_size = target_vocab_size - original_vocab_size
        padding_tensor = torch.zeros(padding_size, hidden_size, dtype=original_embedding_tensor.dtype)
        padded_embedding_tensor = torch.cat([original_embedding_tensor, padding_tensor], dim=0)
    
    megatron_state_dict['embedding.word_embeddings.weight'] = padded_embedding_tensor

    # 2. Convert and pad the Output Layer (lm_head)
    if 'lm_head.weight' in hf_state_dict:
        original_output_tensor = hf_state_dict['lm_head.weight']
        padded_output_tensor = original_output_tensor
        if target_vocab_size and target_vocab_size > original_output_tensor.shape[0]:
            padding_size = target_vocab_size - original_output_tensor.shape[0]
            padding_tensor = torch.zeros(padding_size, hidden_size, dtype=original_output_tensor.dtype)
            padded_output_tensor = torch.cat([original_output_tensor, padding_tensor], dim=0)
        megatron_state_dict['output_layer.weight'] = padded_output_tensor

    megatron_state_dict['decoder.final_layernorm.weight'] = hf_state_dict['model.norm.weight']

    num_layers = config.num_hidden_layers
    num_query_groups = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads
    num_q_heads_per_kv_group = num_attention_heads // num_query_groups

    for i in range(num_layers):
        megatron_state_dict[f'decoder.layers.{i}.input_layernorm.weight'] = hf_state_dict[f'model.layers.{i}.input_layernorm.weight']
        megatron_state_dict[f'decoder.layers.{i}.pre_mlp_layernorm.weight'] = hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight']

        q = hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight']
        k = hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight']
        v = hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight']

        q_reshaped = q.reshape(num_query_groups, num_q_heads_per_kv_group * head_dim, hidden_size)
        k_reshaped = k.reshape(num_query_groups, head_dim, hidden_size)
        v_reshaped = v.reshape(num_query_groups, head_dim, hidden_size)

        qkv_weight = torch.cat([q_reshaped, k_reshaped, v_reshaped], dim=1).reshape(-1, hidden_size)
        
        megatron_state_dict[f'decoder.layers.{i}.self_attention.linear_qkv.weight'] = qkv_weight

        megatron_state_dict[f'decoder.layers.{i}.self_attention.linear_proj.weight'] = hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight']

        gate = hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight']
        up = hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight']

        gate_up_weight = torch.cat([gate, up], dim=0)
        megatron_state_dict[f'decoder.layers.{i}.mlp.linear_fc1.weight'] = gate_up_weight

        megatron_state_dict[f'decoder.layers.{i}.mlp.linear_fc2.weight'] = hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight']

    return megatron_state_dict

def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generates a sequence of tokens autoregressively."""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    for _ in range(max_new_tokens):

        seq_length = input_ids.shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        attention_mask = torch.triu(
            torch.ones((1, seq_length, seq_length), device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )

        with torch.no_grad():
            output_tensor = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        last_token_logits = output_tensor[:, -1, :]
        next_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def test(model):
    model.cuda().bfloat16()

    if (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            prompts = [
                "The capital of France is",
                "Once upon a time,",
                "def hello_world():",
            ]

            for prompt in prompts:
                print_rank_0("-" * 60)
                print_rank_0(f"Prompt: {prompt}")
                
                generated_text = generate_response(model, tokenizer, prompt, max_new_tokens=30)
                
                print_rank_0(f"Generated Text: {generated_text}")

            print_rank_0("-" * 60)
            print_rank_0("Generation test complete.")

        except Exception as e:
            print_rank_0("An error occurred during the generation test:")
            print_rank_0(e)

        model.train()

stimer = StragglerDetector()


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.
    
    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration
        
    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
            )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        hf_tinyllama = AutoModelForCausalLM.from_pretrained(hf_model_name)
        hf_tinyllama.eval()

        converted_weights = convert_hf_tinyllama_to_megatron(hf_tinyllama, 32128)

        model.load_state_dict(converted_weights, strict=False)

        for param in model.parameters():
            param.requires_grad = False

        params_before = count_parameters(model)
        print(f"--- Before LoRA ---")
        print(f"Total parameters:     {params_before['total']:,}")
        print(f"Trainable parameters: {params_before['trainable']:,}\n")

        LORA_R1 = 8
        LORA_R2 = 16
        LORA_ALPHA = 16
        TARGET_MODULES = [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]

        apply_lora(model.decoder, TARGET_MODULES, LORA_R1, LORA_ALPHA)

        params_after = count_parameters(model)

        print(f"--- After LoRA ---")
        print(f"Total parameters:     {params_after['total']:,}")
        print(f"Trainable parameters: {params_after['trainable']:,}")

        test(model)

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
