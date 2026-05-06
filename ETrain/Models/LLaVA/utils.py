from transformers import AutoConfig
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers

from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from torch.utils.data import Dataset

from ETrain.Models.LLaVA import *
from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from ETrain.utils.LLaVA import conversation as conversation_lib

def rank0_print(local_rank,*args):
    if local_rank == 0:
        print(*args)

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def create_LLaVA_model(training_args, model_args, data_args, bnb_model_from_pretrained_args, compute_dtype, local_rank):
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        if model_args.cur_task == None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        else:
            from AddLora.peft import PeftModel, TaskType, get_peft_model, AddMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            cur_task = int(model_args.cur_task)
            num_layers = 32  # 你如果能从 model.config.num_hidden_layers 取更好
            map_path = getattr(model_args, "task_layer_map_file", None)

            moe_create_layers, layer_task_map = build_moe_layers_and_history_map(
                cur_task=cur_task,
                num_layers=num_layers,
                task_layer_map_file=map_path,
            )

            print(moe_create_layers, layer_task_map)
            
            kwargs = {
                "task_embedding_dim": model_args.task_embedding_dim,
                "cur_task": cur_task,
                "moe_create_layers": moe_create_layers,
                "layer_task_map": layer_task_map,
            }

            # 支持自定义 target_modules，默认自动检测全部线性层
            custom_target = getattr(training_args, "lora_target_modules", None)
            if custom_target:
                target_modules = [m.strip() for m in custom_target.split(",")]
            else:
                target_modules = find_all_linear_names(model)

            lora_config = AddMOELoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type=TaskType.CAUSAL_LM_Add,
                **kwargs
            )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print(local_rank,"Adding LoRA adapters...")

        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            print("Freezing Projector")
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # Projector LoRA + Router 集成
        projector = model.get_model().mm_projector
        if hasattr(projector, 'experts') and model_args.cur_task is not None:
            from .multimodal_projector.builder import ProjectorExpertWithLoRA
            from .multimodal_projector.task_classifier import create_projector_task_router

            cur_task = int(model_args.cur_task)
            cur_expert_idx = (cur_task - 1) % projector.num_experts
            proj_lora_r = getattr(training_args, "proj_lora_r", 32)

            # 1) 冻结所有 expert base 权重
            for p in projector.parameters():
                p.requires_grad = False

            # 2) 用 LoRA 包裹当前 expert
            projector.experts[cur_expert_idx] = ProjectorExpertWithLoRA(
                projector.experts[cur_expert_idx], r=proj_lora_r
            )
            rank0_print(local_rank,
                f"[Projector] Expert {cur_expert_idx} wrapped with LoRA (r={proj_lora_r}), "
                f"others frozen"
            )

            # 3) 初始化 Router
            router_load_path = None
            if model_args.previous_task_model_path is not None:
                import os as _os
                candidate = _os.path.join(model_args.previous_task_model_path, "router")
                if _os.path.exists(_os.path.join(candidate, "projector_task_router.pt")):
                    router_load_path = candidate

            router_manager = create_projector_task_router(
                hidden_size=model.config.hidden_size,
                num_experts=projector.num_experts,
                scorer_hidden=64,
                cls_weight=0.1,
                load_from=router_load_path,
            )
            router_manager.prepare_new_task(task_id=cur_task, expert_idx=cur_expert_idx)

            # 挂载到 projector
            projector.router_manager = router_manager
            rank0_print(local_rank,
                f"[Router] Initialized, registered experts: {router_manager.router.registered_experts}"
            )

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    return model, tokenizer


import json
from typing import Dict, List, Optional, Tuple

def build_moe_layers_and_history_map(
    cur_task: int,
    num_layers: int,
    task_layer_map_file: Optional[str],
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    返回:
      moe_create_layers: 当前任务在哪些 layer 创建 cur_task expert
      layer_task_map: {layer_id: [old_task_ids...]}  (不含 shared=0)
    """
    if cur_task <= 0:
        raise ValueError(f"cur_task must be >= 1, got {cur_task}")

    # ===== Task1：全层创建；历史为空 =====
    # if cur_task == 1:
    #     moe_create_layers = list(range(num_layers))
    #     layer_task_map: Dict[int, List[int]] = {}
    #     return moe_create_layers, layer_task_map

    if not task_layer_map_file:
        raise ValueError("task_layer_map_file is required when cur_task > 1")

    with open(task_layer_map_file, "r", encoding="utf-8") as f:
        task2layers = json.load(f)

    
    # 1) 当前任务的 create_layers
    v = task2layers.get(str(cur_task), None)
    if v is None:
        raise KeyError(f"task {cur_task} not found in {task_layer_map_file}")
    moe_create_layers = sorted({int(x) for x in v})

    # 2) 生成 layer_task_map：汇总所有 < cur_task 的任务在各层出现过
    layer_task_map: Dict[int, set] = {i: set() for i in range(num_layers)}
    for t in range(cur_task):  # 小于当前任务
        layers = task2layers.get(str(t), None)
        if layers is None:
            # 你也可以选择 raise；这里选择跳过并继续
            continue
        for lid in layers:
            lid = int(lid)
            if 0 <= lid < num_layers:
                layer_task_map[lid].add(t)

    # 转成 List[int]，并把“空集合的层”删掉（更干净）
    layer_task_map_out: Dict[int, List[int]] = {
        lid: sorted(list(ts))
        for lid, ts in layer_task_map.items()
        if len(ts) > 0
    }

    return moe_create_layers, layer_task_map_out
