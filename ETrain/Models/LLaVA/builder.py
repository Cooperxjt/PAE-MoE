#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os, sys
import warnings
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
import torch
from ETrain.Models.LLaVA import *
from ETrain.utils.LLaVA.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    infer_task=0,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if "llava" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            if "MOE" in model_name:
                from AddLora.peft import (
                    PeftModel,
                    TaskType,
                    get_peft_model,
                    AddMOELoraConfig,
                    WEIGHTS_NAME,
                    set_peft_model_state_dict,
                )
            # else:
            #     from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(
                model, model_path, infer_task=infer_task
            )

            # print("Merging LoRA weights...")
            # model = model.merge_and_unload()

            print("Model is loaded...")

        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_projector_weights = torch.load(
                os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
            )
            mm_projector_weights = {
                k: v.to(torch.float16) for k, v in mm_projector_weights.items()
            }
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def save_model_info(model, filename_prefix, stage):
    """保存模型信息到txt文件"""
    filename = f"{filename_prefix}_{stage}.txt"
    with open(filename, "w") as f:
        f.write(f"=== Model Info at stage: {stage} ===\n\n")

        # 1. 保存模型结构
        f.write("1. MODEL STRUCTURE:\n")
        f.write("=" * 50 + "\n")
        f.write(str(model))
        f.write("\n\n")

        # 2. 保存MoE专家权重信息（重点关注部分）
        f.write("2. MoE EXPERTS WEIGHT INFORMATION:\n")
        f.write("=" * 50 + "\n")

        # 查找所有的MoE相关层
        expert_layers = []
        for name, module in model.named_modules():
            if any(
                keyword in name.lower()
                for keyword in ["moe", "expert", "router", "gate"]
            ):
                expert_layers.append((name, module))

        if expert_layers:
            f.write(f"Found {len(expert_layers)} MoE-related layers:\n\n")

            for idx, (name, module) in enumerate(expert_layers):
                f.write(f"[{idx}] Layer: {name} ({module.__class__.__name__})\n")

                # 尝试获取expert权重信息
                for attr_name in [
                    "experts",
                    "expert_weights",
                    "weight",
                    "lora_A",
                    "lora_B",
                ]:
                    if hasattr(module, attr_name):
                        attr = getattr(module, attr_name)
                        f.write(f"  {attr_name}: ")

                        if isinstance(attr, torch.Tensor):
                            f.write(f"shape={attr.shape}, dtype={attr.dtype}\n")

                            # 如果是2D或更高维的权重，保存前5行
                            if attr.dim() >= 2:
                                data = attr.data.cpu().numpy()
                                rows_to_show = min(5, data.shape[0])
                                f.write(f"  First {rows_to_show} rows:\n")
                                for i in range(rows_to_show):
                                    # 限制每行显示的元素数量
                                    row_data = data[i]
                                    if len(row_data) > 10:  # 如果太多，只显示前10个
                                        row_str = str(row_data[:10]) + " ..."
                                    else:
                                        row_str = str(row_data)
                                    f.write(f"    Row {i}: {row_str}\n")
                        elif isinstance(attr, list) or isinstance(
                            attr, torch.nn.ModuleList
                        ):
                            f.write(f"list of {len(attr)} elements\n")
                            # 如果是专家列表，显示每个专家的基本信息
                            for i, expert in enumerate(attr):
                                if i < 5:  # 只显示前5个专家
                                    f.write(f"  Expert {i}: ")
                                    if hasattr(expert, "weight"):
                                        weight = expert.weight
                                        if weight is not None:
                                            f.write(f"weight shape={weight.shape}\n")
                                            # 保存前5行
                                            weight_data = weight.data.cpu().numpy()
                                            rows_to_show = min(5, weight_data.shape[0])
                                            for row in range(rows_to_show):
                                                row_data = weight_data[row]
                                                if len(row_data) > 5:  # 显示前5个元素
                                                    row_str = str(row_data[:5]) + " ..."
                                                else:
                                                    row_str = str(row_data)
                                                f.write(f"    Row {row}: {row_str}\n")
                                    else:
                                        f.write(f"no weight attribute\n")
                                if i >= 5:
                                    f.write(f"  ... and {len(attr)-5} more experts\n")
                                    break
                        else:
                            f.write(f"type: {type(attr)}\n")
                        f.write("\n")
                f.write("-" * 50 + "\n\n")
        else:
            f.write("No MoE-related layers found.\n\n")

        # 3. 保存参数统计信息（精简版）
        f.write("3. PARAMETER STATISTICS:\n")
        f.write("=" * 50 + "\n")
        total_params = 0
        trainable_params = 0
        moe_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            if any(
                keyword in name.lower()
                for keyword in ["moe", "expert", "router", "gate"]
            ):
                moe_params += param.numel()

        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"MoE-related parameters: {moe_params:,}\n")
        f.write(f"Percentage trainable: {trainable_params/total_params*100:.2f}%\n")
        f.write(f"MoE percentage: {moe_params/total_params*100:.2f}%\n")

        f.write(f"\n=== End of {stage} ===\n")

    print(f"Model info saved to {filename}")
    return filename
