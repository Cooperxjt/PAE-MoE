# LLaVA 模型训练脚本
# 基于 FastChat 和 Stanford Alpaca 项目代码修改
# 原始版权声明:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

# 基础库导入
import os
import copy
from dataclasses import dataclass, field  # 用于定义数据类
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers

from utils import MOELoraStatsCallback, EarlySelectionCallback

from ETrain.utils.LLaVA.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from torch.utils.data import Dataset
from ETrain.Train.LLaVA.llava_trainer import LLaVATrainer

from ETrain.Models.LLaVA import *
from ETrain.Dataset import create_LLaVA_data_module
from ETrain.Dataset.dataset import DataArguments
from ETrain.Train.Base_trainer import *
from ETrain.Train.LLaVA.llava_trainer import load_model_from_previous_task

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None
DS_SKIP_CUDA_CHECK = 1


def rank0_print(*args):
    """仅在rank0进程打印信息，避免分布式训练中的重复输出"""
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """模型相关参数配置类"""

    # 模型名称或路径
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_projector_expert_num: Optional[int] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    cur_task: Optional[int] = field(default=None)

    EWC: bool = field(default=False)
    EWC_lambda: float = field(default=0.5)

    LWF: bool = field(default=False)
    LWF_lambda: float = field(default=0.1)

    task_layer_map_file: Optional[str] = field(default=None) 

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练相关参数配置类，继承自HuggingFace TrainingArguments"""

    # 缓存目录
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "逗号分隔的 LoRA target modules，为空则自动检测全部线性层"},
    )
    mm_projector_lr: Optional[float] = None
    proj_lora_r: int = field(default=32, metadata={"help": "Projector LoRA rank"})
    group_by_modality_length: bool = field(default=False)

    # --- Early Selection ---
    early_selection_enabled: bool = field(default=False)
    early_selection_ratio: float = field(
        default=0.1,
        metadata={"help": "训练到总步数的多少比例时触发 early selection（默认 10%）"},
    )
    early_selection_top_k: float = field(
        default=0.5,
        metadata={"help": "保留 top-k 比例的层（默认 50%）"},
    )
    early_selection_gate_beta: float = field(default=1.0)
    early_selection_gate_gamma: float = field(default=2.0)


def train():
    """主训练函数"""
    global local_rank  # 分布式训练的rank变量

    # 1. 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # 2. 创建LLaVA模型和tokenizer
    model, tokenizer = create_LLaVA_model(
        training_args,
        model_args,
        data_args,
        bnb_model_from_pretrained_args,
        compute_dtype,
        local_rank,
    )

    # 3. 配置EWC(Elastic Weight Consolidation)参数 - 用于防止灾难性遗忘
    if model_args.EWC:
        training_args.EWC = model_args.EWC
        model.base_model.model.EWC = model_args.EWC
        model.base_model.model.EWC_lambda = model_args.EWC_lambda  # EWC正则化强度
    if model_args.LWF:
        training_args.LWF = model_args.LWF
        model.base_model.model.LWF = model_args.LWF
        model.base_model.model.LWF_lambda = model_args.LWF_lambda

    # 4. 如果提供了预训练模型路径，则加载之前任务的模型
    if model_args.previous_task_model_path is not None:
        load_model_from_previous_task(model, model_args)  # 加载之前任务的模型参数

    # 5. 创建数据模块 - 包含训练集、验证集等
    data_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)

    if model_args.EWC and model_args.previous_task_model_path is not None:
        fisher = torch.load(
            os.path.join(model_args.previous_task_model_path, "fisher.bin"),
            map_location="cpu",
        )
        optpar = torch.load(
            os.path.join(model_args.previous_task_model_path, "optpar.bin"),
            map_location="cpu",
        )
        fisher = {(k[6:] if k.startswith("model") else k): v for k, v in fisher.items()}
        optpar = {(k[6:] if k.startswith("model") else k): v for k, v in optpar.items()}
        model.base_model.model.fisher = fisher
        model.base_model.model.optpar = optpar

    # 6. 初始化训练器
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module  # 传入数据集
    )
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:

    # 7. 处理LWF(Learning Without Forgetting)逻辑
    if model_args.LWF:
        final_logits = trainer.before_train()  # 获取之前任务的logits用于知识蒸馏

        model, tokenizer = create_LLaVA_model(
            training_args,
            model_args,
            data_args,
            bnb_model_from_pretrained_args,
            compute_dtype,
            local_rank,
        )

        if model_args.previous_task_model_path is not None:
            # load model from previous task
            load_model_from_previous_task(model, model_args)

        if model_args.LWF:
            training_args.LWF = model_args.LWF
            model.base_model.model.LWF = model_args.LWF
            model.base_model.model.LWF_lambda = model_args.LWF_lambda
        model.base_model.model.previous_logits = final_logits

        data_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)

        trainer = LLaVATrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )

    # === 加入统计回调 ===
    stats_cb = MOELoraStatsCallback(
        output_dir=training_args.output_dir,
        log_every_steps=50,       # 建议 50 或 100 起步
        reset_after_log=False,    # 先别 reset，方便看 EMA 走势
        filename="moe_lora_stats.jsonl",
        include_train_log=True,
    )
    trainer.add_callback(stats_cb)

    # === 加入 Early Selection 回调 ===
    if training_args.early_selection_enabled and model_args.cur_task is not None:
        early_cb = EarlySelectionCallback(
            output_dir=training_args.output_dir,
            cur_task=int(model_args.cur_task),
            adapter_name="default",
            selection_ratio=training_args.early_selection_ratio,
            top_k_ratio=training_args.early_selection_top_k,
            gate_beta=training_args.early_selection_gate_beta,
            gate_gamma=training_args.early_selection_gate_gamma,
        )
        trainer.add_callback(early_cb)
        rank0_print(
            f"[EarlySelection] Enabled: trigger at {training_args.early_selection_ratio*100:.0f}% steps, "
            f"keep top {training_args.early_selection_top_k*100:.0f}% layers"
        )

    # 8. 开始训练
    trainer.train()  # 执行训练循环

    # 9. 保存训练状态和模型
    trainer.save_state()  # 保存训练器状态
    trainer.save_trained_model(training_args)

    # 10. EWC训练后处理
    if model_args.EWC:
        trainer.after_train()  # 计算并保存Fisher信息矩阵


if __name__ == "__main__":
    train()
