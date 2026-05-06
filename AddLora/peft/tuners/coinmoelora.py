# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class CoINMOELoraConfig(LoraConfig):
    """
    CoIN MoE LoRA配置类，继承自标准LoRA配置
    这是基于MMoE (Multi-gate Mixture-of-Experts) 架构的LoRA变种
    
    Attributes:
        task_embedding_dim: 任务嵌入维度，用于区分不同任务的特征学习
        expert_num: 专家数量，控制MoE中专家网络的个数，默认4个
    """
    task_embedding_dim: int = field(default=64)   # 任务嵌入维度，默认为64
    expert_num: int = field(default=4)            # 专家数量，默认为4

    def __post_init__(self):
        """初始化后处理，设置peft类型为MOE_LORA_CoIN"""
        self.peft_type = PeftType.MOE_LORA_CoIN

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN


class CoINMOELoraModel(LoraModel):
    """
    CoIN MoE LoRA模型类，基于MMoE架构的LoRA实现
    从预训练transformer模型创建MoE LoRA模型
    主要特性:
    - 支持多专家网络 (Multi-Expert)
    - 基于任务的路由机制
    - 与标准LoRA兼容的接口
    """
    def __init__(self, model, config, adapter_name):
        """初始化MoE LoRA模型"""
        nn.Module.__init__(self)
        self.model = model                    # 基础模型
        self.forward = self.model.forward     # 继承forward方法
        self.peft_config = config               # 配置信息
        self.add_adapter(adapter_name, self.peft_config[adapter_name])  # 添加适配器

    def add_adapter(self, adapter_name, config=None):
        """
        为模型添加MoE LoRA适配器
        
        Args:
            adapter_name: 适配器名称
            config: MoE LoRA配置，如果为None则使用默认配置
        """
        if config is not None:  # 如果有自定义配置
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_coinmoelora_config(config, model_config)   # 准备配置
            self.peft_config[adapter_name] = config  # 替换原始配置
        
        self._find_and_replace(adapter_name)  # 查找并替换目标模块
        
        # 验证配置：MoE LoRA仅支持1个带bias的适配器
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        # 标记只有LoRA参数可训练，冻结基础模型参数
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        
        # 推理模式下冻结适配器
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
    def _create_new_module(self, lora_config, adapter_name, target):
        """创建新的MoE LoRA模块来替换目标模块"""
        bias = hasattr(target, "bias") and target.bias is not None
        
        # 构建MoE LoRA参数的kwargs
        kwargs = {
            "r": lora_config.r,                           # LoRA秩
            "lora_alpha": lora_config.lora_alpha,         # LoRA缩放系数
            "lora_dropout": lora_config.lora_dropout,     # LoRA dropout率
            "fan_in_fan_out": lora_config.fan_in_fan_out, # 输入输出维度处理
            "init_lora_weights": lora_config.init_lora_weights,  # 权重初始化
            "task_embedding_dim": lora_config.task_embedding_dim,  # MoE: 任务嵌入维度
            "expert_num": lora_config.expert_num,         # MoE: 专家数量
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = CoINMOELoraLinear(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_coinmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class CoINMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: CoINMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: CoINMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class CoINMOELoraLinear(nn.Linear, CoINMOELoraLayer):
    """
    CoIN MoE LoRA线性层，结合了标准线性层和MoE LoRA层
    这是MoE LoRA的核心实现类，实现多专家网络的参数高效微调
    
    设计理念:
    - nn.Linear: LLM中的预训练权重 (冻结)
    - CoINMOELoraLayer: 设计的可训练MoE LoRA参数
    """
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # 如果要替换的层存储权重为(fan_in, fan_out)，则设为True
        **kwargs,
    ):
        """初始化MoE LoRA线性层"""
        init_lora_weights = kwargs.pop("init_lora_weights", True)  # LoRA权重初始化标志
        self.expert_num = kwargs.pop("expert_num", 4)              # MoE专家数量，默认4
        self.te_dim = kwargs.pop("task_embedding_dim", 64)         # 任务嵌入维度，默认64

        # 初始化父类
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CoINMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num)
        
        # 初始化路由网络 (Gate Network)
        # 路由网络根据输入特征选择最合适的专家
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

        # 冻结预训练权重矩阵，只训练LoRA参数
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T  # 转置权重以适应fan_in_fan_out设置

        nn.Linear.reset_parameters(self)  # 重置线性层参数
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)  # 更新LoRA层
        self.active_adapter = adapter_name  # 设置活跃适配器


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        """MoE LoRA前向传播，包含专家路由和加权组合"""
        previous_dtype = x.dtype  # 保存原始数据类型

        # 情况1: 无适配器，直接使用线性层
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        # 情况2: 适配器被禁用
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()  # 取消合并
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        # 情况3: 启用MoE LoRA (核心逻辑)
        elif self.r[self.active_adapter] > 0:
            # 基础线性层输出
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            # 类型和设备转换
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            self.lora_router = self.lora_router.to(x.device)
            
            # 专家路由计算
            router = self.lora_router[self.active_adapter](x)     # 路由层预测
            router = torch.softmax(router, dim=-1)                # Softmax归一化为权重
            
            # MoE处理：加权组合所有专家输出
            for i in range(self.expert_num):
                # 计算单个专家的LoRA贡献
                expert_contribution = self.lora_B[self.active_adapter].loraB[i](
                    self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x))
                )
                
                # 加权添加到结果中
                result += expert_contribution * self.scaling[self.active_adapter] * router[:,:,i].unsqueeze(-1)
        
        # 情况4: LoRA秩为0，仅使用基础线性层
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)  # 恢复原始数据类型
        return result
    


class CoINMOELinearA(nn.Module):
    '''MoE LoRA A层模块，负责将输入投影到低维专家空间'''
    def __init__(self, in_features, out_features, expert_num) -> None:
        """初始化MoE LoRA A层"""
        super().__init__()

        self.expert_num = expert_num                    # 专家数量
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])                  # 专家模块列表

        # LoRA秩必须能被专家数量整除，确保均匀分配
        assert self.out_features % self.expert_num == 0
        self.r = self.out_features // self.expert_num   # 每个专家分配的秩
        
        # 初始化专家网络
        for _ in range(self.expert_num):
            self.loraA.append(CoINMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class CoINMOELinearB(nn.Module):
    '''MoE LoRA B层模块，负责将专家输出投影回原始维度'''
    def __init__(self, in_features, out_features, expert_num) -> None:
        """初始化MoE LoRA B层"""
        super().__init__()

        self.expert_num = expert_num                    # 专家数量
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])                  # 专家模块列表

        # 输入特征必须能被专家数量整除
        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num    # 每个专家输入维度
        
        # 初始化专家网络
        for _ in range(self.expert_num):
            self.loraB.append(CoINMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs



class CoINMOEExpert(nn.Module):
    """单个专家网络，实现LoRA A或B块的功能"""
    
    def __init__(self, in_features, out_features):
        """初始化专家网络"""
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)  # 线性变换
        self.weight = self.mlp.weight                                           # 专家权重

    def forward(self, x):
        """专家网络前向传播"""
        # LoRA A块或B块的前向计算
        y = self.mlp(x)  # 线性变换
        return y



class CoINMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class CoINMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: CoINMOELoraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits