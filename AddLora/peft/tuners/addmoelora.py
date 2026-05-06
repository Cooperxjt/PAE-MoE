# -*- encoding: utf-8 -*-
import warnings
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from attr import has

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)
from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


# =========================
# Config
# =========================
@dataclass
class AddMOELoraConfig(LoraConfig):
    """
    Sparse AddMOE-LoRA 配置（按层增量创建专家）：

    - shared expert 永远是 task_id=0
    - 每一层有一个稀疏的 task->expert 映射（ModuleDict），不会因为“缺专家”而错位
    - 新任务到来时，只在 moe_create_layers 指定的层创建 cur_task 的专家；其他层不创建（省参数）
    - layer_task_map 用于结构一致性：从 previous 模型读取后传入，保证不破坏旧结构

    典型用法：
        Task1: cur_task=1, moe_create_layers=None（全层创建 expert1）
        Task2: cur_task=2, moe_create_layers=[...子集...], layer_task_map 从 Task1 读取后更新
    """

    task_embedding_dim: int = field(default=64)
    cur_task: int = field(default=0)

    enable_stats: bool = field(default=True)
    stats_interval: int = field(default=10)
    stats_ema_beta: float = field(default=0.98)

    # 仅在这些层创建当前任务专家（None => 所有层都创建）
    moe_create_layers: Optional[List[int]] = field(default=None)

    # 结构一致性：每层已有的 task experts（通常来自 previous 模型）例：{0:[1], 1:[1], ..., 31:[1], 6:[1,2]}
    layer_task_map: Optional[Dict[int, List[int]]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_Add


# =========================
# Model wrapper
# =========================
class AddMOELoraModel(LoraModel):
    """
    AddMOE LoRA 模型包装：
    - 在替换模块阶段决定“每层创建哪些 task experts”
    - 稀疏专家存储：ModuleDict(task_id -> expert)，推理按 task_id 查找不会错位
    """

    def __init__(self, model, config, adapter_name: str):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name: str, config=None):
        if config is not None:
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_addmoelora_config(config, model_config)
            self.peft_config[adapter_name] = config

        self._find_and_replace(adapter_name)

        # 冻结 base，解冻 LoRA（PEFT 原逻辑）
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        # gate 可训练（但我们会在 layer 内部进一步控制只训练当前任务 gate）
        for m in self.model.modules():
            if hasattr(m, "lora_gate") and isinstance(m.lora_gate, nn.ModuleDict):
                for p in m.lora_gate.parameters():
                    p.requires_grad = True

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    @staticmethod
    def _extract_layer_id(module_key: str) -> Optional[int]:
        # 适配 LLaMA/LLaVA 常见命名：model.layers.{i}.xxx
        m = re.search(r"model\.layers\.(\d+)\.", module_key)
        return int(m.group(1)) if m else None

    def _find_and_replace(self, adapter_name: str):
        lora_config: AddMOELoraConfig = self.peft_config[adapter_name]
        self._check_quantization_dependency()

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        active_layers = lora_config.moe_create_layers
        active_set = (
            None if active_layers is None else set(int(x) for x in active_layers)
        )

        layer_task_map = lora_config.layer_task_map or {}
        
        train_task = int(lora_config.cur_task)

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, _ = _get_submodules(self.model, key)
            layer_id = self._extract_layer_id(key)

            # is_active：仅 active 层创建当前任务专家；active_set=None => 全层 active（Task1）
            is_active = True
            if active_set is not None and layer_id is not None:
                is_active = layer_id in active_set

            # tasks_to_create = shared(0) + old_tasks(layer_task_map) + (cur_task if is_active)
            tasks_to_create: List[int] = [0]
            if layer_id is not None:
            # 尝试用 int 键获取
                tasks = layer_task_map.get(layer_id, [])
                # 若未获取到，尝试用 str 键获取
                if not tasks:
                    tasks = layer_task_map.get(str(layer_id), [])
                tasks_to_create += list(tasks)
            if is_active:
                tasks_to_create.append(train_task)
            tasks_to_create = sorted(set(int(t) for t in tasks_to_create))

            # ===== 如果已经是我们自己的 AddMOELoraLinear：不要 update_layer 重建！改为增量 ensure =====
            if isinstance(target, AddMOELoraLinear):
                target.ensure_tasks(adapter_name, tasks_to_create)
                target.update_hparams_only(
                    adapter_name=adapter_name,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                )
                target.freeze_for_task(adapter_name, cur_task=train_task)
                continue

            # 如果是别的 LoraLayer（不太会，但保留兼容）
            if isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
                continue

            new_module = self._create_new_module(
                lora_config=lora_config,
                adapter_name=adapter_name,
                target=target,
                layer_name=key,
                tasks_to_create=tasks_to_create,
            )
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(
        self,
        lora_config: AddMOELoraConfig,
        adapter_name: str,
        target,
        layer_name: str,
        tasks_to_create: List[int],
    ):
        bias = hasattr(target, "bias") and target.bias is not None

        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        # 先拒绝量化，避免“看似能跑但 load/shape 不一致”
        if (
            loaded_in_8bit
            and is_bnb_available()
            and isinstance(target, bnb.nn.Linear8bitLt)
        ):
            raise NotImplementedError(
                "Sparse AddMOE is not implemented for bnb 8bit Linear yet."
            )
        if (
            loaded_in_4bit
            and is_bnb_4bit_available()
            and isinstance(target, bnb.nn.Linear4bit)
        ):
            raise NotImplementedError(
                "Sparse AddMOE is not implemented for bnb 4bit Linear yet."
            )
        if not isinstance(target, torch.nn.Linear):
            raise ValueError(
                f"Sparse AddMOE currently supports torch.nn.Linear only, got {type(target)} at {layer_name}"
            )

        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "cur_task": int(lora_config.cur_task),
            "tasks_to_create": [int(t) for t in tasks_to_create],
            "layer_name": layer_name,
            "enable_stats": bool(getattr(lora_config, "enable_stats", False)),
            "stats_interval": int(getattr(lora_config, "stats_interval", 10)),
            "stats_ema_beta": float(getattr(lora_config, "stats_ema_beta", 0.98)),
        }

        in_features, out_features = target.in_features, target.out_features
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but target is torch.nn.Linear. Setting to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False

        return AddMOELoraLinear(
            adapter_name,
            in_features,
            out_features,
            bias=bias,
            **kwargs,
        )

    @staticmethod
    def _prepare_addmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def iter_addmoe_layers(self):
        for m in self.model.modules():
            if isinstance(m, AddMOELoraLinear):
                yield m

# =========================
# LoRA Layer base (sparse experts)
# =========================
class AddMOELoraLayer(LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        cur_task: int,
        tasks_to_create: Optional[List[int]] = None,
    ):
        super().__init__(in_features, out_features)
        self.cur_task = int(cur_task)
        self.tasks_to_create = tasks_to_create

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {
                        adapter_name: AddMOELinearA(
                            self.in_features, r, self.tasks_to_create
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: AddMOELinearB(
                            r, self.out_features, self.tasks_to_create
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A:
            for _k, expA in self.lora_A[adapter_name].loraA.items():
                nn.init.normal_(expA.mlp.weight, mean=0.0, std=0.01)
        if adapter_name in self.lora_B:
            for _k, expB in self.lora_B[adapter_name].loraB.items():
                nn.init.zeros_(expB.mlp.weight)


# =========================
# Linear layer
# =========================
class AddMOELoraLinear(nn.Linear, AddMOELoraLayer):
    """
    Sparse AddMOE-LoRA Linear：

    - 每层总是有 shared expert (task_id=0)
    - 如果该层包含 cur_task expert：
        用 gate(task_id=cur_task, 2维) 在 shared vs cur_task 之间混合
        否则：
        fallback shared-only（只加 shared）
    - experts / gate 都用 ModuleDict(task_id->module) 存，推理按 task_id 查找，不会错位
    """

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = bool(kwargs.pop("init_lora_weights", True))
        cur_task = int(kwargs.pop("cur_task", 0))
        tasks_to_create = kwargs.pop("tasks_to_create", None)
        self.layer_name = kwargs.pop("layer_name", "unknown_layer")

        # stats
        self.enable_stats = bool(kwargs.pop("enable_stats", True))
        self.stats_interval = int(kwargs.pop("stats_interval", 10))
        self.stats_ema_beta = float(kwargs.pop("stats_ema_beta", 0.98))
        self._stats = {
            "delta_lora_total_ema": None,
            "delta_task_removed_ema": None,
            "delta_task_lora_removed_ema": None,
        }

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        AddMOELoraLayer.__init__(
            self, in_features, out_features, cur_task=cur_task, tasks_to_create=tasks_to_create
        )

        # lora_gate[adapter_name] 是 ModuleDict(task_id -> Linear(in,2))
        self.lora_gate = nn.ModuleDict({adapter_name: nn.ModuleDict()})

        # base weight freeze
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

        # 确保 gate 对应 tasks 存在（只为非0任务建 gate）
        self._ensure_gates(adapter_name, self.tasks_to_create)

        # 训练：只训练 shared(0) + cur_task（以及 cur_task 的 gate）
        self.freeze_for_task(adapter_name, cur_task=int(self.cur_task))

        self._gate_reg_loss = None  # 用于暂存本层 gate 正则（tensor，需保留计算图）

    # ===== gate 管理：按 task_id 维护 =====
    def _ensure_gates(self, adapter_name: str, tasks: List[int]):
        if adapter_name not in self.lora_gate:
            self.lora_gate[adapter_name] = nn.ModuleDict()

        gate_dict: nn.ModuleDict = self.lora_gate[adapter_name]
        for t in sorted(set(int(x) for x in tasks)):
            if t == 0:
                continue
            k = str(t)
            if k not in gate_dict:
                gate_dict[k] = nn.Linear(self.in_features, 2, bias=False)

    def has_gate(self, adapter_name: str, task_id: int) -> bool:
        if adapter_name not in self.lora_gate:
            return False
        return str(int(task_id)) in self.lora_gate[adapter_name]

    # ===== 供 wrapper 增量扩展调用 =====
    def ensure_tasks(self, adapter_name: str, tasks: List[int]):
        tasks = sorted(set(int(t) for t in tasks))
        if 0 not in tasks:
            tasks = [0] + tasks

        # 如果 adapter 还没初始化过 lora_A/B，就跳过（正常不会发生）
        if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            return

        r = int(self.r[adapter_name])
        for t in tasks:
            k = str(int(t))
            if not self.lora_A[adapter_name].has(t):
                self.lora_A[adapter_name].loraA[k] = AddMOEExpert(self.in_features, r)
                nn.init.normal_(
                    self.lora_A[adapter_name].loraA[k].mlp.weight, mean=0.0, std=0.01
                )
            if not self.lora_B[adapter_name].has(t):
                self.lora_B[adapter_name].loraB[k] = AddMOEExpert(r, self.out_features)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[k].mlp.weight)

            # ---- Warm-start: 新建任务 expert 用 shared(0) 参数初始化（如果 shared 已存在）----
            # 目的：避免新任务随机初始化导致 gate/更新过度依赖 shared，提升新任务早期可用性并降低遗忘。
            if int(t) != 0 and int(t) != 1:
                shared_k = "0"
                a_dict = self.lora_A[adapter_name].loraA
                b_dict = self.lora_B[adapter_name].loraB
                if shared_k in a_dict and shared_k in b_dict and k in a_dict and k in b_dict:
                    with torch.no_grad():
                        a_dict[k].mlp.weight.copy_(a_dict[shared_k].mlp.weight)
                        b_dict[k].mlp.weight.copy_(b_dict[shared_k].mlp.weight)
                        # 打破完全同构（极小噪声，不改变分布，只避免数值完全一致）
                        a_dict[k].mlp.weight.add_(1e-4 * torch.randn_like(a_dict[k].mlp.weight))

        # gate 也增量补齐（非0任务）
        self._ensure_gates(adapter_name, tasks)

        self.tasks_to_create = sorted(set(self.tasks_to_create).union(set(tasks)))

    def _ema_update(self, key: str, value: float):
        beta = self.stats_ema_beta
        cur = self._stats[key]
        if cur is None:
            self._stats[key] = float(value)
        else:
            self._stats[key] = beta * float(cur) + (1.0 - beta) * float(value)

    @torch.no_grad()
    def get_stats(self, reset: bool = False) -> dict:
        out = {"layer_name": self.layer_name, **self._stats}
        if reset:
            self.reset_stats()
        return out

    def update_hparams_only(
        self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float
    ):
        # r 不能变（否则旧权重 shape 对不上）
        if adapter_name in self.r and int(self.r[adapter_name]) != int(r):
            raise ValueError(
                f"Cannot change r for existing layer. old={self.r[adapter_name]} new={r}"
            )

        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        if lora_dropout > 0.0:
            self.lora_dropout.update(
                nn.ModuleDict({adapter_name: nn.Dropout(p=lora_dropout)})
            )
        else:
            self.lora_dropout.update(nn.ModuleDict({adapter_name: nn.Identity()}))

    def freeze_for_task(self, adapter_name: str, cur_task: int):
        shared_id = 0
        cur_id = int(cur_task)
        train_ids = {shared_id, cur_id}

        # experts
        if adapter_name in self.lora_A:
            for k, exp in self.lora_A[adapter_name].loraA.items():
                exp.mlp.weight.requires_grad = int(k) in train_ids
        if adapter_name in self.lora_B:
            for k, exp in self.lora_B[adapter_name].loraB.items():
                exp.mlp.weight.requires_grad = int(k) in train_ids

        # gates：只训练当前任务 gate
        if adapter_name in self.lora_gate:
            gate_dict: nn.ModuleDict = self.lora_gate[adapter_name]
            for k, gate in gate_dict.items():
                for p in gate.parameters():
                    p.requires_grad = int(k) == cur_id

    def freeze_task_expert(self, adapter_name: str, cur_task: int):
        """
        冻结当前层的 task expert 和 gate（early selection 判定该层不重要时调用）。
        仅冻结 cur_task 的 expert，shared(0) 保持不变。
        """
        cur_id = int(cur_task)
        k = str(cur_id)

        if adapter_name in self.lora_A and self.lora_A[adapter_name].has(cur_id):
            self.lora_A[adapter_name].loraA[k].mlp.weight.requires_grad = False
        if adapter_name in self.lora_B and self.lora_B[adapter_name].has(cur_id):
            self.lora_B[adapter_name].loraB[k].mlp.weight.requires_grad = False

        if adapter_name in self.lora_gate:
            gate_dict: nn.ModuleDict = self.lora_gate[adapter_name]
            if k in gate_dict:
                for p in gate_dict[k].parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, **kwargs):
        self._gate_reg_loss = None

        previous_dtype = x.dtype
        
        # no adapter
        if self.active_adapter not in self.lora_A:
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        # adapters disabled
        if self.disable_adapters:
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        # LoRA rank 0
        if self.r[self.active_adapter] <= 0:
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        # base out
        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )
        base_out = result.detach()

        # cast to LoRA dtype
        exp0 = self.lora_A[self.active_adapter].loraA["0"]
        x_cast = x.to(exp0.mlp.weight.dtype)
        x_drop = self.lora_dropout[self.active_adapter](x_cast)
        scaling = self.scaling[self.active_adapter]

        # shared expert
        shared_a = self.lora_A[self.active_adapter].forward_expert(x_drop, 0)
        shared = self.lora_B[self.active_adapter].forward_expert(shared_a, 0)

        runtime_task_id = getattr(self, "runtime_task_id", None)
        if runtime_task_id is None:
            runtime_task_id = self.cur_task
        task_id = int(runtime_task_id)

        has_task = self.lora_A[self.active_adapter].has(task_id) and self.lora_B[
            self.active_adapter
        ].has(task_id)
        has_gate = self.has_gate(self.active_adapter, task_id)

        # shared-only
        if not has_task or not has_gate:
            shared = shared.detach()
            return (result + shared * scaling).to(previous_dtype)

        # task expert
        task_a = self.lora_A[self.active_adapter].forward_expert(x_drop, task_id)
        task = self.lora_B[self.active_adapter].forward_expert(task_a, task_id)

        # gate(task_id): 2 dims
        gate_logits_2 = self.lora_gate[self.active_adapter][str(task_id)](x_drop)
        gate_w = torch.softmax(gate_logits_2, dim=-1)
        w_shared = gate_w[..., 0].unsqueeze(-1)
        w_task = gate_w[..., 1].unsqueeze(-1)

        lora_out = (w_shared * shared + w_task * task) * scaling

        full_out = result+lora_out

        if self.training:
            # gate entropy regularization: discourage gate collapse
            avg_w = gate_w.detach().mean(dim=tuple(range(gate_w.dim() - 1)))  # [2]
            self._gate_reg_loss = -(avg_w * torch.log(avg_w + 1e-8)).sum()

            if self.enable_stats:
                with torch.no_grad():
                    eps = 1e-8

                    def rms(t):
                        return t.detach().float().pow(2).mean().sqrt()

                    shared_only_out = base_out + (w_shared * shared) * scaling

                    r_full = rms(full_out)

                    delta_task = (rms(full_out - shared_only_out) / (r_full + eps)).cpu().item()

                    delta_task_lora = (
                        rms(lora_out - (w_shared * shared) * scaling) / (rms(lora_out) + eps)
                    ).cpu().item()

                    delta_lora_total = (rms(full_out - base_out) / (r_full + eps)).cpu().item()

                    self._ema_update("delta_lora_total_ema", delta_lora_total)
                    self._ema_update("delta_task_removed_ema", delta_task)
                    self._ema_update("delta_task_lora_removed_ema", delta_task_lora)

        return full_out.to(previous_dtype)

# =========================
# A / B blocks (task_id -> expert)
# =========================
class AddMOELinearA(nn.Module):
    def __init__(self, in_features: int, out_features: int, tasks_to_create: List[int]):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleDict(
            {
                str(int(t)): AddMOEExpert(self.in_features, self.out_features)
                for t in sorted(set(int(x) for x in tasks_to_create))
            }
        )

    def has(self, task_id: int) -> bool:
        return str(int(task_id)) in self.loraA

    def forward_expert(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.loraA[str(int(task_id))](x)


class AddMOELinearB(nn.Module):
    def __init__(self, in_features: int, out_features: int, tasks_to_create: List[int]):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleDict(
            {
                str(int(t)): AddMOEExpert(self.in_features, self.out_features)
                for t in sorted(set(int(x) for x in tasks_to_create))
            }
        )

    def has(self, task_id: int) -> bool:
        return str(int(task_id)) in self.loraB

    def forward_expert(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.loraB[str(int(task_id))](x)


class AddMOEExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)

    @property
    def weight(self):
        return self.mlp.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

def set_runtime_task_id(model, task_id: int):
    for m in model.modules():
        # 你也可以换成 isinstance(m, AddMOELoraLinear)
        if isinstance(m, AddMOELoraLinear):
            setattr(m, "runtime_task_id", int(task_id)) 
        elif hasattr(m, "runtime_task_id"):
            m.runtime_task_id = int(task_id)