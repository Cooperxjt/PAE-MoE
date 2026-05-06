import pdb
import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


# ===========================================================================
# Projector Expert with LoRA
# ===========================================================================
class ProjectorExpertWithLoRA(nn.Module):
    """
    将一个 nn.Sequential expert 包裹上 LoRA adapter。
    base weights 全部冻结，只训练 LoRA 参数。
    参数名包含 'lora_' 以兼容 save/load 逻辑。
    """

    def __init__(self, base_expert: nn.Sequential, r: int = 32):
        super().__init__()
        self.base_layers = nn.ModuleList()
        self.lora_adapters = nn.ModuleDict()

        for i, module in enumerate(base_expert):
            self.base_layers.append(module)
            # 冻结 base 权重
            for p in module.parameters():
                p.requires_grad = False

            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features
                lora_a = nn.Linear(in_f, r, bias=False)
                lora_b = nn.Linear(r, out_f, bias=False)
                nn.init.normal_(lora_a.weight, std=0.01)
                nn.init.zeros_(lora_b.weight)
                self.lora_adapters[str(i)] = nn.ModuleDict({
                    'lora_A': lora_a,
                    'lora_B': lora_b,
                })

    def forward(self, x):
        for i, module in enumerate(self.base_layers):
            k = str(i)
            if k in self.lora_adapters:
                base_out = module(x)
                lora_out = self.lora_adapters[k]['lora_B'](
                    self.lora_adapters[k]['lora_A'](x)
                )
                x = base_out + lora_out
            else:
                x = module(x)
        return x


# ===========================================================================
# Multi-Expert Projector
# ===========================================================================
class MultiExpertProjector(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, depth, num_experts=8, cur_task=None):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList()
        self.build_task = cur_task
        self.runtime_task_id = None

        # router 相关（外部设置）
        self.router_manager = None
        self._router_loss = None

        for _ in range(num_experts):
            modules = [nn.Linear(mm_hidden_size, hidden_size)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(hidden_size, hidden_size))
            self.experts.append(nn.Sequential(*modules))

    def _get_expert_idx(self):
        t = self.runtime_task_id if self.runtime_task_id is not None else self.build_task
        if t is None:
            return 0
        return (int(t) - 1) % self.num_experts

    def _get_task_id(self):
        t = self.runtime_task_id if self.runtime_task_id is not None else self.build_task
        return int(t) if t is not None else 1

    def forward(self, x):
        self._router_loss = None
        expert_idx = self._get_expert_idx()
        main_out = self.experts[expert_idx](x)

        # 如果有 router_manager 且在训练，计算路由损失
        if self.training and self.router_manager is not None:
            task_id = self._get_task_id()
            registered = self.router_manager.router.registered_experts
            if len(registered) > 1:
                # 确保 router 在正确的设备和 dtype 上
                self.router_manager.router.to(device=x.device, dtype=x.dtype)

                expert_outputs_pooled = {}
                for eidx in registered:
                    if eidx != expert_idx:
                        with torch.no_grad():
                            out = self.experts[eidx](x)
                    else:
                        out = self.experts[eidx](x)
                    expert_outputs_pooled[eidx] = out.detach().mean(dim=1)
                self._router_loss = self.router_manager.compute_loss(
                    expert_outputs_pooled, task_id
                )

        return main_out


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))

        if config.mm_projector_expert_num:
            return MultiExpertProjector(
                config.mm_hidden_size,
                config.hidden_size,
                mlp_depth,
                config.mm_projector_expert_num,
                config.cur_task
            )

        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
