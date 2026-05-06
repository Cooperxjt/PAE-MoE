# -*- encoding: utf-8 -*-
"""
Projector-based Task Router for Task-Agnostic Continual Instruction Tuning

核心思想：
    任务判定的信号直接来自多专家投影器（MultiExpertProjector）的输出。
    每个 projector expert 在对应任务上训练过，对"属于自己任务"的输入产生
    更"兼容"的投影。路由器通过对比各 expert 输出的兼容性评分来判断任务归属。

    分类能力不是外加的独立网络，而是多专家投影器结构的自然延伸。

架构：
    vision_feat
        │
        ├──→ Expert_0(feat) → pool → scorer_0 → s_0  ─┐
        ├──→ Expert_1(feat) → pool → scorer_1 → s_1   │
        ├──→ ...                                        ├→ CE loss / argmax → task_id
        └──→ Expert_N(feat) → pool → scorer_N → s_N  ─┘
                                                        (可选) + text_scorer → 辅助信号

为什么不需要数据回放：
    训练 Task k 时，当前 batch 同时过所有 N 个 expert：
    - Expert_k（当前训练的）产生"匹配"的投影 → 正样本
    - 其他 Expert（冻结的/未训练的）产生"不匹配"的投影 → 负样本
    - CE loss 自然在 N 个评分之间做竞争
    - 旧 scorer 冻结 + 旧 expert 冻结 → 旧任务判断不会退化

    整个系统与 MoE 框架一致：冻结 + 增量扩展，无需任何回放。

使用流程：
    训练 Task k：
        1. manager.prepare_new_task(task_id=k, expert_idx=(k-1)%N)
        2. 每个 batch:
             expert_outputs = {i: projector.experts[i](vision_feat) for i in range(N)}
             cls_loss = manager.compute_loss(expert_outputs, task_id=k, text_feat=...)
             total_loss = main_loss + cls_loss
        3. manager.save(save_dir)

    推理（task_id 未知）：
        1. manager = ProjectorTaskRouterManager.load(save_dir)
        2. expert_outputs = {i: projector.experts[i](vision_feat) for i in range(N)}
        3. pred_task, pred_expert = manager.predict(expert_outputs, text_feat=...)
        4. set_runtime_task_id(model, pred_task)
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ProjectorTaskRouter: 基于多专家投影器的任务路由器
# =============================================================================
class ProjectorTaskRouter(nn.Module):
    """
    基于多专家投影器输出的任务路由器。

    每个 projector expert 绑定一个轻量 scorer：
        scorer_k(pool(Expert_k(x))) → 标量兼容性评分

    训练时：CE loss 让当前任务的 scorer 在竞争中胜出。
    推理时：所有 scorer 评分 → argmax → 预测任务。

    与 MultiExpertProjector 的耦合关系：
        - 路由器的特征 100% 来自 projector expert 输出
        - 没有独立的特征提取网络
        - scorer 与 expert 一一绑定，同生命周期（一起创建、一起冻结）
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        scorer_hidden: int = 64,
        text_dim: Optional[int] = None,
    ):
        """
        Args:
            hidden_size:    projector 输出维度（= LLM hidden_size, e.g. 4096）
            num_experts:    projector expert 数量
            scorer_hidden:  scorer 内部隐藏维度（很小，每个 scorer 仅数千参数）
            text_dim:       文本嵌入维度（可选，用于辅助信号）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.scorer_hidden = scorer_hidden

        # --- 每个 expert 一个兼容性评分头 ---
        # key = str(expert_idx), value = small MLP → scalar
        self.scorers = nn.ModuleDict()

        # --- 可选：文本辅助评分 ---
        self.use_text = text_dim is not None
        self.text_dim = text_dim
        if self.use_text:
            # 每个 expert 也有一个文本评分头
            self.text_scorers = nn.ModuleDict()

        # 记录 expert_idx → task_id 的映射
        self._expert_to_task: Dict[int, int] = {}
        self._registered_experts: List[int] = []

    @property
    def registered_experts(self) -> List[int]:
        return list(self._registered_experts)

    def expert_to_task(self, expert_idx: int) -> Optional[int]:
        return self._expert_to_task.get(int(expert_idx))

    def task_to_expert(self, task_id: int) -> Optional[int]:
        for eidx, tid in self._expert_to_task.items():
            if tid == task_id:
                return eidx
        return None

    def register_expert(self, expert_idx: int, task_id: int):
        """
        注册一个 expert-task 绑定，并创建对应的 scorer。

        Args:
            expert_idx: projector expert 索引 (0-based)
            task_id:    对应的任务 ID (1-based)
        """
        k = str(int(expert_idx))
        if k not in self.scorers:
            self.scorers[k] = nn.Sequential(
                nn.Linear(self.hidden_size, self.scorer_hidden),
                nn.GELU(),
                nn.Linear(self.scorer_hidden, 1),
            )
            if self.use_text:
                self.text_scorers[k] = nn.Sequential(
                    nn.Linear(self.text_dim, self.scorer_hidden),
                    nn.GELU(),
                    nn.Linear(self.scorer_hidden, 1),
                )
            self._registered_experts.append(int(expert_idx))

        self._expert_to_task[int(expert_idx)] = int(task_id)

    def freeze_old_scorers(self, current_expert_idx: int):
        """冻结旧 expert 的 scorer，只训练当前 expert 的 scorer。"""
        cur = str(int(current_expert_idx))
        for k, scorer in self.scorers.items():
            for p in scorer.parameters():
                p.requires_grad = (k == cur)
        if self.use_text:
            for k, scorer in self.text_scorers.items():
                for p in scorer.parameters():
                    p.requires_grad = (k == cur)

    def forward(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        text_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        对所有已注册 expert 的投影输出进行兼容性评分。

        Args:
            expert_outputs_pooled: {expert_idx: [B, hidden_size]}
                                   各 expert 输出经均值池化后的结果
            text_feat:             [B, text_dim] 可选文本特征
        Returns:
            scores:         [B, num_registered]  兼容性评分
            expert_indices: List[int]             对应的 expert 索引
        """
        scores = []
        expert_indices = []

        for eidx in self._registered_experts:
            k = str(eidx)
            if eidx not in expert_outputs_pooled:
                continue

            # 视觉兼容性评分
            s = self.scorers[k](expert_outputs_pooled[eidx])  # [B, 1]

            # 可选：叠加文本评分
            if self.use_text and text_feat is not None and k in self.text_scorers:
                t_s = self.text_scorers[k](text_feat)  # [B, 1]
                s = s + t_s

            scores.append(s)
            expert_indices.append(eidx)

        scores = torch.cat(scores, dim=-1)  # [B, num_registered]
        return scores, expert_indices

    @torch.no_grad()
    def predict(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        text_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """
        预测任务 ID 和对应的 expert 索引。

        Returns:
            (predicted_task_id, predicted_expert_idx)
        """
        self.eval()
        scores, expert_indices = self.forward(expert_outputs_pooled, text_feat)

        # batch 级别：取每个样本的最佳 expert，再众数投票
        best_per_sample = scores.argmax(dim=-1)  # [B]
        best_local = best_per_sample.mode().values.item()
        best_expert = expert_indices[best_local]
        best_task = self._expert_to_task.get(best_expert, best_expert + 1)

        return int(best_task), int(best_expert)

    @torch.no_grad()
    def predict_with_confidence(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        text_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int, float]:
        """
        预测任务 ID，附带置信度。

        Returns:
            (predicted_task_id, predicted_expert_idx, confidence)
        """
        self.eval()
        scores, expert_indices = self.forward(expert_outputs_pooled, text_feat)
        probs = F.softmax(scores, dim=-1)  # [B, num_registered]

        avg_probs = probs.mean(dim=0)  # [num_registered]
        best_idx = avg_probs.argmax().item()
        confidence = avg_probs[best_idx].item()

        best_expert = expert_indices[best_idx]
        best_task = self._expert_to_task.get(best_expert, best_expert + 1)

        return int(best_task), int(best_expert), float(confidence)


# =============================================================================
# ProjectorTaskRouterManager: 编排器
# =============================================================================
class ProjectorTaskRouterManager:
    """
    编排路由器的训练和推理。

    训练阶段：
        1. prepare_new_task(task_id, expert_idx) → 注册 + 冻结旧 scorer
        2. compute_loss(expert_outputs, task_id)  → CE over scorer 竞争
        3. save(dir)

    推理阶段：
        1. load(dir)
        2. predict(expert_outputs) → (task_id, expert_idx)

    为什么不需要回放：
        训练 Task k 时，所有 N 个 expert 同时处理当前 batch：
        - scorer_k 是可训练的，学习给 expert_k 的输出打高分
        - scorer_j (j<k) 是冻结的，对 expert_j 在当前数据上的输出给出固定分数
        - CE loss 让 scorer_k 学会"赢过"冻结的 scorer_j
        - 旧 expert 冻结 → 旧 scorer 的判断对旧数据始终有效
    """

    def __init__(
        self,
        router: ProjectorTaskRouter,
        cls_weight: float = 0.1,
    ):
        """
        Args:
            router:     ProjectorTaskRouter 实例
            cls_weight: 路由损失权重
        """
        self.router = router
        self.cls_weight = cls_weight

    # -------------------- 训练阶段 API --------------------

    def prepare_new_task(self, task_id: int, expert_idx: int):
        """
        新任务训练前调用。

        Args:
            task_id:    任务 ID (e.g. 1-based)
            expert_idx: 对应的 projector expert 索引 (0-based)
                        通常 expert_idx = (task_id - 1) % num_experts
        """
        self.router.register_expert(expert_idx, task_id)
        self.router.freeze_old_scorers(expert_idx)
        print(
            f"[ProjectorTaskRouter] Prepared task {task_id} → expert {expert_idx}, "
            f"registered: {self.router.registered_experts}, "
            f"mapping: {self.router._expert_to_task}"
        )

    def compute_loss(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        task_id: int,
        text_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算路由辅助损失。

        训练循环中这样使用：
            # 1. 运行所有 projector expert
            expert_outputs = {}
            for i in range(num_experts):
                out = projector.experts[i](vision_feat)   # [B, seq, hidden]
                expert_outputs[i] = out.mean(dim=1)       # [B, hidden]

            # 2. 计算路由损失
            route_loss = manager.compute_loss(expert_outputs, task_id, text_feat)
            total_loss = main_loss + route_loss

        Args:
            expert_outputs_pooled: {expert_idx: [B, hidden_size]}
            task_id:               当前任务 ID
            text_feat:             [B, text_dim] 可选
        Returns:
            加权后的路由损失
        """
        device = next(iter(expert_outputs_pooled.values())).device

        # detach expert 输出：路由损失不应影响 expert 权重
        detached = {k: v.detach() for k, v in expert_outputs_pooled.items()}
        if text_feat is not None:
            text_feat = text_feat.detach()

        self.router.train()
        scores, expert_indices = self.router.forward(detached, text_feat)
        # scores: [B, num_registered]

        # 目标：当前任务对应的 expert 应获得最高分
        target_expert = self.router.task_to_expert(task_id)
        if target_expert is None or target_expert not in expert_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)

        target_idx = expert_indices.index(target_expert)
        B = scores.size(0)
        labels = torch.full((B,), target_idx, dtype=torch.long, device=device)

        loss = F.cross_entropy(scores, labels)
        return self.cls_weight * loss

    # -------------------- 推理阶段 API --------------------

    @torch.no_grad()
    def predict(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        text_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """
        预测 task_id 和 expert_idx。

        Returns:
            (predicted_task_id, predicted_expert_idx)
        """
        return self.router.predict(expert_outputs_pooled, text_feat)

    @torch.no_grad()
    def predict_with_confidence(
        self,
        expert_outputs_pooled: Dict[int, torch.Tensor],
        text_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int, float]:
        """
        预测 task_id，附带置信度。

        Returns:
            (predicted_task_id, predicted_expert_idx, confidence)
        """
        return self.router.predict_with_confidence(expert_outputs_pooled, text_feat)

    # -------------------- 序列化 --------------------

    def save(self, save_dir: str):
        """保存路由器到目录。"""
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "state_dict": self.router.state_dict(),
                "registered_experts": self.router._registered_experts,
                "expert_to_task": self.router._expert_to_task,
                "config": {
                    "hidden_size": self.router.hidden_size,
                    "num_experts": self.router.num_experts,
                    "scorer_hidden": self.router.scorer_hidden,
                    "text_dim": self.router.text_dim,
                    "use_text": self.router.use_text,
                },
            },
            os.path.join(save_dir, "projector_task_router.pt"),
        )

        with open(os.path.join(save_dir, "router_config.json"), "w") as f:
            json.dump({"cls_weight": self.cls_weight}, f, indent=2)

        print(f"[ProjectorTaskRouter] Saved to {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        device: torch.device = torch.device("cpu"),
    ) -> "ProjectorTaskRouterManager":
        """从目录加载。"""
        ckpt = torch.load(
            os.path.join(save_dir, "projector_task_router.pt"),
            map_location=device,
        )
        cfg = ckpt["config"]

        router = ProjectorTaskRouter(
            hidden_size=cfg["hidden_size"],
            num_experts=cfg["num_experts"],
            scorer_hidden=cfg["scorer_hidden"],
            text_dim=cfg.get("text_dim"),
        )

        # 重建注册结构
        for eidx in ckpt["registered_experts"]:
            tid = ckpt["expert_to_task"][eidx]
            router.register_expert(eidx, tid)

        router.load_state_dict(ckpt["state_dict"])
        router.to(device)
        router.eval()

        with open(os.path.join(save_dir, "router_config.json"), "r") as f:
            mgr_config = json.load(f)

        print(
            f"[ProjectorTaskRouter] Loaded from {save_dir}, "
            f"experts={router.registered_experts}, "
            f"mapping={router._expert_to_task}"
        )

        return cls(router=router, cls_weight=mgr_config.get("cls_weight", 0.1))


# =============================================================================
# 辅助工具函数
# =============================================================================
def pool_features(
    features: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    对序列特征做均值池化。

    Args:
        features:       [B, seq_len, dim]
        attention_mask:  [B, seq_len]  可选，1=有效, 0=padding
    Returns:
        pooled: [B, dim]
    """
    if attention_mask is None:
        return features.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).float()
    masked = features * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return pooled


def run_all_experts(
    projector,
    vision_feat: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    运行所有 projector expert 并返回池化后的输出。

    Args:
        projector:   MultiExpertProjector 实例
        vision_feat: [B, seq_v, mm_hidden_size]  视觉编码器输出
    Returns:
        {expert_idx: [B, hidden_size]}  池化后的各 expert 输出
    """
    outputs = {}
    for i, expert in enumerate(projector.experts):
        out = expert(vision_feat)        # [B, seq_v, hidden_size]
        outputs[i] = out.mean(dim=1)     # [B, hidden_size]
    return outputs


@torch.no_grad()
def extract_text_features(
    input_ids: torch.Tensor,
    embed_tokens_fn,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    提取池化后的文本嵌入特征。

    Args:
        input_ids:       [B, seq_t]  文本 token IDs
        embed_tokens_fn: 文本嵌入函数, e.g. model.model.embed_tokens
        attention_mask:  [B, seq_t]  可选
    Returns:
        text_feat: [B, text_dim]
    """
    text_emb = embed_tokens_fn(input_ids).detach()
    if attention_mask is None:
        attention_mask = (input_ids != 0).long()
    return pool_features(text_emb, attention_mask)


def create_projector_task_router(
    hidden_size: int,
    num_experts: int,
    scorer_hidden: int = 64,
    text_dim: Optional[int] = None,
    cls_weight: float = 0.1,
    load_from: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> ProjectorTaskRouterManager:
    """
    工厂函数：创建或加载 ProjectorTaskRouterManager。

    用法示例：
        # 全新创建
        manager = create_projector_task_router(
            hidden_size=4096,  # LLaMA hidden size (projector 输出维度)
            num_experts=8,
            text_dim=4096,     # 可选
        )

        # 从上一轮训练加载
        manager = create_projector_task_router(
            hidden_size=4096,
            num_experts=8,
            load_from="/path/to/prev_task/router",
        )
    """
    if load_from is not None and os.path.exists(
        os.path.join(load_from, "projector_task_router.pt")
    ):
        print(f"[ProjectorTaskRouter] Loading from {load_from}")
        return ProjectorTaskRouterManager.load(load_from, device=device)

    print(
        f"[ProjectorTaskRouter] Creating new router: "
        f"hidden_size={hidden_size}, num_experts={num_experts}, "
        f"scorer_hidden={scorer_hidden}, text_dim={text_dim}"
    )
    router = ProjectorTaskRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        scorer_hidden=scorer_hidden,
        text_dim=text_dim,
    ).to(device)

    return ProjectorTaskRouterManager(router=router, cls_weight=cls_weight)
