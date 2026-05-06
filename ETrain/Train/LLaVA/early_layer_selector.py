# -*- coding: utf-8 -*-
"""
EarlyLayerSelector: 训练中途（如 10% 数据后）根据 EMA 统计信息
对各层进行 gate scoring，选出重要层，冻结不重要层的 task expert。

评分策略移植自 ana/ana.py 的 gate 策略：
    Z_lora = z(mean_lora_total)
    Z_task = z(mean_task_removed)
    Z_share = z(mean_task_share)
    g = sigmoid(GATE_GAMMA * Z_lora)
    score_gate = g * (Z_task + GATE_BETA * Z_share)
"""

import os
import re
import csv
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional


# =====================
# Utils (与 ana.py 一致)
# =====================
def _parse_layer_id(name: str) -> int:
    m = re.search(r"model\.layers\.(\d+)\.", str(name))
    return int(m.group(1)) if m else -1


def _z_score(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std(ddof=0) + 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class EarlyLayerSelector:
    """
    从模型中收集每个 AddMOELoraLinear 的 EMA 统计，
    按层聚合后用 gate scoring 选出 top-k 层。

    Parameters
    ----------
    gate_beta : float
        gate 策略中 Z_share 的权重系数，默认 1.0
    gate_gamma : float
        gate 策略中 sigmoid 内部的放大系数，默认 2.0
    selection_ratio : float
        保留的层比例（top-k），默认 0.5（保留 50%）
    output_dir : str
        中间结果输出目录
    """

    def __init__(
        self,
        gate_beta: float = 1.0,
        gate_gamma: float = 2.0,
        selection_ratio: float = 0.5,
        output_dir: str = ".",
    ):
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma
        self.selection_ratio = selection_ratio
        self.output_dir = output_dir

    def collect_layer_stats(self, model) -> List[dict]:
        """遍历模型收集所有 AddMOELoraLinear 的 EMA 统计"""
        stats = []
        for m in model.modules():
            if hasattr(m, "get_stats") and callable(m.get_stats):
                try:
                    s = m.get_stats(reset=False)
                except TypeError:
                    s = m.get_stats()
                stats.append(s)
        return stats

    def score_and_select(
        self, layer_stats: List[dict], step: int
    ) -> Tuple[List[int], List[int], dict]:
        """
        对每层计算 gate score，返回 (selected_layers, removed_layers, detail_dict)

        Returns
        -------
        selected_layers : list[int]
            保留的层 id
        removed_layers : list[int]
            冻结的层 id
        details : dict
            包含每层详细得分，用于 CSV 输出
        """
        # --- 1) 过滤、解析 layer_id ---
        records = []
        for s in layer_stats:
            lid = _parse_layer_id(s.get("layer_name", ""))
            if lid < 0:
                continue
            d_lora = s.get("delta_lora_total_ema")
            d_task = s.get("delta_task_removed_ema")
            d_share = s.get("delta_task_lora_removed_ema")
            if d_lora is None or d_task is None or d_share is None:
                continue
            records.append({
                "layer_id": lid,
                "layer_name": s["layer_name"],
                "delta_lora_total_ema": float(d_lora),
                "delta_task_removed_ema": float(d_task),
                "delta_task_lora_removed_ema": float(d_share),
            })

        if not records:
            return [], [], {}

        # --- 2) 按层聚合（同一 layer_id 下可能有多个 module，如 q/k/v/o） ---
        from collections import defaultdict
        layer_agg = defaultdict(lambda: {
            "lora_total": [], "task_removed": [], "task_share": [], "modules": []
        })
        for r in records:
            lid = r["layer_id"]
            layer_agg[lid]["lora_total"].append(r["delta_lora_total_ema"])
            layer_agg[lid]["task_removed"].append(r["delta_task_removed_ema"])
            layer_agg[lid]["task_share"].append(r["delta_task_lora_removed_ema"])
            layer_agg[lid]["modules"].append(r["layer_name"])

        layer_ids = sorted(layer_agg.keys())
        mean_lora = np.array([np.mean(layer_agg[lid]["lora_total"]) for lid in layer_ids])
        mean_task = np.array([np.mean(layer_agg[lid]["task_removed"]) for lid in layer_ids])
        mean_share = np.array([np.mean(layer_agg[lid]["task_share"]) for lid in layer_ids])
        n_modules = np.array([len(layer_agg[lid]["modules"]) for lid in layer_ids])

        # --- 3) Gate scoring ---
        Z_lora = _z_score(mean_lora)
        Z_task = _z_score(mean_task)
        Z_share = _z_score(mean_share)

        g = _sigmoid(self.gate_gamma * Z_lora)
        score_gate = g * (Z_task + self.gate_beta * Z_share)

        # 归一化到 [0, 1]
        s_min, s_max = score_gate.min(), score_gate.max()
        score_norm = (score_gate - s_min) / (s_max - s_min + 1e-12)

        # --- 4) 选层：取 top-k ---
        n_total = len(layer_ids)
        n_keep = max(1, int(np.ceil(n_total * self.selection_ratio)))
        ranked_indices = np.argsort(-score_gate)  # 降序
        selected_set = set(layer_ids[i] for i in ranked_indices[:n_keep])
        removed_set = set(layer_ids[i] for i in ranked_indices[n_keep:])

        selected_layers = sorted(selected_set)
        removed_layers = sorted(removed_set)

        # --- 5) 构建详情 ---
        details = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gate_beta": self.gate_beta,
            "gate_gamma": self.gate_gamma,
            "selection_ratio": self.selection_ratio,
            "n_total_layers": n_total,
            "n_selected": len(selected_layers),
            "n_removed": len(removed_layers),
            "selected_layers": selected_layers,
            "removed_layers": removed_layers,
            "per_layer": [],
        }

        for idx, lid in enumerate(layer_ids):
            details["per_layer"].append({
                "layer_id": lid,
                "n_modules": int(n_modules[idx]),
                "mean_lora_total": float(mean_lora[idx]),
                "mean_task_removed": float(mean_task[idx]),
                "mean_task_share": float(mean_share[idx]),
                "Z_lora": float(Z_lora[idx]),
                "Z_task": float(Z_task[idx]),
                "Z_share": float(Z_share[idx]),
                "score_gate": float(score_gate[idx]),
                "score_gate_norm": float(score_norm[idx]),
                "selected": lid in selected_set,
            })

        return selected_layers, removed_layers, details

    def save_results(self, details: dict):
        """将结果输出为 CSV + JSON，保证易读"""
        stats_dir = os.path.join(self.output_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        step = details.get("step", 0)

        # --- CSV: 每层一行，人工可读 ---
        csv_path = os.path.join(stats_dir, f"early_selection_step{step}.csv")
        fieldnames = [
            "layer_id", "n_modules",
            "mean_lora_total", "mean_task_removed", "mean_task_share",
            "Z_lora", "Z_task", "Z_share",
            "score_gate", "score_gate_norm", "selected",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 按 score_gate 降序排列
            rows = sorted(details["per_layer"], key=lambda x: x["score_gate"], reverse=True)
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})

        # --- JSON: 完整信息（含 meta） ---
        json_path = os.path.join(stats_dir, f"early_selection_step{step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        return csv_path, json_path
