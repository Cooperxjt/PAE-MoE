import os, re, json, time, math
from transformers import TrainerCallback
from early_layer_selector import EarlyLayerSelector


class MOELoraStatsCallback(TrainerCallback):
    """
    定期把 AddMOELoraLinear 的统计信息写入本地 jsonl。
    每个 step 会写很多行（每层一行），所以建议 log_every_steps>=20。
    """

    def __init__(
        self,
        output_dir: str,
        log_every_steps: int = 50,
        reset_after_log: bool = False,
        filename: str = "moe_lora_stats.jsonl",
        include_train_log: bool = True,  # 是否把 loss/lr 等也一起写进去
    ):
        self.output_dir = output_dir
        self.log_every_steps = int(log_every_steps)
        self.reset_after_log = bool(reset_after_log)
        self.include_train_log = bool(include_train_log)

        self.stats_dir = os.path.join(output_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.path = os.path.join(self.stats_dir, filename)

        self._last_logged_step = -1

    def _is_rank0(self, args):
        # HF Trainer/Deepspeed/DDP 通用：process_index==0 是 rank0
        return getattr(args, "process_index", 0) == 0

    def _append_jsonl(self, rows):
        # 追加写文件
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _collect_layer_stats(self, model, reset=False):
        """
        不依赖 model.collect_layer_stats：直接遍历 modules 找 get_stats()
        这样你不用强行修改外层包装结构。
        """
        out = []
        for m in model.modules():
            if hasattr(m, "get_stats") and callable(m.get_stats):
                try:
                    out.append(m.get_stats(reset=reset))
                except TypeError:
                    # 防止你 get_stats() 没有 reset 参数
                    out.append(m.get_stats())
        return out

    def on_step_end(self, args, state, control, **kwargs):
        # 只在 rank0 写
        if not self._is_rank0(args):
            return
        
        step = int(state.global_step)
        if step == self._last_logged_step:
            return
        self._last_logged_step = step

        model = kwargs.get("model", None)
        if model is None:
            return

        # 1) 收集每层 stats（每层一个 dict）
        layer_stats = self._collect_layer_stats(model, reset=self.reset_after_log)

        # 2) 收集 Trainer 自带的 log（loss/lr 等）
        # 注意：state.log_history 是历史列表，最后一个通常是最新日志，但不保证每 step 都有
        last_log = (
            state.log_history[-1]
            if (self.include_train_log and len(state.log_history) > 0)
            else {}
        )

        now = time.time()
        rows = []
        for s in layer_stats:
            # s 里应包含 layer_name 和各种 *_ema
            row = {
                "time": now,
                "step": step,
                # 你如果有 task_id，建议也写进去（可在外面通过闭包或环境变量传入）
                # "task_id": ...,
                **({"train_log": last_log} if last_log else {}),
                **s,
            }
            rows.append(row)

        # 3) 写文件
        self._append_jsonl(rows)


def count_trainable_params(model):
    """
    统计模型可训练参数量，返回 (trainable, total)。
    兼容 DeepSpeed ZeRO-3（参数被分片时，用 ds_numel 获取真实大小）。
    """
    trainable = 0
    total = 0
    for p in model.parameters():
        # DeepSpeed ZeRO-3 下 p.numel() 可能是分片大小，用 ds_numel 获取原始大小
        n = getattr(p, "ds_numel", p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


class EarlySelectionCallback(TrainerCallback):
    """
    在训练到 selection_ratio（默认 10%）步数时，自动执行层选择：
    1. 收集所有 AddMOELoraLinear 的 EMA 统计
    2. 用 gate scoring 计算各层重要性
    3. 冻结不重要层的 task expert（只保留 shared）
    4. 统计探测阶段 vs 正式训练阶段可训练参数量及比值
    5. 输出 CSV + JSON 中间结果

    用法：在 train.py 中 trainer.add_callback(EarlySelectionCallback(...))
    """

    def __init__(
        self,
        output_dir: str,
        cur_task: int,
        adapter_name: str = "default",
        selection_ratio: float = 0.1,       # 在 10% 步数时触发
        top_k_ratio: float = 0.5,           # 保留 top 50% 的层
        gate_beta: float = 1.0,
        gate_gamma: float = 2.0,
    ):
        self.output_dir = output_dir
        self.cur_task = int(cur_task)
        self.adapter_name = adapter_name
        self.selection_ratio = selection_ratio
        self.top_k_ratio = top_k_ratio
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma

        self._triggered = False
        self._trigger_step = None

        self.selector = EarlyLayerSelector(
            gate_beta=gate_beta,
            gate_gamma=gate_gamma,
            selection_ratio=top_k_ratio,
            output_dir=output_dir,
        )

    def _is_rank0(self, args):
        return getattr(args, "process_index", 0) == 0

    @staticmethod
    def _parse_layer_id(name: str) -> int:
        m = re.search(r"model\.layers\.(\d+)\.", str(name))
        return int(m.group(1)) if m else -1

    def on_step_end(self, args, state, control, **kwargs):
        if self._triggered:
            return

        # 计算触发步数（向上取整）
        max_steps = state.max_steps
        if max_steps <= 0:
            return
        if self._trigger_step is None:
            self._trigger_step = int(math.ceil(max_steps * self.selection_ratio))

        step = int(state.global_step)
        if step < self._trigger_step:
            return

        # ============ 触发 Early Selection ============
        self._triggered = True

        model = kwargs.get("model", None)
        if model is None:
            return

        # 0) 统计探测阶段（冻结前）的可训练参数
        probe_trainable, total_params = count_trainable_params(model)

        # 1) 收集统计
        layer_stats = self.selector.collect_layer_stats(model)
        if not layer_stats:
            if self._is_rank0(args):
                print(f"[EarlySelection] step={step}: no layer stats found, skipping.")
            return

        # 2) 评分 & 选层
        selected, removed, details = self.selector.score_and_select(layer_stats, step)

        if self._is_rank0(args):
            print(f"\n{'='*60}")
            print(f"[EarlySelection] Triggered at step {step}/{max_steps}")
            print(f"  gate_beta={self.gate_beta}, gate_gamma={self.gate_gamma}")
            print(f"  top_k_ratio={self.top_k_ratio}")
            print(f"  Selected layers ({len(selected)}): {selected}")
            print(f"  Removed  layers ({len(removed)}): {removed}")
            print(f"{'='*60}\n")

        # 3) 冻结不重要层的 task expert
        removed_set = set(removed)
        n_frozen = 0
        for m in model.modules():
            if hasattr(m, "freeze_task_expert") and hasattr(m, "layer_name"):
                lid = self._parse_layer_id(m.layer_name)
                if lid in removed_set:
                    m.freeze_task_expert(self.adapter_name, self.cur_task)
                    n_frozen += 1

        # 4) 统计正式训练阶段（冻结后）的可训练参数
        formal_trainable, _ = count_trainable_params(model)

        # 计算比值
        if formal_trainable > 0:
            ratio = probe_trainable / formal_trainable
        else:
            ratio = float("inf")

        if self._is_rank0(args):
            print(f"[EarlySelection] Frozen {n_frozen} modules in {len(removed)} removed layers.")
            print(f"[EarlySelection] Trainable Parameters:")
            print(f"  Probe phase (before selection):  {probe_trainable:>12,}")
            print(f"  Formal phase (after selection):  {formal_trainable:>12,}")
            print(f"  Total model parameters:          {total_params:>12,}")
            print(f"  Reduction: formal = probe / {ratio:.2f}")
            print(f"  Saved: {probe_trainable - formal_trainable:,} params ({(1 - formal_trainable / probe_trainable) * 100:.1f}%)")

            # 5) 将参数统计写入 details，一起保存
            details["param_stats"] = {
                "probe_trainable": probe_trainable,
                "formal_trainable": formal_trainable,
                "total_params": total_params,
                "ratio_probe_div_formal": round(ratio, 4),
                "saved_params": probe_trainable - formal_trainable,
                "saved_percent": round((1 - formal_trainable / probe_trainable) * 100, 2) if probe_trainable > 0 else 0,
            }

            # 6) 保存中间结果（只在 rank0）
            csv_path, json_path = self.selector.save_results(details)
            print(f"[EarlySelection] Results saved:")
            print(f"  CSV:  {csv_path}")
            print(f"  JSON: {json_path}")
