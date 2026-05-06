#!/bin/bash
# Task 2: TextVQA - 快速测试 (max_steps=20)
# 从 Task 1 加载模型，继续训练 Task 2
# LoRA: up_proj only, r=64 | Projector LoRA: r=32 | Router: enabled | Early Selection: enabled


PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29601 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --lora_enable True --lora_r 64 --lora_alpha 128 --lora_target_modules up_proj --mm_projector_lr 2e-5 \
    --proj_lora_r 32 \
    --cur_task 2 \
    --task_layer_map_file ./layer_map/top_32.json \
    --model_name_or_path ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
    --previous_task_model_path ./checkpoints/LLaVA/CoIN/ScienceQA_test_task1 \
    --pretrain_mm_mlp_adapter ./llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --version $PROMPT_VERSION \
    --data_path ./playground/Instructions_Original/TextVQA/train.json \
    --image_folder ./cl_dataset \
    --vision_tower ./clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_projector_expert_num 8 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/LLaVA/CoIN/TextVQA_test_task2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --max_steps 20 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --early_selection_enabled True \
    --early_selection_ratio 0.1 \
    --early_selection_top_k 0.5
