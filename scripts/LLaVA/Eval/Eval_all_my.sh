# #!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/ScienceQA_llava_add_MOE_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/TextVQA_llava_add_MOE_lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/ImageNet_llava_MoE_lora