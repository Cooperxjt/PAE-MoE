sh scripts/LLaVA/Train_MOE/1_Science.sh
sh scripts/LLaVA/Train_MOE/2_TextVQA.sh
sh scripts/LLaVA/Train_MOE/3_ImageNet.sh
sh scripts/LLaVA/Train_MOE/4_GQA.sh
sh scripts/LLaVA/Train_MOE/5_VizWiz.sh
sh scripts/LLaVA/Train_MOE/6_Grounding.sh
sh scripts/LLaVA/Train_MOE/7_vqav2.sh
sh scripts/LLaVA/Train_MOE/8_OCRVQA.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/ScienceQA_llava_add_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/TextVQA_llava_add_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/TextVQA_llava_add_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/ImageNet_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/ImageNet_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/ImageNet_llava_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/4_eval_gqa.sh ./checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/4_eval_gqa.sh ./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/5_eval_vizwiz.sh ./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/4_eval_gqa.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/5_eval_vizwiz.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/6_eval_grounding.sh ./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/4_eval_gqa.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/5_eval_vizwiz.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/6_eval_grounding.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/7_eval_vqav2.sh ./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/1_eval_sqa.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/2_eval_textqa.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/3_eval_ImageNet.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/4_eval_gqa.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/5_eval_vizwiz.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/6_eval_grounding.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/7_eval_vqav2.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/LLaVA/Eval/8_eval_ocrvqa.sh ./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/


