# PAE-MoE: Adaptive Expert Expansion for Efficient Multimodal Continual Instruction Tuning

## Abstract
    
    Mixture-of-Experts architectures have demonstrated advantages in Multimodal Continual Instruction Tuning due to their parameter isolation and sparse activation properties. However, most existing methods fully isolate task experts, which suppresses cross-task interference but simultaneously severs shared representations. This absence not only hinders knowledge transfer, but also deprives the model of a unified reference for assessing whether expert expansion is necessary at each layer, resulting in indiscriminate layer-wise expansion and redundant parameter growth. To address this, we propose Probing-Assessment-Expanding (PAE) framework, which introduces a shared expert trained continuously across all tasks, unifying knowledge accumulation and structural assessment within a seamless training process. The shared expert encodes general representations while serving as an endogenous reference to guide adaptive expansion, thereby enabling selective parameter growth without additional offline evaluation. Furthermore, we design the Adaptation-Aware Task Selector (AATS) by leveraging the feature diversity induced by LoRA adapters in the visual projection layer. This enables automatic expert selection under task-agnostic inference while supporting continual visual adaptation with minimal overhead. On the CoIN benchmark, our method achieves state-of-the-art performance across multiple metrics while maintaining superior parameter efficiency. Our code is publicly available at https://github.com/Cooperxjt/PAE-MoE.

## Install

1. Clone this repository and navigate to folder

2. Install Package

```
conda create -n coin python=3.10 -y
conda activate pae
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

This repo is based on [LLaVA](https://github.com/haotian-liu/LLaVA).
If you meet a problem, maybe you could find some solutions in issuses.

## Dataset

We use the dataset of CoIN, please download the images and instuctions from https://github.com/zackschen/CoIN

After downloading all of datasets, organize the data as follows:

```
├── COCO2014
│   └── train2014
├── GQA
│   └── images
├── OCR-VQA
│   └── images
├── TextVQA
│   └── train_images
│   └── test_images
```

Then, please download the instructions from our datasets path: [CoIN_Dataset](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main)
then, organize the instructions as follows:

```
├── Instruction_Original
│   └── GQA
│       └── train.json
│       └── test.json
│   └── ScienceQA
│       └── train.json
│       └── test.json
├── Instruction_Type2
│   └── GQA
│       └── train.json
│       └── test.json
```

## Instruction Tuning

First, downloading the pretrained projectors in [LLaVA Model_Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

Setting `pretrain_mm_mlp_adapter` to the projector path.
You could modify the `deepspeed config` to change the deepspeed config.

We provide the scripts of our train order in `scripts/*/Train`.
Note, the `output_dir` of the previous script is the `previous_task_model_path` of the next training process.
Then, you could tune these datasets in your order.

## Evaluation

We have prepared the scripts to evaluate the trained model in `scripts/*/Eval`.

These scripts will evalute the trained model and create the prompts (`prompt_to_eval.json`) for evaluating the general knowldege.

To evaluate the general knowldege, you could add the result path to `scripts/Eval_GeneralKnowledge/eval_prompt_slim.sh` and run it, this script file will output a score to indicate the general knowledge.
