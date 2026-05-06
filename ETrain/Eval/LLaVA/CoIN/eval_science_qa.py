import argparse
import json
import os
import re
import random


def get_args():
    """解析命令行参数，获取ScienceQA评估所需的配置参数
    
    Returns:
        argparse.Namespace: 包含以下参数的命名空间对象:
            --base-dir: ScienceQA数据集的基础目录路径
            --result-file: 模型预测结果的JSONL文件路径
            --output-file: 评估详细结果的输出文件路径
            --output-result: 评估统计结果的输出文件路径
            --split: 使用的数据集划分（test/val/train）
            --options: 选择题选项列表（默认为A-E）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default = './cl_dataset/ScienceQA')
    parser.add_argument('--result-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/merge.jsonl')
    parser.add_argument('--output-file', type=str, default= './results/CoIN/Qwen/ScienceQA/Finetune/output.jsonl')
    parser.add_argument('--output-result', type=str, default= './results/CoIN/Qwen/ScienceQA/Finetune/output_result.jsonl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    """将模型预测结果转换为标准的图像描述格式
    
    Args:
        results (list): 包含预测结果的字典列表，每个字典包含'question_id'和'text'字段
        
    Returns:
        list: 格式化的图像描述列表，包含'image_id'和'caption'字段
    """
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """从预测文本中提取选项索引
    
    该函数将模型输出的文本预测转换为对应的选项索引（如将'A'转换为0）
    
    Args:
        prediction (str): 模型的原始预测文本
        choices (list): 问题的选项列表
        options (list): 可接受的选项字符（如['A','B','C','D','E']）
        
    Returns:
        int: 预测对应的选项索引，如果无法解析则返回-1，并在调试模式下随机选择一个选项
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    """ScienceQA数据集评估的主函数
    
    主要流程：
    1. 解析命令行参数和数据文件
    2. 加载问题和预测结果
    3. 对每个问题解析模型预测
    4. 计算准确率并输出结果
    """
    args = get_args()

    # 加载数据集分割和问题数据
    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    # 初始化结果存储结构
    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    # 处理每个问题的预测结果
    for prob_id, prob in split_problems.items():
        # 获取模型预测，如果缺失则标记为失败
        if prob_id not in predictions:
            pred = {'text': 'FAILED', 'prompt': 'Unknown'}
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text']

        # 多种方式解析预测文本中的答案
        if pred_text in args.options:
            # 情况1：预测文本直接就是选项（如"A"）
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            # 情况2：预测文本以"选项. "开头（如"A. 答案描述"）
            answer = pred_text[0]
        else:
            # 情况3：使用正则表达式提取单个字母选项
            pattern = re.compile(r'\b(\w)\b')
            res = pattern.findall(pred_text)
            if len(res) > 0:
                answer = res[0].upper()  # 'A', 'B', ...
            else:
                answer = "FAILED"

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        # 构建详细的分析记录
        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,  # 解析后的最终答案
            'ground_truth': args.options[prob['answer']],  # 真实答案
            'question': pred['prompt'],  # 问题文本
            'pred': pred_text,  # 原始预测文本
            'is_multimodal': '<image>' in pred['prompt'],  # 是否为多模态问题
        }

        # 存储评估结果
        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        # 分类正确和错误的答案
        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    # 计算准确率统计
    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    # multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    # multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    # multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    # 输出准确率结果
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')

    # 更新统计结果
    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
