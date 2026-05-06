#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

def smooth(logits, temp, dim):
    """
    对logits进行温度缩放平滑处理
    Args:
        logits: 模型输出的预测分布
        temp: 温度参数，控制分布的平滑程度（>1更平滑，<1更尖锐）
        dim: 归一化的维度
    Returns:
        平滑后的概率分布
    """
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)

def modified_kl_div(old, new):
    """
    修改的KL散度计算，用于LWF（Learning Without Forgetting）损失
    Args:
        old: 旧模型的输出分布
        new: 新模型的输出分布
    Returns:
        KL散度损失值
    """
    return -torch.mean(torch.sum(old * torch.log(new), 1))

class LlavaConfig(LlamaConfig):
    """LLaVA模型的配置类，继承自LlamaConfig"""
    model_type = "llava"  # 注册新的模型类型


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """
    LLaVA的多模态语言模型，结合了文本处理和视觉特征融合能力
    继承关系: LlavaMetaModel (多模态功能) + LlamaModel (文本生成能力)
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        """初始化LLaVA模型，设置多模态处理能力"""
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    LLaVA语言模型的因果语言建模版本，支持文本生成和多模态输入
    核心功能：文本生成 + 图像理解 + 持续学习（EWC/LWF）
    """
    config_class = LlavaConfig

    def __init__(self, config):
        """初始化因果语言建模版本的LLaVA模型"""
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)  # 核心多模态模型
        self.pretraining_tp = config.pretraining_tp  # 预训练张量并行配置
        self.vocab_size = config.vocab_size  # 词汇表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 语言模型头
        self.soft = torch.nn.Softmax(dim=1)  # Softmax层，用于概率计算
        
        # Initialize weights and apply final processing
        self.post_init()  # 权重初始化和最终处理

    def get_model(self):
        """获取内部模型实例"""
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,  # 多模态输入：图像特征
        return_dict: Optional[bool] = None,
        **kwargs,  # 额外参数，可能包含question_ids等
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        模型前向传播，支持多模态输入和持续学习损失计算
        主要流程：多模态输入准备 → 文本生成 → 持续学习损失计算
        """

        # 步骤1: 多模态输入准备 - 将图像特征与文本token结合
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            
        # 步骤2: 调用父类的文本生成前向传播
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # 步骤3: 弹性权重巩固(EWC)损失计算 - 防止灾难性遗忘
        if hasattr(self,'EWC') and self.EWC and hasattr(self,'fisher'):
            ewc_loss = 0
            # 遍历所有可训练参数，计算EWC正则化项
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    dev = p.device
                    # EWC损失 = λ * Fisher信息 * (当前参数 - 最优参数)^2
                    l = self.EWC_lambda * self.fisher[n].to(dev) * (p.data - self.optpar[n].to(dev)).pow(2)
                    ewc_loss += l.sum()  # 累加所有参数的EWC损失
            output['loss'] += ewc_loss  # 将EWC损失加入总损失

        # 步骤4: 学习无遗忘(LWF)损失计算 - 保持旧任务性能
        if  hasattr(self, 'LWF') and self.LWF and hasattr(self, 'previous_logits'):
            lwf_loss = []
            previous_keys = self.previous_logits.keys()
            # 遍历当前batch中的每个样本
            for index, question_id in enumerate(kwargs['question_ids']):
                if question_id in previous_keys:
                    # 获取之前任务中该样本的logits
                    previous_logits = self.previous_logits[question_id]
                    current_logits = output['logits'][index]
                    # 对齐序列长度
                    short_index = min(len(previous_logits), len(current_logits))
                    previous_logits = previous_logits[:short_index]
                    current_logits = current_logits[:short_index]
                    # 计算修改的KL散度损失
                    lwf_loss.append(modified_kl_div(
                        smooth(self.soft(previous_logits).to(current_logits.device), 2, 1),
                        smooth(self.soft(current_logits), 2, 1)
                    ))
            if len(lwf_loss) > 0:
                # 将LWF损失加入总损失
                output['loss'] += self.LWF_lambda * torch.stack(lwf_loss, dim=0).sum(0)
        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """准备生成任务的输入，支持多模态输入"""
        images = kwargs.pop("images", None)  # 提取图像参数
        # 调用父类方法准备基础输入
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images  # 添加图像特征到输入中
        return _inputs

# 注册LLaVA模型类型，使其能够被transformers库识别
AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
