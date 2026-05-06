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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower  # 视觉编码器构建器
from .multimodal_projector.builder import build_vision_projector  # 视觉投影器构建器

from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:
    """
    LLaVA多模态模型的基础元类，负责管理视觉编码器和投影器
    提供了多模态模型的核心组件管理功能
    """

    def __init__(self, config):
        """初始化多模态模型，设置视觉组件"""
        super(LlavaMetaModel, self).__init__(config)

        # 如果配置中指定了视觉编码器，则构建视觉组件
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)  # 构建视觉编码器（延迟加载）
            self.mm_projector = build_vision_projector(config)  # 构建视觉投影器

    def get_vision_tower(self):
        """获取视觉编码器实例，处理FSDP分布式训练的情况"""
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]  # 如果是列表，取第一个（FSDP情况）
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        初始化视觉模块，这是多模态模型设置的核心方法
        Args:
            model_args: 模型参数配置
            fsdp: 全分片数据并行配置（可选）
        """
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer  # 选择视觉编码器的层
        mm_vision_select_feature = model_args.mm_vision_select_feature  # 选择特征类型
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter  # 预训练投影器路径
        mm_projector_expert_num = model_args.mm_projector_expert_num
        cur_task = model_args.cur_task
        
        self.config.mm_vision_tower = vision_tower  # 更新配置

        # 步骤1: 初始化视觉编码器
        if self.get_vision_tower() is None:
            # 如果视觉编码器不存在，创建新的
            vision_tower = build_vision_tower(model_args)
            
            # 处理FSDP分布式训练的情况
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]  # FSDP需要封装为列表
            else:
                self.vision_tower = vision_tower
        else:
            # 如果视觉编码器已存在，加载模型权重
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()  # 加载预训练权重

        # 步骤2: 配置多模态投影器参数
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')  # 投影器类型
        self.config.mm_hidden_size = vision_tower.hidden_size  # 视觉特征维度
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_projector_expert_num = mm_projector_expert_num
        self.config.cur_task = cur_task

        # 步骤3: 初始化或解冻视觉投影器
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)  # 创建新的投影器
        else:
            # 如果投影器已存在（可能被LoRA冻结），确保参数可训练
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # 步骤4: 加载预训练的投影器权重（如果提供）
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                """从权重字典中提取指定前缀的权重"""
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            projector_weights = get_w(mm_projector_weights, 'mm_projector')

            if hasattr(self.mm_projector, 'experts'):
                # 是8专家投影器：为每个专家加载相同的预训练权重
                for expert in self.mm_projector.experts:
                    expert.load_state_dict(projector_weights, strict=False)
                print(f"Loaded pretrained proj weights to all {len(self.mm_projector.experts)} experts")
            else:
                # 普通投影器：保持原有逻辑
                self.mm_projector.load_state_dict(projector_weights, strict=False)


class LlavaMetaForCausalLM(ABC):
    """
    因果语言建模的多模态元类，抽象基类
    提供多模态输入处理的核心功能
    """

    @abstractmethod
    def get_model(self):
        """抽象方法，子类必须实现以返回内部模型实例"""
        pass

    def get_vision_tower(self):
        """获取视觉编码器，委托给内部模型"""
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        编码图像为特征向量
        流程：视觉编码器 → 视觉投影器
        """
        image_features = self.get_model().get_vision_tower()(images)  # 视觉编码
        image_features = self.get_model().mm_projector(image_features)  # 投影到语言模型空间
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        """
        准备多模态输入和标签的核心方法
        将文本token和图像特征组合成统一的输入序列
        这是LLaVA模型中最复杂和关键的方法之一
        """
        vision_tower = self.get_vision_tower()
        
        # 情况1: 如果没有视觉组件或图像，或只有单个token（推理阶段）
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                # 推理时处理：扩展注意力掩码和位置ID以包含图像特征
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 情况2: 有图像输入，开始多模态处理流程
        
        # 步骤1: 图像特征编码
        if type(images) is list or images.ndim == 5:
            # 处理多个图像或视频帧的情况
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)  # 批量编码图像
            split_sizes = [image.shape[0] for image in images]  # 计算每个样本的图像数量
            image_features = torch.split(image_features, split_sizes, dim=0)  # 按样本拆分特征
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]  # 展平并转移到设备
        else:
            # 处理单个图像的情况
            image_features = self.encode_images(images).to(self.device)

        # 步骤2: 输入预处理和完整性检查
        
        # TODO: 图像开始/结束标记功能尚未实现以支持预训练
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # 步骤3: 处理可能的None值，创建默认张量
        # 保存原始值以便后续恢复
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        # 为None的输入创建默认值
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)  # 全1掩码
        else:
            attention_mask = attention_mask.bool()  # 确保是布尔类型
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)  # 顺序位置ID
            
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)  # 使用忽略索引填充标签

        # 步骤4: 使用注意力掩码去除填充token
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 步骤5: 构建多模态输入序列（核心逻辑）
        new_input_embeds = []  # 新的输入嵌入序列
        new_labels = []  # 新的标签序列
        cur_image_idx = 0  # 当前图像特征索引
        
        # 逐个处理batch中的每个样本
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()  # 统计图像token数量
            
            # 情况A: 当前样本没有图像token
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)  # 文本嵌入
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)  # 拼接空图像特征
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 情况B: 当前样本包含图像token（多模态对话）
            # 定位所有图像token的位置
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []  # 不包含图像token的文本序列
            cur_labels = labels[batch_idx]
            cur_labels_noim = []  # 对应的标签序列
            
            # 按图像token分割文本序列
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                
            # 嵌入文本token并分割回原始片段
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 构建交替的文本-图像序列
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):  # 有N个图像token，就有N+1个文本段
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])  # 文本嵌入
                cur_new_labels.append(cur_labels_noim[i])  # 文本标签
                
                if i < num_images:
                    # 在文本段之间插入图像特征
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)  # 图像特征嵌入
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))  # 图像部分标签设为忽略

            # 合并当前样本的所有片段
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 步骤6: 序列截断和填充处理
        
        # 截断过长的序列（图像特征可能使序列变长）
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 步骤7: 批次填充对齐
        max_len = max(x.shape[0] for x in new_input_embeds)  # 找到最长序列
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []  # 填充后的输入嵌入
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 逐个样本进行填充
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # 左填充：在序列左侧添加零填充
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels  # 标签右对齐
                    attention_mask[i, -cur_len:] = True  # 有效位置设为True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # 右填充（默认）：在序列右侧添加零填充
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels  # 标签左对齐
                    attention_mask[i, :cur_len] = True  # 有效位置设为True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # 步骤8: 最终张量组合
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)  # 堆叠为批次

        # 恢复原始的空值状态（如果原始为None）
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # 返回处理后的多模态输入
        # input_ids设为None，因为已经转换为inputs_embeds
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
                    
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

