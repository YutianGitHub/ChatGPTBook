# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/3/21 10:08
"""
    文件说明：
            
"""
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel


class PromptModel(BertPreTrainedModel):
    """Prompt分类模型"""

    def __init__(self, config):
        super().__init__(config)
        """
        初始化函数
        Args:
            config: 配置参数
        """
        self.bert = BertModel(config, add_pooling_layer=False)
        # (*,hidden_size)->(*,vocab_size)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, attention_mask, mask_index, token_handler, words_ids, words_ids_mask,
                label=None):
        """
        前向函数，计算Prompt模型预测结果
        Args:
            input_ids:
            attention_mask:[注意力遮罩]
            mask_index:[MASK]的索引
            token_handler:
            words_ids:[标签映射ids]
            words_ids_mask:[mask]
            label:

        Returns:

        """
        # 获取BERT模型的输出结果,(batch,seq_len,hidden_size)
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        # 经过一个全连接层，获取隐层节点状态中的每一个位置的词表,(batch,seq_len,vocab_size)
        logits = self.cls(sequence_output)
        # 获取批次数据中每个样本内容对应mask位置标记
        logits_shapes = logits.shape

        mask = mask_index + torch.range(0, logits_shapes[0] - 1, dtype=torch.long, device=logits.device) * \
               logits_shapes[1]
        mask = mask.reshape([-1, 1]).repeat([1, logits_shapes[2]]) # (batch,vocab_size)
        # 获取每个mask标记对应的logits内容(batch,vocab_size),应该是每个[MASK]位置的logits
        mask_logits = logits.reshape([-1, logits_shapes[2]]).gather(0, mask).reshape(-1, logits_shapes[2])
        # 获取答案空间映射的标签向量
        label_words_logits = self.process_logits(mask_logits, token_handler, words_ids, words_ids_mask)
        # 将其进行归一化以及获取对应标签，(batch,cls_num)
        score = torch.nn.functional.softmax(label_words_logits, dim=-1)
        pre_label = torch.argmax(label_words_logits, dim=1)
        outputs = (score, pre_label)
        # 当label不为空时，计算损失值
        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(label_words_logits, label)
            outputs = (loss,) + outputs
        return outputs

    def process_logits(self, mask_logits, token_handler, words_ids, words_ids_mask):
        """
        获取答案空间映射的标签向量，用于分类判断
        Args:
            mask_logits: mask位置的logits
            token_handler: 多token操作策略，包含first、mask和mean
            words_ids: 标签词id矩阵
            words_ids_mask: 标签词id掩码矩阵

        Returns:

        """
        # 获取标签词id及掩码矩阵
        label_words_ids = nn.Parameter(words_ids, requires_grad=False)
        label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        # 获取mask位置上标签词向量值(batch,cls_num,each_cls_words_num,word_len)
        label_words_logits = mask_logits[:, label_words_ids]
        # 根据多token操作策略进行标签词向量构建
        if token_handler == "first":#选择每个词的第1个作为这个词的logits
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif token_handler == "max":# 选最大
            label_words_logits = label_words_logits - 1000 * (1 - words_ids_mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif token_handler == "mean":
            label_words_logits = (label_words_logits * words_ids_mask.unsqueeze(0)).sum(dim=-1) / (
                    words_ids_mask.unsqueeze(0).sum(dim=-1) + 1e-15)
        # 将填充的位置进行掩码 #(batch,label_num,each_label_words_num),(批数，每个标签情况，每个标签的每个词的logits)
        label_words_logits -= 10000 * (1 - label_words_mask)
        # 最终获取mask标记对应的答案空间映射向量（每个类别的logits是每个类别下所有词的logits的平均）
        label_words_logits = (label_words_logits * label_words_mask).sum(-1) / label_words_mask.sum(-1)
        return label_words_logits #(batch,label_num),(批数，每个类别的logits)
