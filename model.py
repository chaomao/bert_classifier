# coding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel


# Bert
class BertClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # 定义BERT模型
        self.bert = BertModel(config=bert_config)
        # 定义分类器
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT的输出
        # 分为两个部分，第一个元素是输入序列所有 token 的 Embedding 向量层，第二个变量是[CLS]位的隐层信息
        # [CLS]id[SEP]    [4 768]  [[1_CLS], [2_CLS], [], []]
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output  [4, 768]
        pooled = bert_output[1]
        # 分类  [CLS] [] [] []
        # [4, 512]
        # [4, 512, 768]
        # [4, 512, 768]
        # [CLS]
        # [4, 768]  * [768, 10] = [4, 10]
        logits = self.classifier(pooled)
        # 返回softmax后结果
        return torch.softmax(logits, dim=1)
