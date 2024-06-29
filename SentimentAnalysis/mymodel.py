import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token的输出
        logits = self.classifier(cls_output)
        return torch.sigmoid(logits)