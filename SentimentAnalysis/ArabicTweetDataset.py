import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True,
                                padding='max_length')
        return inputs['input_ids'].squeeze(), torch.tensor(labels, dtype=torch.float)

