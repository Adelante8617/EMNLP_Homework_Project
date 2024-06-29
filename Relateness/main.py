import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import argparse
import os
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
from torchmetrics.regression import SpearmanCorrCoef

plt.style.use('ggplot')

class SentencePairDataset(Dataset):
    def __init__(self, sentences1, sentences2, scores, tokenizer, max_len):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        sentence1 = self.sentences1[idx]
        sentence2 = self.sentences2[idx]
        score = self.scores[idx]

        inputs1 = self.tokenizer(sentence1, return_tensors="pt", padding='max_length', truncation=True,
                                 max_length=self.max_len)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", padding='max_length', truncation=True,
                                 max_length=self.max_len)

        return inputs1, inputs2, torch.tensor(score, dtype=torch.float)


class SiameseBert(nn.Module):
    def __init__(self, pretrained_model_name):
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert(input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2, attention_mask=attention_mask2)

        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output

        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)

        # 通过全连接层和注意力层
        x1 = F.relu(self.fc1(pooled_output1))
        x2 = F.relu(self.fc1(pooled_output2))

        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc2(x2))

        score1 = self.fc4(x1)
        score2 = self.fc4(x2)

        return torch.exp(-abs(score1 - score2))


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        preds_mean = torch.mean(preds)
        targets_mean = torch.mean(targets)

        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean

        correlation = torch.sum(preds_centered * targets_centered) / (
                torch.sqrt(torch.sum(preds_centered ** 2)) * torch.sqrt(torch.sum(targets_centered ** 2))
        )

        return 1 - correlation


def main():
    print("Running...")
    # Argument parsing
    parser = argparse.ArgumentParser(description='Siamese BERT model for sentence similarity')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased', help='Name of the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_len', type=int, default=32, help='Maximum length of tokenized input')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--loss_function', type=str, default='pearson', choices=['pearson', 'mse'], help='Loss function to use')
    parser.add_argument('--log_file', type=str, default="experiment_log.txt",  help='Path to the log file')
    parser.add_argument('--output_csv', type=str,default='', help='Directory to save the output CSV file')
    args = parser.parse_args()

    df_str_rel = pd.read_csv(args.train_data_path)
    df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))

    gc.collect()
    torch.cuda.empty_cache()

    pretrained_model_name = args.pretrained_model_name

    sentences_train = df_str_rel["Split_Text"].to_list()
    scores = df_str_rel['Score'].to_list()

    sentence_1 = [x[0] for x in sentences_train]
    sentence_2 = [x[1] for x in sentences_train]

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    dataset = SentencePairDataset(sentence_1, sentence_2, scores, tokenizer, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseBert(pretrained_model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()

    gc.collect()
    torch.cuda.empty_cache()

    loss_fn = None
    if args.loss_function == 'pearson':
        loss_fn = PearsonLoss()
    elif args.loss_function == 'mse':
        loss_fn = nn.MSELoss()

    # Open the log file
    with open(args.log_file, 'a') as log_file:
        log_file.write(f'Pretrained model: {args.pretrained_model_name}\n')

        # Training loop
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            for inputs1, inputs2, targets in tqdm(dataloader):
                input_ids1 = inputs1['input_ids'].squeeze(1).to(device)
                attention_mask1 = inputs1['attention_mask'].squeeze(1).to(device)
                input_ids2 = inputs2['input_ids'].squeeze(1).to(device)
                attention_mask2 = inputs2['attention_mask'].squeeze(1).to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)

                loss = loss_fn(output, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            log_file.write(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}\n')
            print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")

        # Evaluation
        df_str_rel_test = pd.read_csv(args.test_data_path)
        df_str_rel_test['Split_Text'] = df_str_rel_test['Text'].apply(lambda x: x.split("\n"))
        sentences_test = df_str_rel_test["Split_Text"].to_list()
        scores = df_str_rel_test['Score'].to_list()
        
        sentence_1 = [x[0] for x in sentences_test]
        sentence_2 = [x[1] for x in sentences_test]

        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        max_len = 32
        test_dataset = SentencePairDataset(sentence_1, sentence_2, scores, tokenizer, max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        
        pred_score = []
        model.eval()
        with torch.no_grad():
            for inputs1, inputs2, targets in test_dataloader:
                input_ids1 = inputs1['input_ids'].squeeze(1).cuda()
                attention_mask1 = inputs1['attention_mask'].squeeze(1).cuda()
                input_ids2 = inputs2['input_ids'].squeeze(1).cuda()
                attention_mask2 = inputs2['attention_mask'].squeeze(1).cuda()

                output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                output = output.squeeze(1)

                pred_score.extend(output.cpu().numpy().tolist())

        df_str_rel_test['Pred_Score'] = pred_score
        gold_scores = df_str_rel_test['Score']
        pred_scores = df_str_rel_test['Pred_Score']
        score = spearmanr(np.array(gold_scores), np.array(pred_scores))[0]
        log_file.write(f'Final score: {score}\n')
        print(score)
        
        
        if args.output_csv:
            df_output = df_str_rel_test[['PairID', 'Pred_Score']]
            df_output.to_csv(args.output_csv, index=False)
            log_file.write(f'Saved test results to {args.output_csv}\n')
            
        log_file.write('========================================================================\n')


if __name__ == "__main__":
    main()
