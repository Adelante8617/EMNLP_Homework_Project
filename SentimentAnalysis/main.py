import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import gc
from tqdm import tqdm
import argparse
import os

from text_preprocess import loadDataAndPreprocess
from ArabicTweetDataset import MultiLabelTextDataset
from mymodel import BertForMultiLabelClassification

def record(filename, model_name, epoch, loss_dic, sentiments):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Epoch: {epoch}\n")
        f.write(f"Loss: {loss_dic[model_name]}\n")
        f.write("Sentiments:\n")
        for sentiment, count in sentiments.items():
            f.write(f"{sentiment}: {count}\n")
        f.write("="*70 + "\n")

def main(args):
    print("running")
    
    sentiment_dic = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name
    loss_dic = {model_name: []}
    print(loss_dic)

    train_data = loadDataAndPreprocess(args.train_file)
    test_data = loadDataAndPreprocess(args.test_file, isTest=True)

    if args.dialect_only:
        test_data = test_data[test_data['location'].isin(["Morocco", "Algeria"])]

    texts = train_data['Tweet'].to_list()
    labels = np.array(train_data.iloc[:, 2:]).astype(float).tolist()

    test_texts = test_data['Tweet'].to_list()
    test_labels = np.array(test_data.iloc[:, 2:-1]).astype(float).tolist()

    sentiment_names = test_data.columns[2:13].tolist()

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)

    dataset = MultiLabelTextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    num_labels = 11  # 标签的数量
    model = BertForMultiLabelClassification(model_name, num_labels).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for input_ids, labels in tqdm(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss / len(dataloader):.4f}')
        loss_dic[model_name].append(total_loss / len(dataloader))

    test_dataset = MultiLabelTextDataset(test_texts, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()

    each_sentiment = [0] * 11

    with torch.no_grad():
        for input_ids, labels in tqdm(test_dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            predicted = (outputs > 0.5).float()
            for sentiment in range(num_labels):
                if predicted[0][sentiment] == labels[0][sentiment]:
                    each_sentiment[sentiment] += 1

    print("Encoder:", model_name)
    for sentiment in range(num_labels):
        sentiment_dic[sentiment_names[sentiment]] = each_sentiment[sentiment]
        print("{:<15} {:<5} / {}".format(sentiment_names[sentiment], each_sentiment[sentiment], len(test_labels)))
    print("==================================================================")
    
    result_file = "results.txt"
    record(result_file, model_name, epoch + 1, loss_dic, sentiment_dic)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT-based Multi-Label Classification')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing data file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--dialect_only', action='store_true', help='Filter test data for specific dialects')
    parser.add_argument('--model_name', type=str, required=True, default="bert-base-multilingual-cased",
                        choices=["aubmindlab/bert-base-arabertv02", "bert-base-uncased",
                                 "bert-base-multilingual-cased", "asafaya/bert-base-arabic"],
                        help='BERT model name to use, select from given four choices')

    args = parser.parse_args()
    main(args)

