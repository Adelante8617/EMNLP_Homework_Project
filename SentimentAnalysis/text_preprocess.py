# 导入必要的库
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# 加载预训练的分词器和模型
tokenizer = AutoTokenizer.from_pretrained("lafifi-24/arabicBert_arabic_dialect_identification")
model_encode_text = AutoModelForSequenceClassification.from_pretrained(
    "lafifi-24/arabicBert_arabic_dialect_identification")

# 阿拉伯地区国家编号
loca = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM',
        'PL', 'QA', 'SA', 'SD', 'SY', 'TN', 'YE', 'MSA']

# 编号对应的国家
country = {'EG': 'Egypt', 'SA': 'Saudi Arabia', 'MA': 'Morocco', 'DZ': 'Algeria', 'SY': 'Syria', 'QA': 'Qatar',
           'LB': 'Lebanon', 'YE': 'Yemen', "MSA": "Not Dialect",
           'AE': 'United Arab Emirates', 'KW': 'Kuwait', 'SD': 'Sudan', 'BH': 'Bahrain', 'JO': 'Jordan', 'IQ': 'Iraq',
           'PL': 'Palestine', 'OM': 'Oman', 'LY': 'Libya', 'TN': 'Tunisia'}


def loadDataAndPreprocess(file_path, needPreprocess=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines_train = file.readlines()

    lines_train = [x.replace("\n", "").split('\t') for x in lines_train]

    df = pd.DataFrame(lines_train)
    df.columns = df.iloc[0]
    df = df[1:]

    # 重置索引
    df.reset_index(drop=True, inplace=True)

    if needPreprocess:
        Location = []

        for idx in tqdm(range(df.shape[0])):
            text = df.iloc[idx, 1]

            # 对文本进行预处理
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)

            # 模型推理（预测）
            with torch.no_grad():
                outputs = model_encode_text(**inputs)

            # 获取预测结果
            predictions = torch.argmax(outputs.logits, dim=-1)

            # 打印预测结果
            for text, pred in zip(text, predictions):
                # print(f"文本: {text} -> 预测标签: {pred.item()}")
                Location.append(country[loca[pred.item()]])

        df['location'] = Location
    return df
