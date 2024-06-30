# EMNLP 2024
本项目为EMNLP课程大作业。

---

## 主题
使用多种方式评估现有的NLP技术对阿拉伯语方言的处理能力。

测评方式包含：
- 阿拉伯语编码器
	- 针对阿拉伯语方言分别实验，综合对比各个编码器对特定任务的完成效果，比较不同编码器对实验的影响
- 大语言模型对阿拉伯语的支持
	- 编写案例，调研评估大模型对方言相关问题的回复

## 文件内容
- LLMEvaluation，包含了测试大模型所用到的问题集与收到的回复
- Relateness，在SemEval2024数据集的基础上实现了双生网络(Siamese Network) 用于计算语句相关度
- SentimentAnalysis，在SemEval2018数据集的基础上实现了阿拉伯语方言情感分类任务
- requirements.txt文件
- 报告PDF文件

## 使用方式

对于语义相关性计算任务
> 通过 `pip install requirements.txt`配置基本的环境
> 
> 在此基础上在终端使用`python main.py --train_path "your_path.csv" --test_path "your_test_path.csv"`
> 即可运行程序。其余参数可使用help命令查看用法。
> 
> 示例：
> 
> `python main.py --train_data_path "ary_train.csv" --test_data_path "ary_test_with_labels.csv" --pretrained_model_name "bert-base-uncased" --loss_function pearson`
>
> 对于希望按照SemEval提供的evaluation.py文件直接测评，需要对relateness/input_dir/res中的希望测评的文件名称修改为pred_score.py然后配合对应的label文件使用。
> 
> *由于该过程较麻烦，main.py中已经直接整合了这部分的计算和输出结果，可以直接使用main.py得到评估结果*

对于情感分析任务：
> 与前一部分基本相同，在对应文件夹下运行main.py即可。每次运行的损失函数值以及最终分类结果都会写入result.txt文件，可以完成运行后直接观看。
> 
> 数据集从SemEval2018得到，但在它的基础上增加了预处理，得到了每句话对应的方言标签。可通过parser参数need_location_label控制是否需要重新预处理。

对于大模型测评任务：
> 这并非一个可执行程序，直接从报告和数据中得到结果。
> 
> 在LLMEvaluation 文件夹下有提供给大模型的问题以及对应的回复。为了便于不了解阿拉伯语的人直观感受，我们还提供了这些文本的人工中文翻译。txt文件为纯文本，而Word文件内容相同，但是有更好的排版效果。



