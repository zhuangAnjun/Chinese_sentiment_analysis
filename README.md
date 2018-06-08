# 项目说明
该项目为中文情感分析，一共三分类（褒义，中性，贬义）<br>
本项目采用了200v,300v,400三种不同尺度的输入，一共六通道，每个尺度有两个通道，一个可训练，一个固定。 <br>
在特征提取方面使用了三层cnn，再将第二层和第三层的cnn结果pooling后连接到全连接层。<br>

# 项目文件说明
## model.py 
模型文件 <br>
## pre_process.py 
nlp数据预处理，包括生成词典，padding，生成embedding矩阵等<br>
## f1_score.py 
计算f1分数的函数<br>
## data_process.py 
对原始数据进行编码转换以及清洗分词等函数<br>
## train.py 
训练模型，其中包含有预定义的各种参数，包括文件路径等.训练好后对训练数据进行预测并写入文件<br>
## sampling.py 
sampling函数，对训练数据不采样或者过采样。默认进行采样

# data_process文件包含两个函数 <br>

## process_data(input_file, out_file) <br>
### 将训练数据集和测试数据集评论部分提取出来，将句子编码转换utf-8，分割成单独的字后存储到文件里 <br>
### 调用： <br>
process_data(input_file, out_file)<br>
### 注意：<br>
这个函数只针对特定数据集的数据整理，并不通用 <br>

## process_corpus(input_file, out_file) <br>
### 功能 <br>
 清洗搜狗语料，同时将整段话以'。'和'；'为标志划分成单独的句子，再将句子划分成单独的字 <br>
### 调用 <br>
process_corpus(input_file, out_file)<br>