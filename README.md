该项目为中文情感分析，一共三分类（褒义，中性，贬义）
本项目采用了200v,300v,400三种不同尺度的输入，一共六通道，每个尺度有两个通道，一个可训练，一个固定。
在特征提取方面使用了三层cnn，再将第二层和第三层的cnn结果pooling后连接到全连接层。

model.py 模型文件
pre_process.py nlp数据预处理，包括生成词典，padding，生成embedding矩阵等
f1_score.py 计算f1分数的函数
data_process.py 对原始数据进行编码转换以及清洗分词等函数
train.py 训练模型，其中包含有预定义的各种参数，包括文件路径等.训练好后对训练数据进行预测并写入文件
sampling.py sampling函数，对训练数据不采样或者过采样。默认进行采样# Chinese_sentiment_analysis
Chinese sentiment analysis
