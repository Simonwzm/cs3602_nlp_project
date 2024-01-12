
重训练本文中的所有模型的方法如下所示

首先确保项目根目录下有 save2/ 文件夹防止模型运行出错

因为超过1G，最优的模型请通过交大云盘下载，下载链接为：
https://jbox.sjtu.edu.cn/l/G1M2t6



0. 测试最优模型
```python
python ./scripts/slu_bert_architecture.py --use_crf False --architecture 4 --testing
```

1. 训练BertOnly模型：
```
python ./scripts/slu_bert_architecture.py --use_crf False --architecture 0
```

1. 训练 Bert Multi-task 模型：
```
python ./scripts/slu_bert_architecture.py --use_crf True --architecture 0
```

1. 训练 bert Encoder + Decoder 模型：
```python
python ./scripts/slu_bert_architecture.py --use_crf False --architecture 1
```

1. 训练 bert + LSTM 模型：
```python
python ./scripts/slu_bert_architecture.py --use_crf False --architecture 4
```

1. 训练非bert 模型：

最基础的baseline
```python
python ./scripts/slu_baseline.py --use_crf False
```

加入词典的baseline
词典下载链接 
https://jbox.sjtu.edu.cn/l/F1iF5F
```python
python ./scripts/slu_baseline.py --use_crf False --word2vec_path ./output_mix.txt
```

加入数据增强的baseline
```python
# use old augmentation
python ./scripts/slu_baseline.py --use_crf False --aug_level 1
# use new augmentation
python ./scripts/slu_baseline.py --use_crf False --aug_level 2

```
