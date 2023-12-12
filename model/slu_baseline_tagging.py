#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# 定义 SLUTagging 类，继承自 nn.Module，用于构建双向 LSTM 的模型
class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        # 存储配置参数
        self.config = config
        # 指定编码器的类型，这里使用 LSTM 或其他 RNN 变种
        self.cell = config.encoder_cell
        # 创建词嵌入层
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        # 构建双向 LSTM
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        # 添加 dropout 层以防止过拟合
        self.dropout_layer = nn.Dropout(p=config.dropout)
        # 输出层，采用线性层后接 CrossEntropyLoss，用于标签预测
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    # 定义模型的前向传播
    def forward(self, batch):
        # 从批次中提取必要的数据
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        # 将输入的词 ID 序列转换为词嵌入
        embed = self.word_embed(input_ids)
        # 打包填充的序列以便于 LSTM 处理可变长度的输入
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # 维度为 bsize x seqlen x dim
        # 解包序列以将其转换回填充的格式
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        # 应用 dropout
        hiddens = self.dropout_layer(rnn_out)
        # 通过输出层得到标签的输出
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        # print(tag_ids[0])
        # print(tag_output[0][0])
        # exit()

        return tag_output

    # 定义解码函数，用于从模型的输出中生成标签预测，并计算损失
    def decode(self, label_vocab, batch):
        # 获取批次大小
        batch_size = len(batch)
        # 获取真实标签序列
        labels = batch.labels
        # 获取模型的输出
        output = self.forward(batch)
        # 从输出中得到概率分布
        prob = output[0]
        predictions = []
        # 为批次中的每个序列生成预测
        for i in range(batch_size):
            # 根据概率的最大值选择预测的标签
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # 保持预测长度与真实句子长度一致
            pred = pred[:len(batch.utt[i])]
            # 将标签索引转换为标签字符串，并生成标记的元组
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                print(tag)
                # 使用 B-I 标记方案来识别和提取实体
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            print(tag_buff)
            # 最后检查是否有剩余的实体需要添加
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        # 如果没有提供标签，仅返回预测
        if len(output) == 1:
            return predictions
        else:
            # 如果提供了标签，返回预测、标签和损失
            loss = output[1]
            return predictions, labels, loss.cpu().item()

# 定义 TaggingFNNDecoder 类，用于 SLUTagging 模型中的输出层
class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        # 输出层的标签数量
        self.num_tags = num_tags
        # 线性层，用于将隐藏状态转换为标签空间
        self.output_layer = nn.Linear(input_size, num_tags)
        # 定义损失函数，忽略填充的标签
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 定义前向传播，计算 logits 和损失
    def forward(self, hiddens, mask, labels=None):
        # 经过线性层得到 logits
        logits = self.output_layer(hiddens)
        # 应用掩码以忽略填充位置的 logits
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        # 计算概率分布
        prob = torch.softmax(logits, dim=-1)
        # 如果提供了标签，计算损失
        if labels is not None:
            # print(labels.clone().detach().cpu()[0])
            # print(logits.clone().detach().cpu()[0])
            # exit()
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        # 如果没有提供标签，仅返回概率分布
        return (prob, )
