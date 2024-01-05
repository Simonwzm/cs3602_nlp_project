# coding: utf-8


# 导入所需的库和模块
import sys, os, time, gc, json
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 设置安装路径以导入相关模块
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

# 导入自定义模块
from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
# from model.slu_baseline_tagging import SLUTagging as SLUTagging
# from model.bilstm_crf import SLUTagging 
# from model.slu_bert_tagging import SLUTaggingBERTCascaded as SLUTagging
from model.slu_bert_tagging import SLUTagging , SLUTaggingBERT

# 初始化参数、设置随机种子和配置设备（CPU或GPU）
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# 加载数据集
start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
# 配置数据处理的相关参数
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
# 加载训练和验证数据集
train_dataset = Example.load_dataset(train_path, cheat=True)
# Example.to_json(train_dataset, path='data/export_train.json')
# exit()
dev_dataset = Example.load_dataset(dev_path, cheat=False)
# Example.to_json(dev_dataset, path='data/export_dev.json')
test_dataset = Example.load_dataset(test_path, cheat=False)
# Example.to_json(test_dataset, path='data/export_test.json')
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
experiment_name = args.experiment_name + '_lr_' + str(args.lr) + '_cheat_' + str(args.cheat)   + "_baseline_"
writer = SummaryWriter(log_dir=args.log_dir + experiment_name)

# 根据加载的数据配置模型参数
args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

# 实例化模型并加载预训练的词嵌入
model = SLUTagging(args).to(device)
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

# 如果处于测试模式，加载保存的模型状态
if args.testing:
    check_point = torch.load(open('model.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")

# set_optimizer函数：定义并返回模型的优化器
def set_optimizer(model, args):
    # 只为需要梯度的参数创建优化器
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer

# decode函数：在训练集或验证集上评估模型性能，计算损失和准确率
def decode(choice):
    # 确保评估的数据集是 'train' 或 'dev'
    assert choice in ['train', 'dev']
    # 将模型设置为评估模式
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    noise_indicator = []
    total_loss, count = 0, 0
    # 不计算梯度，以加速和节省内存
    with torch.no_grad():
        # 分批处理数据集
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            # 创建当前批次的数据
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            has_manuscript = current_batch.has_manuscript
            if has_manuscript:
                noise = current_batch.noise
            else:
                noise = None
            # 解码模型预测
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            # 可选：如果预测中有异常，打印出来
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            # 将预测结果和真实标签添加到列表中
            predictions.extend(pred)
            labels.extend(label)
            noise_indicator.extend(noise)
            # 累加损失
            total_loss += loss
            count += 1
        # 计算总体指标
        metrics = Example.evaluator.acc(predictions, labels, noise_indicator)
    # 清理CUDA缓存和垃圾收集以释放内存
    torch.cuda.empty_cache()
    gc.collect()
    # 返回性能指标和平均损失
    return metrics, total_loss / count

# predict函数：在没有标签的测试数据上进行预测，并保存结果
def predict():
    # 将模型设置为评估模式
    model.eval()
    # 加载测试数据集
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    # 不计算梯度，以加速和节省内存
    with torch.no_grad():
        # 分批处理测试数据集
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            # 创建当前批次的数据
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            # 解码模型预测
            pred = model.decode(Example.label_vocab, current_batch)
            # 将预测结果保存到字典中
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    # 加载测试数据文件以添加预测
    test_json = json.load(open(test_path, 'r', encoding="utf-8"))
    ptr = 0
    # 遍历测试数据，将预测添加到每条数据中
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    # 将数据和预测一起保存为JSON文件
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)

# 如果不是测试模式，执行训练循环
if not args.testing:
    # 根据批次大小和最大训练周期数计算总训练步数
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    # 设置优化器
    optimizer = set_optimizer(model, args)
    # 初始化训练相关的变量
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    # 生成训练索引和批次大小
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    # 对于每一个训练周期
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        # 随机打乱训练数据索引
        np.random.shuffle(train_index)
        # 将模型设置为训练模式
        model.train()
        count = 0
        # 分批次训练模型
        for j in range(0, nsamples, step_size):
            # 提取当前批次的训练数据
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            # 创建当前批次的数据
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # 通过 taggingfnn 模型的forward输出
            # 通过模型得到输出和损失
            # output, loss = model(current_batch)
            temp = model(current_batch)
            if len(temp) == 2:
                output, loss = temp
            elif len(temp)==4:
                # loss = sep_loss
                output, loss, sep_loss, tag_loss = temp
            if loss == 0 :
                continue
            # 累加损失
            epoch_loss += loss.item()
            # 反向传播优化模型
            loss.backward()
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
            count += 1
            writer.add_scalar('train/loss', loss.item(), i * (nsamples // step_size) + j // step_size)
        # 打印训练周期的统计信息
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        # 清理CUDA缓存和垃圾收集
        torch.cuda.empty_cache()
        gc.collect()

        # 在验证集上评估模型
        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        writer.add_scalar('dev/loss', dev_loss, i)
        writer.add_scalar('dev/acc', dev_acc, i)
        writer.add_scalar('dev/f1', dev_fscore['fscore'], i)

        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        # 如果达到最佳结果，保存模型
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(args.save_dir + experiment_name, 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if i % args.save_epoch == 0:
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(args.save_dir + experiment_name + str(i), 'wb'))

    # 打印最终的最佳结果
    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    writer.close()
# 如果是测试模式，进行评估和预测
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
