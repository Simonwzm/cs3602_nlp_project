

import sys
import os
import time
import gc
import json
from torch.optim import Adam


install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from slu.utils.args import init_args
from slu.utils.initialization import *
from slu.utils.example import Example
from slu.utils.batch import from_example_list
from slu.utils.vocab import PAD
# from slu.model.slu_baseline_tagging import SLUTagging
from slu.model.dca_net_models import Joint_model
from slu.dca_net.model.Radam import RAdam
from slu.dca_net.data_util import config as dca_config


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice, args, model, train_dataset, dev_dataset):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict(args, model):
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r', encoding="utf-8"))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


if __name__ == '__main__':

    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    device = set_torch_device(args.device)
    print("Initialization finished ...")
    print("Random seed is set to %d" % (args.seed))
    print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

    ckpt_path = '../dca_net/ckpt/'

    start_time = time.time()
    train_path = os.path.join(args.dataroot, 'train.json')
    dev_path = os.path.join(args.dataroot, 'development.json')
    Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
    train_dataset = Example.load_dataset(train_path, "asr_1best")
    dev_dataset = Example.load_dataset(dev_path, "asr_1best")

    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

    model = Joint_model(n_class=Example.intent_vocab.num_tags, n_tag=Example.label_vocab.num_tags, vocab_size=args.vocab_size, embed_size=args.embed_size, batch_size=args.batch_size).to(device)
    if args.testing:
        check_point = torch.load(open('model.bin', 'rb'), map_location=device)
        model.load_state_dict(check_point['model'])
        print("Load saved model from root path")
    Example.word2vec.load_embeddings(model.embed, Example.word_vocab, device=device)

    print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=dca_config.lr, weight_decay=0.000001)
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=dca_config.lr_scheduler_gama, last_epoch=-1)

    if not args.testing:
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        print('Total training steps: %d' % (num_training_steps))

        nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
        train_index, step_size = np.arange(nsamples), args.batch_size
        print('Start training ......')

        for i in range(args.max_epoch):
            print(scheduler.get_lr())
            start_time = time.time()
            epoch_loss = 0
            np.random.shuffle(train_index)
            model.train()
            count = 0
            for j in range(0, nsamples, step_size):
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
                current_batch = from_example_list(args, cur_dataset, device, train=True)

                logits_intent, logits_slot = model(current_batch)
                loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, current_batch)

                if i < 40:
                    loss = loss_slot + loss_intent
                else:
                    loss = 0.8 * loss_intent + 0.2 * loss_slot

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                count += 1
            print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
            torch.cuda.empty_cache()
            gc.collect()

            start_time = time.time()
            metrics, dev_loss = decode('dev', args, model, train_dataset, dev_dataset)
            dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
            print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i

                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, open(os.path.join(os.path.dirname(__file__), ckpt_path, "model.bin"), 'wb'))
                print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

            scheduler.step()

        print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    else:
        start_time = time.time()
        metrics, dev_loss = decode('dev', args, model, train_dataset, dev_dataset)
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        predict(args, model)
        print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
