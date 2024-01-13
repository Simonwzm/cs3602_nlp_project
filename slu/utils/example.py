import json

from slu.utils.vocab import Vocab, LabelVocab, IntentVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)
        cls.intent_vocab = IntentVocab(root)

    @classmethod
    def load_dataset(cls, data_path, choice='asr_1best'):
        assert choice in ['asr_1best', 'manual_transcript']
        dataset = json.load(open(data_path, 'r', encoding="utf-8"))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', choice=choice)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, choice='asr_1best'):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex[choice]  # ! since manual transcript contais real labels, so the acc went up!
        self.slot = {}
        self.intent = Example.intent_vocab.convert_intent_to_idx("O")
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        # ! "导航重庆师范大学图书馆"'s tag corresponds to "B-inform-导航 I-inform-导航 B-inform-终点名称 I-inform-终点名称 I-inform-终点名称 I-inform-终点名称 I-inform-终点名称 ...."
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
                self.intent = Example.intent_vocab.convert_intent_to_idx(slot)
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
