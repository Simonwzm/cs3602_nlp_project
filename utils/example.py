import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, cheat=False):
        dataset = json.load(open(data_path, 'r', encoding="utf-8"))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', cheat)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, cheat=False):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']
        try: 
            self.manuscript = ex['manual_transcript']
            self.noise = (self.manuscript != self.utt)
            if cheat:
                self.utt = ex['manual_transcript']
            self.has_manuscript = 1
        except:
            self.has_manuscript = 0
            self.manuscript = None
            self.noise = False
            print("manuscript is None in this dataset")

        self.sep_tag_id = [1] * len(self.utt)

        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}' # 标签列表

                self.sep_tag_id[bidx: bidx + len(value)] = [2] * len(value)
                self.sep_tag_id[bidx] = 3
        # 三元组 : e.g.: ['B-Inform-导航', 'I-Inform-导航',]
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]         
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        # self.slot_id = [l.convert_slot_to_idx(tag) for tag in self.slots]

    
    @classmethod
    def to_json(cls, examples, path='data/export_train.json'):
        '''
        将 Example 列表转换为 JSON 格式, 存储位置为 path
        对 Examples 中的每个ex， 生成一行 json 数据
        每一行形如：
        {"utt": ex.utt, "tags": ex.tags,  "slotvalue": ex.slotvalue, "did": ex.did}
        '''
        for ex in examples:
            if ex.manuscript:
                manuscript = ex.manuscript
            else:
                manuscript = ""
            # if ex.slotvalue == []:
            #     ex.slotvalue = ["DK"] * len(ex.utt)
            ex_dict = {"utt": ex.utt, "tags": ex.tags,  "slotvalue": ex.slotvalue, "did": ex.did, "manuscript": manuscript}
            with open(path, 'a', encoding="utf-8") as f:
                json.dump(ex_dict, f, ensure_ascii=False)
                f.write('\n')


