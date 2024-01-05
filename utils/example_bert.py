import json
# import jieba
import re
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer, BertModel
# from translate import Translator # translate Eng to CHinese

class Example():

    @classmethod
    def configuration(cls, root, asr=True,train_path=None, word2vec_path=None, tokenizer_name=None, extend_cais=False, extend_ecdt=False):
        cls.evaluator = Evaluator() # 评价
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path) #no use now
        cls.word2vec = Word2vecUtils(word2vec_path) # 词向量
        cls.label_vocab = LabelVocab(root) # ['B', 'I', 'O', '<pad>']
        cls.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    @classmethod
    def load_dataset(cls, data_path, extra_data1=None, extra_data2=None, use_asr=False, testing=False):
        datas = json.load(open(data_path, 'r', encoding="utf-8"))

        examples = [cls(utt, use_asr, testing) for data in datas for utt in data]

        if extra_data1 is not None:
            datas = json.load(open(extra_data1, 'r'))
            extra_examples_1 = [cls(utt, use_asr, testing) for data in datas for utt in data]
            examples += extra_examples_1

        if extra_data2 is not None:
            datas = json.load(open(extra_data2, 'r'))
            extra_examples_2 = [cls(utt, use_asr, testing) for data in datas for utt in data]
            examples += extra_examples_2
        
        return examples

    def __init__(self, ex: dict, cheat=False, testing=False):
        super(Example, self).__init__()
        self.ex = ex
        self.ex = ex
        # self.did = did

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

        # self.sep_tag_id = [1] * len(self.utt)




        # if use_asr:
        #     self.utt = ex['asr_1best'].upper()
        # else:
        #     self.utt = ex['manual_transcript']

        self.slot = {}
        if not testing:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
        
        self.tags = ['O'] * len(self.utt)
        self.sep_tag_id=[1] * len(self.utt)
        # self.acts = [''] * len(self.utt)
        self.slots = [''] * len(self.utt)

        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'

                self.sep_tag_id[bidx: bidx + len(value)] = [2] * len(value)
                self.sep_tag_id[bidx] = 3

                # self.acts[bidx: bidx + len(value)] = [slot.split('-')[0]] * len(value)

                self.slots[bidx: bidx + len(value)] = [slot.split('-')[1]] * len(value)
                # print(self.acts,self.slots)

        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        # self.act_id = [l.convert_act_to_idx(tag) for tag in self.acts]
        self.slot_id = [l.convert_slot_to_idx(tag) for tag in self.slots]

        # self.input_idx_ori = [Example.word_vocab[c] for c in self.utt]


        ### v1 ###  #78.324

        # self.utt=self.utt.replace(" ","_")

        # non_chinese = re.findall(r'[^\u4e00-\u9fff]', self.utt)
        # # Insert a space before each non-Chinese character
        # for c in non_chinese:
        #     self.utt = self.utt.replace(c, " " + c)
        
        # self.input_idx = Example.tokenizer(self.utt)["input_ids"][1:-1]
        # print(self.utt,self.input_idx) #bert tokenizer will pad <sos> and <eos>

        ### v1 ###



        ### v2 ### 77.99

        # replace_dict = {"(side)":"未知话语两侧", "(unknown)":"未知未知未知的内容", "(robot)":"未知的机器内容", "(dialect)":"未知未知的方言内容", "(noise)":"未知的噪声内容"," ":"空","null":"空的内容"
        #         } #"ok":"ok "*2, "ktv":"ktv "*3, "hi":"hi "*2, "beyond":"beyond "*6

        # # 对奇怪字符的处理
        # for key, value in replace_dict.items():
        #     self.utt = self.utt.replace(key, value)

        # self.input_idx = Example.tokenizer(self.utt)["input_ids"][1:-1]

        # # 对英文分词的处理 （逐字符与逐单词不符）
        # words = set(re.findall(r'[a-zA-Z]+', self.utt))
        # for word in words:
        #     # print(word)
        #     word_token_ori=Example.tokenizer(word)["input_ids"][1:-1]
        #     word_token=[]
        #     while len(word_token) < len(word):
        #         word_token = word_token + word_token_ori #c重复当前token，补齐至与word一样长
        #     if len(word_token) > len(word):
        #         word_token=word_token[0:len(word)]
        #     i=0
        #     while i < (len(self.input_idx) - len(word_token_ori) + 1):
        #         if self.input_idx[i:i+len(word_token_ori)] == word_token_ori:
        #             self.input_idx[i:i+len(word_token_ori)] = word_token
        #             i+=len(word)-1
        #         i+=1

        ### v2 ###
        
        ### v3 ### 78.44
        self.utt_ori = self.utt
        self.utt = self.utt.replace(" ","_").replace("～","~")
        self.input_idx = Example.tokenizer(self.utt)["input_ids"][1:-1]

        # 对英文分词的处理 （逐字符与逐单词不符）
        words = set(re.findall(r'[a-zA-Z0-9]+', self.utt))
        for word in words:
            # print(word)
            word_token_ori=Example.tokenizer(word)["input_ids"][1:-1]
            word_token=[]
            while len(word_token) < len(word):
                word_token = word_token + word_token_ori #c重复当前token，补齐至与word一样长
            if len(word_token) > len(word):
                word_token=word_token[0:len(word)]

            # bidx = self.utt.find(word)
            # eidx = bidx + len(word)
            # self.input_idx = self.input_idx[]
            i = 0
            while i < (len(self.input_idx) - len(word_token_ori) + 1):
                if self.input_idx[i:i+len(word_token_ori)] == word_token_ori:
                    self.input_idx[i:i+len(word_token_ori)] = word_token
                    i+=len(word)-1
                i+=1


        # ### v3 ###

        # assert len(self.utt_ori) == len(self.input_idx), f"Mismatch in length: {self.utt_ori} {self.input_idx}"
        # if self.utt_ori!=self.utt:
        #     print(self.utt_ori,self.utt)