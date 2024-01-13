import os

root_dir = os.path.expanduser("~")
# path
data_path = "./data/atis/"
vocab_path = data_path + "vocab.txt"
model_save_dir = "./ckpt/"
model_path = "atis_model.bin"

# model hyperparameters
hidden_dim = 256
emb_dorpout = 0.75
lstm_dropout = 0.5
attention_dropout = 0.1
num_attention_heads = 8

# hyperparameters
max_len = 64
lr_scheduler_gama = 0.5
epoch = 200
seed = 2023
lr = 0.00001
eps = 1e-12
use_gpu = True
