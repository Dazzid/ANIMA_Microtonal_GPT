
from trainer import Trainer, TrainerConfig
from mingpt_utils import set_seed
from model import GPT, GPTConfig
import torch
from utils import *

from torch.utils.tensorboard import SummaryWriter
from mingpt_utils import sample
torch.cuda.empty_cache()

print("Available devices: ", torch.cuda.device_count())
print("torch version:", torch.__version__)
print("cudnn version:", torch.backends.cudnn.version())
print("cuda version:", torch.version.cuda)

print("loading the dataset....")
max_length = 1024
id = 0
tokens = np.load('../dataset/formatted/tokens.npy', allow_pickle=True)
train = np.load('../dataset/shuffled/dataset_train.npy', allow_pickle=True)
test = np.load('../dataset/shuffled/dataset_test.npy', allow_pickle=True)
midi_train = np.load('../dataset/shuffled/midi_train.npy', allow_pickle=True)
midi_test = np.load('../dataset/shuffled/midi_test.npy', allow_pickle=True)

print("Dataset ready!")

#Convert midi into dtype int
midi_train = midi_train.astype(int)
midi_test = midi_test.astype(int)

print(train.shape, test.shape, midi_train.shape, midi_test.shape)

dataset = TokenDatasetMidi(train, midi_train,  max_length, tokens)
validation = TokenDatasetMidi(test, midi_test, max_length, tokens)

import wandb
#wandb.login()


wandb.init(
    # set the wandb project where this run will be logged
    project="music_gpt_new_voicing",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-5,
    "architecture": "Transformer - minGPT",
    "dataset": "chords from iRealPro",
    "epochs": 250,
    }
)

epochs = 110
embedding = 512
heads = 4
layers = 4
batch_size = 128
learning_rate = 3e-5
num_workers = 4
midi_vocab = 128
token_size = len(tokens)

#print parameters in a new line
print("-----------------------------")
print("\t\tParameters:")
print(
    f'epochs: {epochs}\n'
    f'embedding: {embedding}\n'
    f'heads: {heads}\n'
    f'layers: {layers}\n'
    f'batch_size: {batch_size}\n'
    f'learning_rate: {learning_rate}\n'
    f'num_workers: {num_workers}\n'
    f'midi_vocab: {midi_vocab}\n'
    f'token_size: {token_size}'
)
print("-----------------------------")

mconf = GPTConfig(len(tokens), dataset.block_size, midi_vocab, n_layer=layers, n_head=heads, n_embd=embedding)
session_model = GPT(mconf)
MODEL_NAME = "../models/model_"+ "epochs->" + str(epochs) + "_heads->" + str(heads) + "_embd->" + str(embedding) + "_batch->" + str(batch_size) + "_new_midi_embeddings"
print(MODEL_NAME)

session_model = load_model(MODEL_NAME, session_model)

if (session_model == None):
    #mconf = GPTConfig(len(tokens), dataset.block_size, n_layer=layers, n_head=heads, n_embd=embbedings)
    session_model = GPT(mconf)
    tconf = TrainerConfig(max_epochs=epochs, 
                          batch_size=batch_size, 
                          learning_rate=learning_rate, 
                          num_workers=num_workers
                          )
    writer = SummaryWriter(log_dir='../runs/'+'logs') 
    trainer = Trainer(session_model, dataset, validation, tconf, writer)
    trainer.train()
    save_model(MODEL_NAME, session_model)
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()