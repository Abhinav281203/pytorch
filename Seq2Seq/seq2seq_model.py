import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import random
import spacy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint, bleu,  translate_sentence


spacy_german = spacy.load('de_core_news_sm') # German tokenizer
spacy_english = spacy.load('en_core_web_sm') # English tokenizer

def tokenizer_german(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]

def tokenizer_english(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]

german = Field(tokenizer_german, lower=True,
               init_token='<sos>', eos_token='<eos>') # Define preprocessing

english = Field(tokenizer_english, lower=True,
               init_token='<sos>', eos_token='<eos>') # Define preprocessing

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english),
                                                         root='data')

german.build_vocab(train_data, max_size=10000, min_freq=2) # Build vocabulary
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size) # embedding_size -> d diminesion
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
    
    def forward(self, x): # shape of x: (seq_length, N)
        embedding = self.dropout(self.embedding(x)) # x is a vector of indexes of words in the vocabulary
        # (seq_length, N) -> (seq_length, N, embedding_size) Each word is a d diminsion vector
        outputs, (hidden, cell) = self.rnn(embedding)
        
        return hidden, cell # In seq2seq we only care about context vector from encoder


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell): 
        # shape of x: (N) but we want (1, N) --- x is Previous predicted
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x)) # embedding: (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # output: (1, N, hidden_size)
        prediction = self.fc(outputs) # prediction: (1, N, length_of_vocab)
        prediction = prediction.squeeze(0) # -> (N, length_of_vocab)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        # teacher_force_ratio:
        # While training, 50% of the time the previous output are fed to next word prediction
        # Regardless of it's correctness, and other 50% of the time the correct one is fed from training set

        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_len = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_len).to(device)
        # 1 word at a time of batch size and each of the word have target_vocab_len

        hidden, cell = self.encoder(source) # Get context vector

        x = target[0] # Grabbing start token

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output # In line 87, that 1 word is this
            best_guess = output.argmax(1) # (N, length_of_vocab) 1 -> probability of each word in vocab

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    

# Training
# Hyperparameters
num_epochs = 30
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size_decoder = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 3
encoder_dropout = 0.5
decoder_dropout = 0.5

writer = SummaryWriter(f'runs/Seq2Seq/Tensorboard')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src), # make almost entire batch have same length, Padding is same. so, save on compute
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, 
                      encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, 
                      decoder_dropout, output_size_decoder).to(device)

seq2seq_model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # Loss function to ignore padding
optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), seq2seq_model, optimizer)

sentence = 'Drei Mädchen stehen vor einem Fenster eines Gebäudes.'
print("Original Sentence:", sentence)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch} / {num_epochs}]")
    epoch_loss = 0.0

    checkpoint = {'state_dict': seq2seq_model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    seq2seq_model.eval()
    translated_sentence = translate_sentence(seq2seq_model, sentence, german, english, device, max_length=50)
    print(translated_sentence)
    seq2seq_model.train()

    for batch_idx, batch in  enumerate(tqdm(train_iterator)):
        input_data = batch.src.to(device)
        target_data = batch.trg.to(device)

        output = seq2seq_model(input_data, target_data) # (target_len, batch_size, english_vocab_length)
        output = output[1:].reshape(-1, output.shape[2])
        target_data = target_data[1:].reshape(-1) # [1:] - Ignoring start token

        optimizer.zero_grad()
        loss = criterion(output, target_data)

        loss.backward()

        # Make sure gradients are in healthy range (Exploding gradient problem)
        torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step= step)
        step += 1

        epoch_loss += loss.item()
    
    print(f'Loss: {epoch_loss / len(train_iterator)}')
