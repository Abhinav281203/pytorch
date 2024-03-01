import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, bleu, translate_sentence
from tqdm import tqdm

spacy_german = spacy.load('de_core_news_sm')  # German tokenizer
spacy_english = spacy.load('en_core_web_sm')  # English tokenizer

def tokenize_german(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]

def tokenize_english(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]


german = Field(tokenize_german, lower=True,
               init_token="<sos>", eos_token="<eos>")

english = Field(tokenize_english, lower=True,
                init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english), root="data"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
            self,
            embed_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_head,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_len, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_len, embed_size)
        self.device = device
        self.transformer = nn.Transformer(
            embed_size,
            num_head,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        # Pytorch needs it as (N, src_len)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(device)

    def forward(self, src, trg):
        src_seq_len, n = src.shape
        trg_seq_len, n = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, n).to(device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, n).to(device)
        )

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask
        )
        out = self.fc_out(out)
        return out

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Model hyperparams
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]
trg_pad_idx = english.vocab.stoi["<pad>"]

writer = SummaryWriter("runs/loss")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),  # make almost entire batch have same length, Padding is same. so, save on compute
    device=device
)

model = Transformer(
    embed_size=embedding_size,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    num_head=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_len=max_len,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = 'ein pferd geht unter einer brücke neben einem boot.'

for epoch in range(num_epochs):
    print(f"Epoch [{epoch} / {num_epochs}]")
    epoch_loss = []

    if save_model:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )
    print(translated_sentence)
    model.train()

    for batch_idx, batch in enumerate(tqdm(train_iterator)):
        input_data = batch.src.to(device)
        target_data = batch.trg.to(device)

        # Forward
        # print(input_data.shape, target_data.shape)
        output = model(input_data, target_data[:-1, :])
        output = output.reshape(-1, output.shape[2])  # For cross entropy to work
        target_data = target_data[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target_data)
        epoch_loss.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(epoch_loss) / len(epoch_loss)
    print("Loss:", mean_loss)
    scheduler.step(mean_loss)

score = bleu(test_data, model, german, english, device)
print(f"Bleu score {score*100:.2f}")
