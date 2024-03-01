import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

english_spacy = spacy.load("en_core_web_sm")
german_spacy = spacy.load("de_core_news_sm")


def tokenize_engish(text):
    return [tok.text for tok in english_spacy.tokenizer(text)]

def tokenize_german(text):
    return [tok.text for tok in german_spacy.tokenizer(text)]

english = Field(sequential=True, use_vocab=True, tokenize=tokenize_engish, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_german, lower=True)

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), 
    fields=(german, english)
)

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), 
    batch_size=64, 
    device="cpu"
)

for batch in train_iterator:
    print(batch)

print(english.vocab.stoi["the"])
print(english.vocab.itos[1612])