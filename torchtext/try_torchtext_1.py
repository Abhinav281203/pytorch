from torchtext.data import Field, TabularDataset, BucketIterator
import spacy

# 1. How preprocessing should be done -> Fields
# 2. Loading dataset of format JSON/CSV/TSV -> TabularDataset
# 3. Batching and padding of train, validation and test data -> BucketIterator

spacy_en = spacy.load("en_core_web_sm") 

def tokenize(text): # Tokenize the sentence using english tokenizer
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {"quote": ("q", quote), "score": ("s", score)}
# Field name in json file, their corresponding, preprossessing steps

train_data, test_data = TabularDataset.splits(
    path="", 
    train="train.json", 
    test="test.json", 
    format="json", # JSON/CSV/TSV
    fields=fields
)

# Build vocabulary of the words
quote.build_vocab(train_data, max_size=10000, min_freq=1)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=2, device=device
)

print(train_data[0].__dict__.keys()) # dict_keys(['q', 's'])
print(train_data[0].__dict__.values()) # dict_values([['success', 'is', 'not', 'final', ',', 'failure', 'is', 'not', 'fatal', ':', 'it', 'is', 'the', 'courage', 'to', 'continue', 'that', 'counts', '.'], 1])

# Iterators of data each loop - one batch
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=64, 
    device="cpu"
)

# for batch in train_iterator:
#     print(batch.q)
#     print(batch.s)