import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head = heads
        # We take an embedding and split in into heads -- 8
        self.head_dimension = embed_size // heads

        # 256 // 8 would be good, but things like 256 // 7 is not possible
        assert (self.head_dimension * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dimension, embed_size)

    def forward(self, value, key, query, mask):
        N = query.shape[0]
        value_len, query_len, key_len = value.shape[1], query.shape[1], key.shape[1]

        # print("Original value:",value.shape, value.shape[0]*value.shape[1]*value.shape[2])
        # Split embeddings into head pieces where each head having head_dimension
        value = value.reshape(N, value_len, self.head, self.head_dimension)
        key = key.reshape(N, key_len, self.head, self.head_dimension)
        query = query.reshape(N, query_len, self.head, self.head_dimension)

        # Sending Q, K, V through Linear layer
        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)

        # Attention (Q, K, V) = softmax(_Q__*_Kᵀ_) * V
        #                                 √ dₖ

        # Q * Kᵀ
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        # Query Shape: (N, query_len, heads, head_dim)
        # Key shape: (N, key_len, heads, head_dim)
        # Energy: (N, heads, query_len, key_len)
        # For each word in target, how much should be pay attention to each word in input

        if mask is not None:  # Masking in decoder part if a mask is sent
            energy = energy.masked_fill(mask == 0, float("1e20"))

        # Normalize across the key_len --> dim=3
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, value])
        # Attention shape = (N, heads, query_len, key_len)
        # Values shape = (N, value_len, heads, head_dim)
        # Resultant = (N, query_len, heads, head_dim)

        # Revert to original_embed size = (N, query_len, head*head_dim)
        out = out.reshape(N, query_len, self.head * self.head_dimension)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    # inputs are sent to through multi-headed_attention -> norm -> feed forward -> norm
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))  # Skip connection from before attention to norm
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Skip connection from after 1st norm to 2nd norm
        return out


class Encoder(nn.Module):
    def __init__(
            self, src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # Max length of a sequence will be embedded to embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(  # Transformer block is used N times "Nx to the left of Transformer block"
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # Every example will have 0, 1, 2, 3...seq_len -> shape: (N, seq_length)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)

        # Positional encoding
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
            # In encoder Value, Key, Query are the same

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        # print("Original value:", value.shape, value.shape[0]*value.shape[1]*value.shape[2])
        # value = value.reshape(value.shape[0], value.shape[1], 8, 512//8)
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        # Max length of a sequence will be embedded to embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Decoder block is used multiple times in decoder "Nx to the right of the image"
        self.layers = [
            DecoderBlock(
                embed_size, heads, forward_expansion, dropout
            )
            for _ in range(num_layers)
        ]

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_length = x.shape
        # Every example will have 0, 1, 2, 3...seq_len -> shape: (N, seq_length)
        # Transformers are permutationally invariant
        # Therefore, positions of the words changes the output
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)

        # Positional encoding
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=512,
            num_encoder_layers=6,
            num_decoder_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, embed_size, num_encoder_layers, heads, forward_expansion, dropout, max_length
        )
        self.decoder = Decoder(
            trg_vocab_size, embed_size, num_decoder_layers, heads, forward_expansion, dropout, max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # To add up shape in attention (N, src_len) -> (N, 1, 1, src_len)
        return src_mask.to(device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # 1 for each example in sequence
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(device)

    # Masks are important especially for decoder
    # Without mask, the machine just map the words after one other
    # i.e; every word has access to the all the words only before it
    # It forms a lower triangular matrix
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        return decoder_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    # print(x.shape, trg.shape)
    out = model(x, trg[:, :-1])  # Not passing the eos token -> We want the model to learn to predict eos
    print(out.shape)  # Output: torch.Size([2, 7, 10])


