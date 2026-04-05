import numpy as np
from fygrad.node import Node, xp
from fygrad.module import Module, Linear, ReLU, Softmax, Embedding, ScaledDotProductAttention, LayerNorm, PositionalEncoding

PAD = 0
SOS = 6
EOS = 7
VOCAB_SIZE = EOS + 1
MAX_SEQ_LEN = 10

np.random.seed(42)


class MutliHeadAttention(Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = [
            Node.randn(
                f"Wq{h}", (d_model, self.d_k), scale=xp(self.device).sqrt(1 / d_model)
            )
            for h in range(num_heads)
        ]
        self.Wk = [
            Node.randn(
                f"Wk{h}", (d_model, self.d_k), scale=xp(self.device).sqrt(1 / d_model)
            )
            for h in range(num_heads)
        ]
        self.Wv = [
            Node.randn(
                f"Wv{h}", (d_model, self.d_k), scale=xp(self.device).sqrt(1 / d_model)
            )
            for h in range(num_heads)
        ]
        for i in range(num_heads):
            self.add_parameter(f"Wq{i}", self.Wq[i])
            self.add_parameter(f"Wk{i}", self.Wk[i])
            self.add_parameter(f"Wv{i}", self.Wv[i])

        self.Wo = Node.randn(
            "Wo", (d_model, d_model), scale=xp(self.device).sqrt(1 / d_model)
        )
        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        heads = []
        for h in range(self.num_heads):
            Qh = Node.matmul(Q, self.Wq[h], device=self.device)
            Kh = Node.matmul(K, self.Wk[h], device=self.device)
            Vh = Node.matmul(V, self.Wv[h], device=self.device)
            heads.append(self.attn(Qh, Kh, Vh, mask=mask))

        multi = heads[0]
        for head in heads[1:]:
            multi = Node.concat(multi, head, device=self.device)
        return Node.matmul(multi, self.Wo, device=self.device)


class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.fc1 = Linear(d_model, d_ff, label="ff1")
        self.fc2 = Linear(d_ff, d_model, label="ff2")
        self.relu = ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()

        self.attn = MutliHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: Node, mask=None):
        label = str(x)
        attend = self.attn(x, x, x, mask=mask)
        x = self.norm1(attend + x)
        fed = self.ff(x)
        x = self.norm2(fed + x)
        x.label = f"encoder({label})"
        return x


class DecoderBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()

        self.masked_attn = MutliHeadAttention(d_model, num_heads)
        self.cross_attn = MutliHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_out, mask=None, encoder_mask=None):
        self_attended = self.masked_attn(x, x, x, mask=mask)
        x = self.norm1(self_attended + x)
        cross_attended = self.cross_attn(x, encoder_out, encoder_out, mask=encoder_mask)
        x = self.norm2(cross_attended + x)
        fed = self.ff(x)
        return self.norm3(fed + x)


class Transformer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.src_embed = Embedding((src_vocab_size, d_model), "src")
        self.tgt_embed = Embedding((tgt_vocab_size, d_model), "tgt")

        self.encoders, self.decoders = [
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ], [
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        for i in range(num_layers):
            self.add_module(f"enc{i}", self.encoders[i])
            self.add_module(f"dec{i}", self.decoders[i])

        self.proj = Linear(d_model, tgt_vocab_size, label="proj")
        self.softmax = Softmax()

    def __mask(self, v: np.ndarray):
        return xp(self.device).where(v == PAD, -xp(self.device).inf, 0.0).reshape(1, -1)
    
    def __causal_mask(self, T: int):
        mask = xp(self.device).zeros((T, T))
        mask[xp(self.device).triu_indices(T, k=1)] = -np.inf
        return mask

    def encode(self, src: np.ndarray):
        x = self.src_embed(src)
        src_mask = self.__mask(src)

        x = self.pos_enc(x)
        for block in self.encoders:
            x = block(x, mask=src_mask)
        return x, src_mask

    def decode(self, tgt, encoder_out, encoder_mask=None):
        x = self.tgt_embed(tgt)
        tgt_pad = self.__mask(tgt)

        T = x.shape[0]
        causal = self.__causal_mask(T)
        decoder_mask = causal + tgt_pad

        x = self.pos_enc(x)
        for block in self.decoders:
            x = block(x, encoder_out, mask=decoder_mask, encoder_mask=encoder_mask)
        return x

    def forward(self, src, tgt):
        encoder_out, src_mask = self.encode(src)
        decoder_out = self.decode(tgt, encoder_out, encoder_mask=src_mask)
        return self.softmax(self.proj(decoder_out))
    

model = Transformer(
    d_model=32,
    num_heads=4,
    d_ff=32,
    num_layers=1,
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
)

model.load("test/transformer.json")
# optimizer = SGD(model.parameters(), lr=0.01)

def make_input(seq: list):
    src = seq + [PAD] * (MAX_SEQ_LEN - len(seq))
    tgt_in = [SOS] + list(reversed(seq)) + [PAD] * (MAX_SEQ_LEN - len(seq) - 1)
    tgt_out = list(reversed(seq)) + [EOS] + [PAD] * (MAX_SEQ_LEN - len(seq) - 1)
    return np.array(src), np.array(tgt_in), np.array(tgt_out)

# def random_example():
#     length = np.random.randint(2, MAX_SEQ_LEN)
#     seq = np.random.randint(1, VOCAB_SIZE-2, size=length).tolist()
#     return make_input(seq)

# epochs = 1000
# samples = 100
# for epoch in range(epochs):
#     total_loss = 0

#     for i in range(samples):
#         src, tgt_in, tgt_out = random_example()
#         optimizer.zero_grad()

#         probs = model(src, tgt_in)

#         loss = Node.cross_entropy(probs, tgt_out, model.device)
#         mask = Node("mask", (tgt_out != PAD).astype(np.float32).reshape(-1, 1), device=model.device)
#         loss = Node.mean(loss * mask, axis=0)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.value.squeeze()

#     if epoch % 50 == 0:
#         print(f"epoch {epoch:3d}  loss {total_loss / samples:.4f}")

# model.save("test/transformer.json")

src, tgt_in, tgt_out = make_input([3, 2, 1, 2])
print("actual:", src[:np.argmin(src)])

out = model(src, tgt_in)
print("expect:", tgt_out[:np.argmax(tgt_out)])
result = np.argmax(out.value, axis=1)
print("result:", result[:np.argmax(result)])
