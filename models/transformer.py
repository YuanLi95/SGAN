import torch
import torch.nn as nn
import  numpy as np
import torch.nn.functional as F
class Transformer(nn.Module):

    def __init__(self,
               src_vocab_size,
               src_max_len,
               tgt_vocab_size,
               tgt_max_len,
               num_layers=6,
               model_dim=300,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, ctx_attn = self.decoder(
          tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        # print(d_model)
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        # print(d_model)
        position_encoding = torch.from_numpy(position_encoding).float()



        position_encoding = torch.cat((pad_row, position_encoding))
        # print(position_encoding.shape)
        # exit()

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置



        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len).item() for len in input_len])

        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=300, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim,model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        # print(output.shape)
        # output = x
        # print(output.shape)
        #
        # print((self.w1(output)).shape)
        output = self.w2(torch.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):


    def __init__(self, model_dim=300, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs,attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
	#多层EncoderLayer组成Encoder
    def __init__(self,
             embedding_matrix,
               max_seq_len,
               num_layers=6,
               model_dim=300,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        # print(inputs[1])
        output = self.seq_embedding(inputs)
        # print(output[1,:,:])
        # print(output[1,:,:].shape)

        # print(output.shape)
        output += self.pos_embedding(inputs_len)
        # print(output.shape)

        self_attention_mask = padding_mask(inputs, inputs)
        # print(self_attention_mask[1])
        # print(self_attention_mask[1].shape)
        # exit()

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output,self_attention_mask)
            attentions.append(attention)

        return output, attentions

class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
                dec_inputs,
                enc_outputs,
                self_attn_mask=None,
                context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
            enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention

class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=300,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        # print(model_dim)
        # exit()
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

def residual(sublayer_fn,x):
	return sublayer_fn(x)+x

class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """Init.

        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """前向传播.

        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

def padding_mask(seq_k, seq_q):
	# seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, key_depth, value_depth, output_depth, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self._query = nn.Linear(key_depth, key_depth, bias=False)
        self._key = nn.Linear(key_depth, key_depth, bias=False)
        self._value = nn.Linear(value_depth, value_depth, bias=False)
        self.output_perform = nn.Linear(value_depth, output_depth, bias=False)
        self.dot_product_attention=ScaledDotProductAttention(attention_dropout=dropout)
        self.num_heads = num_heads
        self.key_depth_per_head = key_depth // num_heads
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout = dropout
		# multi-head attention之后需要做layer norm
        self. layer_norm = nn.LayerNorm(key_depth)

    def forward(self, key, value, query, attn_mask=None,to_weights=True):
		# 残差连接

        # num_heads = self.num_heads

        # linear projection
        key = self._key(key)
        value = self._value(value)
        query = self._query(query)

        # split by heads
        # split heads
        query *= self.key_depth_per_head ** -0.5
        q = split_heads(query, self.num_heads)
        k = split_heads(key, self.num_heads)
        v = split_heads(value, self.num_heads)


        # scaled dot product attention


        # context, attention = self.dot_product_attention(
        #   query, key, value, scale, attn_mask)
        x = []
        for i in range(self.num_heads):
            results = self.dot_product_attention(q[i], k[i], v[i],None,)
            if to_weights:
                y, attn_scores = results
            x.append(y)
        x_combine = combine_heads(x)
        # final linear projection
        # print(context.shape)
        output = self.output_perform(x_combine)
        # print(output.shape)

        # dropout
        output = self.dropout_layer(output)

        # add residual and norm layer
        # print(output.shape)
        # print(residual.shape)
        if to_weights:
            return output, attn_scores / self.num_heads
        else:
            return output

def combine_heads(x):
    """combine multi heads
    Args:
        x: [batch_size, length, depth / num_heads] x heads
    Returns:
        x: [batch_size, length, depth]
    """
    return torch.cat(x, 2)
def split_heads(x, num_heads):
    """split x into multi heads
    Args:
        x: [batch_size, length, depth]
    Returns:
        y: [[batch_size, length, depth / num_heads] x heads]
    """
    sz = x.size()
    # x -> [batch_size, length, heads, depth / num_heads]
    x = x.view(sz[0], sz[1], num_heads, sz[2] // num_heads)
    # [batch_size, length, 1, depth // num_heads] *
    heads = torch.chunk(x, num_heads, 2)
    x = []
    for i in range(num_heads):
        x.append(torch.squeeze(heads[i], 2))
    return x

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.dropout = attention_dropout
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        # print(q.shape)
        # print(k.shape)

        attention = torch.bmm(q, k.transpose(1, 2))
        # if scale is not None:
        #     attention = attention * scale
        # if attn_mask is not None:
        #     # 给需要mask的地方设置一个负无穷
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
            # 计算softmax
        attention = torch.softmax(attention,dim=2)
        # 添加dropout
        attention = torch.dropout(attention,self.dropout,train=True)
        # 和V做点积
        # context = torch.bmm(attention, v)
        return v, attention