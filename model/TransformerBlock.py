
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn

class MultiheadAttention(nn.Module):

    def __init__(self, input_size, output_size, d_k=16, d_v=16, num_heads=8, is_layer_norm=False, attn_dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, num_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v*num_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W_q)
        init.xavier_uniform_(self.W_k)
        init.xavier_uniform_(self.W_v)
        init.xavier_uniform_(self.W_o)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

    def feed_forword_layer(self, X):
        lay1 = F.relu(self.linear1(X))
        lay1 = self.dropout(lay1)

        output = self.linear2(lay1)
        return output

    def scaled_dot_product_attention(self, Q, K, V, key_padding_mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, num_heads, input_size)
        :param K: (*, max_k_words, num_heads, input_size)
        :param V: (*, max_v_words, num_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)

        if key_padding_mask is not None:
            bsz, src_len = Q.size(0) // self.num_heads, Q.size(1)
            tgt_len = V.size(1)
            Q_K = Q_K.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
            Q_K = Q_K.masked_fill(key_padding_mask, -2 ** 32 + 1)
            Q_K = Q_K.view(bsz * self.num_heads, tgt_len, src_len)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, key_padding_mask):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.num_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.num_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.num_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, key_padding_mask)
        V_att = V_att.view(bsz, self.num_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.num_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        '''
        :param query: (batch_size, max_q_words, input_size)
        :param key: (batch_size, max_k_words, input_size)
        :param value: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        bsz, src_len, _ = query.size()
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        V_att = self.multi_head_attention(query, key, value, key_padding_mask)

        if self.is_layer_norm:
            X = self.layer_morm(query + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.feed_forword_layer(X) + X)
        else:
            X = query + V_att
            output = self.feed_forword_layer(X) + X

        output = self.linear3(output)
        return output
