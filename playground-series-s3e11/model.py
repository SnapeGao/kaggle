import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(self.args.num_features, 128), nn.LeakyReLU(), nn.BatchNorm1d(128),
            nn.Dropout(p=self.args.dropout),
            nn.Linear(128, 512), nn.LeakyReLU(), nn.Dropout(p=self.args.dropout),
            nn.Linear(512, 128), nn.LeakyReLU(), nn.LayerNorm(128)
        )
        self.feature_net = Feature_Net(args)
        self.FCTransformerEncoder = FCTransformerEncoder(args)
        self.readout = Readout(args)

    def forward(self, x):
        x = self.net(x)
        x = self.feature_net(x)
        x=self.FCTransformerEncoder(x)
        x = torch.sum(x, dim=-1)
        x = self.readout(x)
        return x


class Feature_Net(nn.Module):
    def __init__(self, args):
        super(Feature_Net, self).__init__()
        self.args = args
        dim = args.embedding_size
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)
        self.relu = nn.LeakyReLU(self.args.negative_slope)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x):
        B, d = x.size()
        x = x.reshape(B, d, 1)
        x1 = self.dropout(self.relu(self.fc1(x)))
        x2 = self.dropout(self.relu(self.fc2(x1)))
        x3 = self.fc3(x2)

        x = x + x3
        x = self.LN(x)
        return x


class Readout(nn.Module):
    def __init__(self, args):
        super(Readout, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.LN = nn.LayerNorm(128)
        self.relu = nn.LeakyReLU(self.args.negative_slope)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x):
        x1 = self.dropout(self.relu(self.fc1(x)))
        x2 = self.fc2(x1).squeeze()

        return x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, args):
        super(FeedForwardNetwork, self).__init__()
        self.args = args

        hidden_size = args.embedding_size
        ffn_size = args.encoder_ffn_size
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.LeakyReLU = nn.LeakyReLU(self.args.negative_slope)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.LeakyReLU(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        self.num_heads = num_heads = args.n_heads
        embedding_size = args.embedding_size

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = q if q else nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_k = k if k else nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.att_dropout = nn.Dropout(args.dropout)

        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3).transpose(-1, -2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, d_v).transpose(-2, -3)

        q = q * self.scale
        a = torch.matmul(q, k)
        a = self.att_dropout(a)
        y = a.matmul(v).transpose(-2, -3).contiguous().view(batch_size, -1, self.num_heads * d_v)
        y = self.output_layer(y)

        return y


class FCTransformerEncoder(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(FCTransformerEncoder, self).__init__()
        self.args = args
        self.d_k = args.embedding_size

        self.self_attention = MultiHeadAttention(args, q, k)
        self.self_attention_dropout = nn.Dropout(args.dropout)
        self.self_attention_norm = nn.LayerNorm(args.embedding_size)

        self.ffn = FeedForwardNetwork(args)
        self.ffn_dropout = nn.Dropout(args.dropout)
        self.ffn_norm = nn.LayerNorm(args.embedding_size)

    def forward(self, x):
        self_att_result = self.self_attention(x)
        self_att_result = self.self_attention_dropout(self_att_result)
        self_att_result = x + self_att_result
        self_att_result = self.self_attention_norm(self_att_result)

        ffn_result = self.ffn(self_att_result)
        ffn_result = self.ffn_dropout(ffn_result)
        self_att_result = self_att_result + ffn_result
        ffn_result = self.ffn_norm(self_att_result)
        encoder_result = x + ffn_result
        return encoder_result
