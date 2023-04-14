import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Split_Attention(nn.Module):
    def __init__(self, args):
        self.args = args
        super(FC_Split_Attention, self).__init__()
        self.readin = nn.Linear(self.args.num_features, self.args.embedding_size)
        self.leftFC = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.rightFC1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.rightFC2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.middFC1 = nn.Linear(self.args.embedding_size, self.args.num_features)
        self.middFC2 = nn.Linear(self.args.num_features, self.args.num_features)
        self.lastFC = nn.Linear(self.args.embedding_size, 1)
        self.readout = nn.Linear(self.args.num_features, 1)
        # self.relu=nn.LeakyReLU(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.LN_embedding_size = nn.LayerNorm(self.args.embedding_size)
        self.LN_num_features = nn.LayerNorm(self.args.num_features)
        self.BN_num_features = nn.BatchNorm1d(self.args.num_features)

    def forward(self, x):
        x_in = self.dropout(self.relu(self.readin(x)))
        left_x = self.leftFC(x_in)
        right_x = self.rightFC1(x_in)
        right_x = right_x + left_x

        # right_x = self.LN_embedding_size(right_x)

        right_x = self.rightFC2(right_x)
        midd_x = left_x + right_x

        # midd_x = self.LN_embedding_size(midd_x)

        midd_x = self.dropout(self.relu(self.middFC1(midd_x)))
        midd_x = self.BN_num_features(midd_x)

        midd_x = self.softmax(self.middFC2(midd_x))
        midd_x = midd_x.unsqueeze(-1)
        left_x = left_x.unsqueeze(-2)
        right_x = right_x.unsqueeze(-2)
        midd_x1 = torch.matmul(midd_x, left_x)
        midd_x2 = torch.matmul(midd_x, right_x)
        midd_x = midd_x1 + midd_x2

        midd_x = self.lastFC(midd_x).squeeze()
        x = x + midd_x

        x = self.LN_num_features(x)

        y = self.readout(x).squeeze()

        return y


class FCNET(nn.Module):
    def __init__(self, args):
        super(FCNET, self).__init__()
        self.args = args
        embedding_size = 128
        # self.net = nn.Sequential(
        #     nn.Linear(self.args.num_features, 64), nn.LeakyReLU(), nn.Dropout(p=self.args.dropout),
        #     nn.Linear(64, 64), nn.LeakyReLU(), nn.LayerNorm(64), nn.Dropout(p=self.args.dropout),
        #     nn.Linear(64, 2), nn.Sigmoid()
        # )
        self.fc1 = nn.Linear(self.args.num_features, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.fc4 = nn.Linear(embedding_size, embedding_size)
        self.fc5 = nn.Linear(embedding_size, 1)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=self.args.dropout)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x1 = self.dp(self.relu(self.fc1(x)))
        x2 = self.dp(self.relu(self.fc2(x1)))
        x2 = self.layernorm(x2 + x1)
        x3 = self.dp(self.relu(self.fc3(x2)))
        x3 = self.layernorm(x3 + x2)
        x4 = self.dp(self.relu(self.fc4(x3)))
        x4 = self.layernorm(x4 + x3)
        x5 = self.sigmoid(self.fc5(x4)).squeeze()
        # print(x3[0])
        # x=torch.argmax(x5,dim=-1).float()
        return x5


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(self.args.num_features, 128), nn.LeakyReLU(0.33), nn.BatchNorm1d(128),
            nn.Dropout(p=self.args.dropout),
            nn.Linear(128, 512), nn.LeakyReLU(0.33), nn.Dropout(p=self.args.dropout),
            nn.Linear(512, 128), nn.LeakyReLU(0.33), nn.LayerNorm(128)
        )
        self.feature_net = Feature_Net(args)
        self.FCTransformerEncoder = FCTransformerEncoder(args)
        self.readout = Readout(args)

    def forward(self, x):
        x = self.net(x)
        x = self.feature_net(x)
        x = self.FCTransformerEncoder(x)
        x = torch.sum(x, dim=-1)
        x = self.readout(x)
        # x = torch.argmax(x,dim=-1).float()
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
        self.relu = nn.LeakyReLU(0.33)
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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x):
        x1 = self.dropout(self.relu(self.fc1(x)))
        x2 = self.relu(self.fc2(x1)).squeeze()

        return x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, args):
        super(FeedForwardNetwork, self).__init__()
        self.args = args

        hidden_size = args.embedding_size
        ffn_size = args.encoder_ffn_size
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.LeakyReLU = nn.LeakyReLU(self.args.negative_slope)
        self.ReLU = nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
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
