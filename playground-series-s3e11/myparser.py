import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../datasets/playground-series-s3e11/')
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--seed", default=2023)

parser.add_argument("--hidden_dim1", type=int, default=128)
parser.add_argument("--hidden_dim2", type=int, default=512)
parser.add_argument("--hidden_dim3", type=int, default=512)
parser.add_argument("--hidden_dim4", type=int, default=64)
parser.add_argument("--out_dim", type=int, default=1)
parser.add_argument("--embedding_size", type=int, default=64)
parser.add_argument("--encoder_ffn_size", type=int, default=128)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--msa_bias", type=bool, default=True)

# contrastive training parameters
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--negative_slope', type=float, default=0.05)
# 模型设置
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
parser.add_argument('--lr_schedule_patience', type=int, default=300)
parser.add_argument('--min_lr', type=float, default=1e-6)
parsed_args = parser.parse_args()
