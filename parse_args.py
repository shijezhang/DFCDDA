import argparse
from util.script_util import (diffusion_defaults, add_dict_to_argparser)

parser = argparse.ArgumentParser(description='DFCDDA')

parser.add_argument('--device', default='0', type=int)

parser.add_argument('--dataset', default='C-dataset', type=str)

parser.add_argument('--max_epochs', type=int, default=4000)

parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--dropout', type=float, default=0.3)

parser.add_argument('--grad_clip', type=float, default=1.0)

parser.add_argument('--knn_neighbor', type=int, default=5)

parser.add_argument('--alpha', type=float, default=1, help="the proportion of duffusion loss")

parser.add_argument('--beta', type=float, default=0.00001, help="the proportion of contrastive loss")

parser.add_argument('--intra_ssl_temperature', type=float, default=0.05)

parser.add_argument('--inter_ssl_temperature', type=float, default=0.05)

parser.add_argument('--hid_dim', type=int, default=500)

parser.add_argument('--embed_dim', type=int, default=256)

parser.add_argument('--hgt_layer', default=2, type=int)

parser.add_argument('--hgt_out_dim', default=256, type=int)

parser.add_argument('--hgt_head', default=4, type=int)

parser.add_argument('--tr_layer', default=2, type=int)

parser.add_argument('--tr_head', default=4, type=int)

parser.add_argument('--K_fold', type=int, default=10)

parser.add_argument('--dataset_percent', type=float, default=1)

parser.add_argument('--ca_embed_dim', type=int, default=64)

parser.add_argument('--ca_head', type=int, default=8)

parser.add_argument('--negative_rate', type=float, default=1)

parser.add_argument('--seed', default=125, type=int)

defaults = dict(hidden_dims=[1024, 1024, 512, 512])
defaults.update(diffusion_defaults())
add_dict_to_argparser(parser, defaults)

args = parser.parse_args()
