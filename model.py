from layers import *
import torch
import torch.nn as nn
import dgl.nn.pytorch
from dgl.nn.pytorch import HGTConv
from util.unet import Linear_UNet
from parse_args import args

th.set_printoptions(profile="full")


class DFCDDA(nn.Module):
    def __init__(self, args, meta_g):
        super(DFCDDA, self).__init__()
        self.meta_g = meta_g

        self.fggcn = FGGCN(args.dr_num, args.di_num, args.hid_dim, args.embed_dim, args.dropout)
        self.bicrossatt = BiCrossAtt(args.embed_dim, args.ca_embed_dim, args.ca_head)
        self.adafusion = AdaFusion(args.embed_dim)
        self.decoder = Decoder(args.embed_dim)

        self.hgt_dgl = HGTConv(args.embed_dim,
                               int(args.hgt_out_dim / args.hgt_head),
                               args.hgt_head,
                               len(self.meta_g.nodes()),
                               len(self.meta_g.edges()),
                               args.dropout)
        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(self.hgt_dgl)

        self.unet_x = Linear_UNet(input_dim=args.dr_num, cond_dim=args.embed_dim, hidden_dims=args.hidden_dims)
        self.unet_y = Linear_UNet(input_dim=args.di_num, cond_dim=args.embed_dim, hidden_dims=args.hidden_dims)

        self.dr_linear = nn.Linear(300, args.embed_dim)
        self.di_linear = nn.Linear(64, args.embed_dim)
        self.pr_linear = nn.Linear(320, args.embed_dim)
        self.shared_linear = nn.Linear(args.embed_dim, args.embed_dim)

    def _init_weights(self):
        mudules = [self.dr_linear, self.di_linear, self.pr_linear, self.shared_linear]
        for module in mudules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, drf_knn, drf, drg_knn, dip_knn, dip, dig_knn,
                drdipr_graph, dr_m, di_m, pr_m, dec_graph):

        dr_hete_init = self.dr_linear(dr_m)
        di_hete_init = self.di_linear(di_m)
        pr_hete_init = self.pr_linear(pr_m)

        feature_dict = {
            'drug': dr_hete_init,
            'disease': di_hete_init,
            'protein': pr_hete_init
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((dr_hete_init, di_hete_init, pr_hete_init), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:args.dr_num, :]
        di_hgt = hgt_out[args.dr_num:args.di_num + args.dr_num, :]

        dr_sim, di_sim = self.fggcn(drf_knn, drf, drg_knn, dip_knn, dip, dig_knn)

        dr_ca, di_ca = self.bicrossatt(dr_sim, di_sim)
        dr_sim = dr_ca + dr_sim
        di_sim = di_ca + di_sim

        dr_hgt = self.shared_linear(dr_hgt)
        di_hgt = self.shared_linear(di_hgt)
        dr_sim = self.shared_linear(dr_sim)
        di_sim = self.shared_linear(di_sim)

        dr = th.stack([dr_hgt, dr_sim], dim=1)
        di = th.stack([di_hgt, di_sim], dim=1)

        dr_fu, att_dr = self.adafusion(dr)
        di_fu, att_di = self.adafusion(di)

        output = self.decoder(dec_graph, dr_fu, di_fu)

        return output, dr_hgt, dr_sim, di_hgt, di_sim, dr_fu, di_fu
