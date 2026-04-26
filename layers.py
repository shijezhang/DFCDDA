import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FGGCN(nn.Module):
    def __init__(self, dr_num, di_num, hid_dim, out_dim, dropout):
        super(FGGCN, self).__init__()

        self.FGCN1 = GCN(dr_num, hid_dim, out_dim, dropout)
        self.FGCN2 = GCN(di_num, hid_dim, out_dim, dropout)
        self.dropout = dropout

    def forward(self, drf_knn, drf, drg_knn, dip_knn, dip, dig_knn):
        emb1 = self.FGCN1(drf, drf_knn, drg_knn)
        emb2 = self.FGCN2(dip, dip_knn, dig_knn)
        return emb1, emb2


class GCN(nn.Module):
    def __init__(self, features, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(features, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj1, adj2):
        x = F.relu(self.gc1(x, adj1, adj2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj1, adj2)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, use_attention=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_attention = use_attention
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.node_level_fusion = NodeLevelFusion(in_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj1, adj2):
        device = x.device
        adj1 = adj1.to(device)
        adj2 = adj2.to(device)

        if adj1.is_sparse:
            adj1 = adj1.to_dense()
        if adj2.is_sparse:
            adj2 = adj2.to_dense()

        att1 = self.node_level_fusion(x)
        adj = att1 * adj1 + (1 - att1) * adj2

        support = th.mm(x, self.weight)
        output = th.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class NodeLevelFusion(nn.Module):
    def __init__(self, in_features, hidden_size=16):
        super(NodeLevelFusion, self).__init__()
        self.fc1 = nn.Linear(in_features * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        N = x.size(0)

        x_i = x.unsqueeze(1).expand(-1, N, -1)
        x_j = x.unsqueeze(0).expand(N, -1, -1)
        edge_features = th.cat([x_i, x_j], dim=-1)

        att = self.fc1(edge_features)
        att = self.activation(att)
        att = self.fc2(att).squeeze(-1)
        att = th.sigmoid(att)

        return att


class BiCrossAtt(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_heads):
        super(BiCrossAtt, self).__init__()

        self.dr_proj1 = nn.Sequential(
            nn.Linear(embed_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.di_proj1 = nn.Sequential(
            nn.Linear(embed_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.dr_proj2 = nn.Linear(hid_dim, embed_dim)
        self.di_proj2 = nn.Linear(hid_dim, embed_dim)

        self.dr_ca = nn.MultiheadAttention(hid_dim, num_heads)
        self.di_ca = nn.MultiheadAttention(hid_dim, num_heads)

        self.dr_norm = nn.LayerNorm(embed_dim)
        self.di_norm = nn.LayerNorm(embed_dim)

    def _init_weights(self):
        for module in [self.dr_proj1, self.di_proj1, self.dr_proj2, self.di_proj2]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, dr, di):
        or_dr = dr
        or_di = di

        dr = self.dr_proj1(dr)
        di = self.di_proj1(di)

        dr = dr.unsqueeze(1)
        di = di.unsqueeze(1)

        dr_fu, _ = self.dr_ca(query=dr, key=di, value=di)
        di_fu, _ = self.di_ca(query=di, key=dr, value=dr)

        dr_fu = dr_fu.squeeze(1)
        di_fu = di_fu.squeeze(1)

        dr_fu = self.dr_proj2(dr_fu)
        di_fu = self.di_proj2(di_fu)

        dr_fu = self.dr_norm(dr_fu + or_dr)
        di_fu = self.di_norm(di_fu + or_di)

        return dr_fu, di_fu


class AdaFusion(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(AdaFusion, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Decoder(nn.Module):
    def __init__(self,
                 in_dim,
                 dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.lin1 = nn.Linear(2 * in_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, dr_fu, di_fu):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = dr_fu
            graph.nodes['disease'].data['h'] = di_fu
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']
            out = F.relu(self.lin1(out))
            out = self.dropout(out)
            out = F.relu(self.lin2(out))
            out = self.dropout(out)
            out = self.lin3(out)

        return out


def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}
