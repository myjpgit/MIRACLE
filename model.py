import copy
import math

import torch

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):

        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(args, x, x_aug, x_edge, temperature=0.2, sym=True):  # Calculating contrastive loss
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)  # anchor graph embedding 求指定维度上的范数
        x_edge_abs = x_edge.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)  # learned graph embedding

        sim_matrix_1 = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix_1 = torch.exp(sim_matrix_1 / temperature)
        pos_sim_1 = sim_matrix_1[range(batch_size), range(batch_size)]

        sim_matrix_2 = torch.einsum('ik,jk->ij', x_edge, x_aug) / torch.einsum('i,j->ij', x_edge_abs, x_aug_abs)
        sim_matrix_2 = torch.exp(sim_matrix_2 / temperature)
        pos_sim_2 = sim_matrix_1[range(batch_size), range(batch_size)]

        if sym:
            loss_0 = pos_sim_1 / (sim_matrix_1.sum(dim=0) - pos_sim_1)
            loss_1 = pos_sim_1 / (sim_matrix_1.sum(dim=1) - pos_sim_1)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()

            loss_2 = pos_sim_2 / (sim_matrix_1.sum(dim=0) - pos_sim_2)
            loss_3 = pos_sim_2 / (sim_matrix_1.sum(dim=1) - pos_sim_2)

            loss_2 = - torch.log(loss_2).mean()
            loss_3 = - torch.log(loss_3).mean()

            loss_aug = (loss_0 + loss_1) / 2.0
            loss_edge = (loss_2 + loss_3) / 2.0

            loss = args.loss_alpha * loss_aug + (1 - args.loss_alpha) * loss_edge
            return loss
        else:
            loss_1 = pos_sim_1 / (sim_matrix_1.sum(dim=1) - pos_sim_1)
            loss_1 = - torch.log(loss_1).mean()

            loss_2 = pos_sim_2 / (sim_matrix_2.sum(dim=1) - pos_sim_2)
            loss_2 = - torch.log(loss_2).mean()

            loss = args.loss_alpha * loss_1 + (1 - args.loss_alpha) * loss_2

            return loss