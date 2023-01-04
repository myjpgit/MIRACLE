import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

from itertools import repeat
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce as coalesce_fn
import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from utils import get_link_labels, evaluate_AUC, link_prediction, generate_pos_edge_index, generate_neg_edge_index
from collections import defaultdict

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_index_file(filename):
    """Parse index file.解析索引文件 """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask.创建任务划分"""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def generate_data(edge_index, num_nodes):
    graph = defaultdict(list)
    for j in range(num_nodes):
        graph[int(j)].append(int(j))
    for i in range(len(edge_index[0])):
        graph[int(edge_index[0][i])].append(int(edge_index[1][i]))
        graph[int(edge_index[1][i])].append(int(edge_index[0][i]))
    return graph

def load_citation_network(dataset_str, sparse=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_str)
    if dataset_str in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, dataset_str)
    elif dataset_str in ["samecity", "terror", "advisor"]:
        dataset = TUDataset(path, dataset_str, T.NormalizeFeatures(), use_node_attr=True, use_edge_attr=True)
    #dataset = Planetoid(path, dataset_str)
    data = dataset[0]

    features = data.x  # 得到整个数据的全部特征矩阵
    # data.pos_edge_index = generate_pos_edge_index(data.edge_index, data.y)
    # graph = generate_data(data.pos_edge_index, data.num_nodes)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(data.graph)) # 得到邻接矩阵
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = torch.tensor(data.y , dtype=torch.int64)  # 得到整个数据的全部标签
    edge_index = data.edge_index
    labels = torch.LongTensor(labels) # 将label转换为torch.Long

    nfeats = features.shape[1] # 特征维度
    nclasses = torch.max(labels).item() + 1 # labels中数字最大的值对应节点总共的类别数

    return data, features, nfeats, labels, nclasses, adj, edge_index

def load_data(args):
    return load_citation_network(args.dataset, args.sparse)

def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=True):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    if coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce_fn(edge_index, None, num_nodes, num_nodes)
    return edge_index