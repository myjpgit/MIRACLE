import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, roc_curve
from torch_geometric.utils import negative_sampling
from math import log

EOS = 1e-10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def link_prediction(data, node_embedding_matrix, mode="train"):
    if mode == "train":
        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                           # num_nodes=self.graph_data.num_nodes,
                                           num_neg_samples=data.train_pos_edge_index.size(1),
                                           force_undirected=True).to(device)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # edge_index = data.train_edge_index

    elif mode == "val":
        edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)

    elif mode == "test":
        edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)

    source_node_embedding = node_embedding_matrix[edge_index[0]]
    target_node_embedding = node_embedding_matrix[edge_index[1]]
    link_predict = (source_node_embedding * target_node_embedding).sum(dim=-1)
    link_predict = link_predict.sigmoid()

    return link_predict

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = len(pos_edge_index[0]) + len(neg_edge_index[0])
    link_labels = np.array([0 for i in range(num_links)])
    link_labels[:len(pos_edge_index[0])] = 1.
    return link_labels

def generate_pos_edge_index(edge_index, label):
    pos_edge_index = [[] for i in range(2)]
    for i in range(len(label)):
        if int(label[i]) == 1:
            pos_edge_index[0].append(int(edge_index[0][i]))
            pos_edge_index[1].append(int(edge_index[1][i]))
        else:
            pass
    # print(torch.tensor(pos_edge_index))
    return torch.tensor(pos_edge_index)

def generate_neg_edge_index(edge_index, label):
    neg_edge_index = [[] for i in range(2)]
    for i in range(len(label)):
        if int(label[i]) == 0:
            neg_edge_index[0].append(int(edge_index[0][i]))
            neg_edge_index[1].append(int(edge_index[1][i]))
        else:
            pass
    return torch.tensor(neg_edge_index)


def evaluate_AUC(y_predict, y_ture):
    y_predict = y_predict.to("cpu").detach().numpy().reshape(-1,1)
    y_ture = y_ture.to("cpu").detach().numpy()
    roc_auc_score_ = roc_auc_score(y_ture, y_predict)
    return roc_auc_score_

def evaluate_ks(y_predict, y_ture):
    '''
    :param y: array，真实值
    :param y_score: array，预测概率值
    :return: float
    '''
    y_predict = y_predict.to("cpu").detach().numpy().reshape(-1, 1)
    y_ture = y_ture.to("cpu").detach().numpy()
    fpr, tpr, thresholds = roc_curve(y_ture, y_predict)
    return max(tpr - fpr)

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    # return mask.cuda(), samples
    return mask.to(device), samples


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric) # 为X中的点计算k-Neighbors的(加权)图
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0]) # 添加自链接
    adj = adj * i - i
    return adj

def common_neighbor(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    return len(n1&n2)

def admic_adar(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    res=0.0
    if len(n1&n2)==0:
        return 0.0
    for k in n1&n2:
        if len(indices[k])>1:
            res+=1/log(len(indices[k]))
    return res

def jaccard(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    if len(n1)==0 or len(n2)==0:
        return 0.0
    return len(n1&n2)/len(n1|n2)

def edge_feature(args, node_num, edge_index):
    node_node=sp.lil_matrix((node_num,node_num),dtype=np.float32)
    # with open('refined_data/data.txt') as file:
    #     for line in file:
    #         line=line.strip().split('\t')
    #         id1=int(line[0])
    #         id2=int(line[1])
    #         aut_aut[id1,id2]=1
    #         aut_aut[id2,id1]=1
    for i in range(len(edge_index[0])):
        node_node[int(edge_index[0][i]), int(edge_index[1][i])] = 1
        node_node[int(edge_index[1][i]), int(edge_index[0][i])] = 1
    node_node=sp.csr_matrix(node_node)
    degrees=np.ravel(np.sum(node_node,axis=1))
    indptr = node_node.indptr
    indices = node_node.indices
    split_indices = np.split(indices, indptr[1:-1])
    efeature=np.zeros((node_num,node_num,6),dtype=np.float32)
    sumf=np.zeros(shape=(6,),dtype=np.float32)
    for i in range(node_num):
        fea = [degrees[i], degrees[i], common_neighbor(split_indices, i, i), \
                admic_adar(split_indices, i, i), jaccard(split_indices, i, i), degrees[i] * degrees[i]]
        # print(fea)
        efeature[i, i] = fea
        sumf += fea
        for j in split_indices[i]:
                fea=[degrees[i],degrees[j],common_neighbor(split_indices,i,j),\
                 admic_adar(split_indices,i,j),jaccard(split_indices,i,j),degrees[i]*degrees[j]]
                # print(fea)
                efeature[i,j]=fea
                sumf+=fea
    sumf=np.reciprocal(sumf)
    efeature=efeature*sumf
    # efeature = np.mean(efeature, axis=2)
    # efeature = efeature[:,:,5]
    if args.edge_featurte == 'degree_i':
        efeature = efeature[:,:,0]
    if args.edge_featurte == 'degree_j':
        efeature = efeature[:,:,1]
    if args.edge_featurte == 'common_neighbor':
        efeature = efeature[:,:,2]
    if args.edge_featurte == 'admic_adar':
        efeature = efeature[:,:,3]
    if args.edge_featurte == 'jaccard':
        efeature = efeature[:,:,4]
    if args.edge_featurte == 'PA':
        efeature = efeature[:,:,5]
    if args.edge_featurte == 'mean':
        efeature = np.mean(efeature, axis=2)
    return efeature

def normalize(adj, mode, sparse=False): # normalize function
    if not sparse: # If it is not a sparse graph
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else: # If it is a sparse graph
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()) # Reconstructing sparse graph


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    # mask = torch.zeros(raw_graph.shape).cuda()
    mask = torch.zeros(raw_graph.shape).to(device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    # values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    # rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    # cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    # norm_row = torch.zeros(X.shape[0]).cuda()
    # norm_col = torch.zeros(X.shape[0]).cuda()
    values = torch.zeros(X.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(device)
    norm_row = torch.zeros(X.shape[0]).to(device)
    norm_col = torch.zeros(X.shape[0]).to(device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.to_sparse()
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda:0')
    # dgl_graph.edata['w'] = values.detach().cuda()
    # dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cpu')
    dgl_graph.edata['w'] = values.detach().to(device)

    return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).to(device)
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore