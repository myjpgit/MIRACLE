import argparse
import copy

from data_loader import load_data
from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl
from torch_geometric.utils import train_test_split_edges
import random
from torch_geometric.utils import negative_sampling
from utils import get_link_labels, evaluate_AUC, evaluate_ks, link_prediction, generate_pos_edge_index, generate_neg_edge_index
from torch_geometric.transforms import RandomLinkSplit

#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_cls(self, model, mask, features, labels):
        # Evaluation Network loss (Classification)
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def loss_cls_lp(self, model, features, data, mode="train"):
        # Evaluation Network loss (Classification)
        logits = model(features)
        if mode == "train":
            logits = link_prediction(data, logits, mode="train")
            loss = F.binary_cross_entropy_with_logits(logits, data.train_y)
            auc = evaluate_AUC(logits, data.train_y)
            ks = evaluate_ks(logits, data.train_y)
        if mode == "val":
            logits = link_prediction(data, logits, mode="val")
            loss = F.binary_cross_entropy_with_logits(logits, data.val_y)
            auc = evaluate_AUC(logits, data.val_y)
            ks = evaluate_ks(logits, data.val_y)
        if mode == "test":
            logits = link_prediction(data, logits, mode="test")
            loss = F.binary_cross_entropy_with_logits(logits, data.test_y)
            auc = evaluate_AUC(logits, data.test_y)
            ks = evaluate_ks(logits, data.test_y)
        return loss, auc, ks


    def loss_gcl(self, model, graph_learner, features, anchor_adj, anchor_weight_adj):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor) # Data Augmentation
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        # print(anchor_adj)
        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask_v2, _ = get_feat_mask(features, args.maskfeat_rate_learner) # Data Augmentation
            features_v2 = features * (1 - mask_v2)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        # print(learned_adj)

        if not args.sparse:
            learned_adj = symmetrize(learned_adj) # Post-processor
            learned_adj = normalize(learned_adj, 'sym', args.sparse)  # Post-processor

        #print(learned_adj)
        z2, _ = model(features_v2, learned_adj, 'learner')

        # # view3: Edge weights adj graph
        # if args.maskfeat_rate_learner:
        #     mask_v3, _ = get_feat_mask(features, args.maskfeat_rate_learner) # Data Augmentation
        #     features_v3 = features * (1 - mask_v3)
        # else:
        #     features_v3 = copy.deepcopy(features)

        # edge_weight_adj = edge_learner(features)

        # if not args.sparse:
        #     edge_weight_adj = symmetrize(edge_weight_adj) # Post-processor
        #     edge_weight_adj = normalize(edge_weight_adj, 'sym', args.sparse)  # Post-processor

        # z3, _ = model(features_v3, edge_weight_adj, 'learner') #edge_weightadj 带有边权值的邻接矩阵视图

        # view 3: anchor_weight graph
        if args.maskfeat_rate_anchor:
            mask_v3, _ = get_feat_mask(features, args.maskfeat_rate_anchor)  # Data Augmentation
            features_v3 = features * (1 - mask_v3)
        else:
            features_v3 = copy.deepcopy(features)

        z3, _ = model(features_v3, anchor_weight_adj, 'anchor')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch], z3[batch]) * weight
        else:
            # loss = model.calc_loss(z1, z2) # Calculating contrastive loss
            loss = model.calc_loss(args, z1, z2, z3)  # Calculating contrastive loss

        # return loss, learned_adj # return the contrastive loss and learned adjacency matrix
        return loss, learned_adj, z2 # return the contrastive loss and learned adjacency matrix

    def evaluate_adj_by_cls(self, Adj, features, nfeats, nclasses, data, args):
        # Using GCN with label participation to evaluate the adjacency matrix learned by contrast learning

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_ks = 0
        best_model = None

        model = model.to(device)
        features = features.to(device)
        data = data.to(device)

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            #loss, accu = self.loss_cls(model, train_mask, features, labels)
            loss, accu, ks = self.loss_cls_lp(model, features, data, "train")
            optimizer.zero_grad()
            loss.backward() # Backward Propagation

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                # val_loss, accu = self.loss_cls(model, val_mask, features, labels)
                val_loss, accu, ks = self.loss_cls_lp(model, features, data, "val")
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_ks = ks
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        # test_loss, test_accu = self.loss_cls(best_model, test_mask, features, labels)
        test_loss, test_accu, test_ks = self.loss_cls_lp(best_model, features, data, "test")
        return best_val, best_ks, test_accu, test_ks, best_model


    def train(self, args):

        # torch.cuda.set_device(args.gpu)

        data, features, nfeats, labels, nclasses, adj_original, edge_index = load_data(args)  # 加入了原始图的边索引

        if args.dataset in ["cora", "citeseer", "pubmed"]:
            data = train_test_split_edges(data, val_ratio=0.6, test_ratio=0.3)  # 重要步骤，生成负样本，划分训练集等
            pos_train_edge_index = data.train_pos_edge_index
            neg_train_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1),
                force_undirected=True)

            # construct labels
            train_labels = get_link_labels(pos_train_edge_index, neg_train_edge_index)
            val_labels = get_link_labels(data.val_pos_edge_index, data.val_neg_edge_index)
            test_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)

            data.train_y = torch.tensor(train_labels, dtype=torch.float32).to(device)
            data.val_y = torch.tensor(val_labels, dtype=torch.float32).to(device)
            data.test_y = torch.tensor(test_labels, dtype=torch.float32).to(device)

            data.num_features = data.num_features
            data.num_labels = 2
            data.num_nodes = data.num_nodes

        elif args.dataset in ["samecity", "terror", "advisor"]:
            # transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.3)
            transform = RandomLinkSplit(is_undirected=True, num_val=0.6, num_test=0.3)
            train_data, val_data, test_data = transform(data)

            data.train_edge_index = train_data.edge_index
            data.val_edge_index = val_data.edge_index
            data.test_edge_index = test_data.edge_index

            train_labels = train_data.y.to(device)
            val_labels = val_data.y.to(device)
            test_labels = test_data.y.to(device)

            data.train_pos_edge_index = generate_pos_edge_index(data.train_edge_index, train_labels)
            pos_train_edge_index = data.train_pos_edge_index
            neg_train_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1),
                force_undirected=True)
            train_labels = get_link_labels(pos_train_edge_index, neg_train_edge_index)

            data.val_pos_edge_index = generate_pos_edge_index(data.val_edge_index, val_labels)
            data.val_neg_edge_index = negative_sampling(
                edge_index=data.val_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.val_pos_edge_index.size(1),
                force_undirected=True)

            data.test_pos_edge_index = generate_pos_edge_index(data.test_edge_index, test_labels)
            data.test_neg_edge_index = negative_sampling(
                edge_index=data.test_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.test_pos_edge_index.size(1),
                force_undirected=True)

            val_labels = get_link_labels(data.val_pos_edge_index, data.val_neg_edge_index)
            test_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)

            data.train_y = torch.tensor(train_labels, dtype=torch.float32).to(device)
            data.val_y = torch.tensor(val_labels, dtype=torch.float32).to(device)
            data.test_y = torch.tensor(test_labels, dtype=torch.float32).to(device)

            data.num_features = data.num_features
            data.num_labels = 2
            data.num_nodes = data.num_nodes

        test_accuracies = []
        validation_accuracies = []
        test_kss = []
        validation_kss = []

        for trial in range(args.ntrials):

            self.setup_seed(trial)

            if args.sparse:
                anchor_adj_raw = adj_original  # Structure refinement use adjacency matrices (adj_original not a sparse graph)
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)  # Used to convert an array to a tensor
            if args.sparse:
                anchor_weight_adj_raw = adj_original * edge_feature(args, data.num_nodes, edge_index)  # Structure refinement use adjacency matrices (adj_original not a sparse graph)
            else:
                anchor_weight_adj_raw = torch.from_numpy(adj_original * edge_feature(args, data.num_nodes, edge_index))  # Used to convert an array to a tensor

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse) # normalize anchor_adj
            orignal_adj = normalize(anchor_adj_raw, 'sym', args.sparse) # normalize anchor_adj
            anchor_weight_adj = normalize(anchor_weight_adj_raw, 'sym', args.sparse)
            # orignal_weight_adj = normalize(anchor_weight_adj_raw, 'sym', args.sparse)
            # print(orignal_adj)

            if args.sparse: # If it is a sparse graph
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)  # Deep copy of anchor_adj
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)
                # orignal_adj_torch_sparse = copy.deepcopy(orignal_adj)
                # orignal_adj = torch_sparse_to_dgl_graph(orignal_adj)
                anchor_weight_adj_torch_sparse = copy.deepcopy(anchor_weight_adj)
                anchor_weight_adj = torch_sparse_to_dgl_graph(anchor_weight_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

            model = model.to(device)
            graph_learner = graph_learner.to(device)
            features = features.to(device)
            labels = labels.to(device)
            orignal_adj = orignal_adj.to(device)

            if not args.sparse:
                anchor_adj = anchor_adj.to(device)
                anchor_weight_adj = anchor_weight_adj.to(device)

            if args.downstream_task == 'link_prediction':
                best_val = 0
                best_val_test = 0
                best_ks = 0
                best_ks_test = 0
                best_epoch = 0

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                # loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, anchor_weight_adj)
                loss, Adj, new_features = self.loss_gcl(model, graph_learner, features, anchor_adj, anchor_weight_adj)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_weight_adj_torch_sparse = anchor_weight_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                        anchor_weight_adj = torch_sparse_to_dgl_graph(anchor_weight_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)
                        anchor_weight_adj = anchor_weight_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'link_prediction':
                        model.eval()
                        graph_learner.eval()
                        f_adj = Adj + orignal_adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, val_ks, test_accu, test_ks, _ = self.evaluate_adj_by_cls(f_adj, features, nfeats, nclasses, data, args)

                        if val_accu > best_val:
                            best_val = val_accu
                            best_ks = val_ks
                            best_val_test = test_accu
                            best_ks_test = test_ks
                            best_epoch = epoch

            if args.downstream_task == 'link_prediction':
                validation_accuracies.append(best_val.item())
                validation_kss.append(best_ks.item())
                test_accuracies.append(best_val_test.item())
                test_kss.append(best_ks_test.item())
                print("Trial: ", trial + 1)
                print("Best val AUC: ", best_val.item())
                print("Best test AUC: ", best_val_test.item())
                print("Best val KS: ", best_ks.item())
                print("Best test KS: ", best_ks_test.item())

        if args.downstream_task == 'link_prediction' and trial != 0:
            self.print_results(validation_accuracies, validation_kss, test_accuracies, test_kss)


    def print_results(self, validation_accu, validation_ks, test_accu, test_ks):
        s_val = "Val AUC: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test AUC: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        ks_val = "Val KS: {:.4f} +/- {:.4f}".format(np.mean(validation_ks), np.std(validation_ks))
        ks_test = "Test KS: {:.4f} +/- {:.4f}".format(np.mean(test_ks),np.std(test_ks))
        print(s_val)
        print(s_test)
        print(ks_val)
        print(ks_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='samecity',
                        choices=['cora', 'citeseer', 'pubmed', 'samecity', 'terror', 'advisor'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='link_prediction',
                        choices=['link_prediction', 'node_classification'])
    parser.add_argument('-gpu', type=int, default=-1)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "mlp"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('-loss_alpha', type=float, default=0.5)
    parser.add_argument('-edge_featurte', type=str, default='mean', choices=["degree_i", "degree_j", "common_neighbor", "admic_adar", "jaccard", "PA", "mean"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
