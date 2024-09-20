import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.decomposition import PCA
import argparse

import sys
sys.path.append("..")

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--percent', type=str, default='50',
                    help='choose percent')
parser.add_argument('--times', type=int, default=1,
                    help='Random seed. ( seed = seed_list[args.times] )')
parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus"], 
                    help='choose dataset')

args = parser.parse_args()

from CSGDN.utils import DataLoad

args.device = device

pheo_index_start = 0

class MySGNNMD(nn.Module):

    def __init__(self, bio_dim = 4, topo_dim = 16, k = 2) -> None:
        super().__init__()

        self.topo_dim = topo_dim
        self.k = k

        # transform
        self.transform = nn.Linear(topo_dim, 1).to(device)

        # activation
        self.activation = nn.Tanh()

        # predictor
        self.predictor = Predictor(bio_dim, topo_dim, k)

    def topo_feature_generate(self, node_label, max_n = 6):
        return F.one_hot(node_label, max_n)

    def forward(self, pos_edge, neg_edge, gene_feat = None, pheo_feat = None, h=1):

        edge_embedding, y = self.subgraph_generate(pos_edge, neg_edge, gene_feat, pheo_feat, h)
        score = self.predictor(edge_embedding)

        # loss
        loss = self.compute_label_loss(score, y)
        return loss, score, y

    # 2. NL
    def node_labeling(self, pair, pos_edge, neg_edge, gene_feat = None, pheo_feat = None, state = 1, h = 1):
        """state: 1: pos, 0: neg"""
        node_label = []
        # h hop for gene
        gene, pheo = pair[0], pair[1]
        pos_h_hop_gene_index = torch.where(pos_edge[1] == pheo)[0].to(device)
        neg_h_hop_gene_index = torch.where(neg_edge[1] == pheo)[0].to(device)

        pos_gene_len = pos_h_hop_gene_index.shape[0] - state  # except gene itself
        neg_gene_len = neg_h_hop_gene_index.shape[0] + state - 1  # except gene itself
        gene_len = 1 + pos_gene_len + neg_gene_len

        # h_hop_gene_index = torch.concat((pos_h_hop_gene_index, neg_h_hop_gene_index)).to(device)
        gene_label = [0] + pos_gene_len * [6 * h - 4] + neg_gene_len * [6 * h - 2]

        # h hop for pheo
        pos_h_hop_pheo_index = torch.where(pos_edge[0] == gene)[0].to(device)
        neg_h_hop_pheo_index = torch.where(neg_edge[0] == gene)[0].to(device)

        pos_pheo_len = pos_h_hop_pheo_index.shape[0] - state  # except pheo itself
        neg_pheo_len = neg_h_hop_pheo_index.shape[0] + state - 1  # except pheo itself
        pheo_len = 1 + pos_pheo_len + neg_pheo_len

        # h_hop_pheo_index = torch.concat((pos_h_hop_pheo_index, neg_h_hop_pheo_index)).to(device)
        pheo_label = [1] + pos_pheo_len * [6 * h - 3] + neg_pheo_len * [6 * h - 1]

        node_label += ([0] + gene_label + pheo_label)
        topo_feat = self.topo_feature_generate(torch.tensor(node_label), max(node_label)+1).to(torch.float).to(device)

        # 3. GCN  (order: gene pos_gene neg_gene pheo pos_pheo neg_pheo)
        # pos_edge
        gene_pos_edge_index = torch.concat((torch.arange(2, pos_gene_len+2).reshape(1, -1), torch.ones(size=(1, pos_gene_len)) * (gene_len+1))).to(device)
        gene_neg_edge_index = torch.concat((torch.arange(pos_gene_len+2, pos_gene_len+neg_gene_len+2).reshape(1, -1), torch.ones(size=(1, neg_gene_len)) * (gene_len+1))).to(device)

        # neg_edge
        pheo_pos_edge_index = torch.concat((torch.ones(size=(1, pos_pheo_len)), torch.arange(gene_len+2, gene_len+pos_pheo_len+2).reshape(1, -1))).to(device)
        pheo_neg_edge_index = torch.concat((torch.ones(size=(1, neg_pheo_len)), torch.arange(gene_len+pos_pheo_len+2, gene_len+pos_pheo_len+neg_pheo_len+2).reshape(1, -1))).to(device)

        pos_edge_index = torch.concat((gene_pos_edge_index, pheo_pos_edge_index), dim=1).to(torch.int)
        neg_edge_index = torch.concat((gene_neg_edge_index, pheo_neg_edge_index), dim=1).to(torch.int)

        topo_feat = self.GCN_embedding(topo_feat, pos_edge_index, neg_edge_index, self.topo_dim)[1: ]

        # 5. Select 
        bio_feat = torch.concat((gene_feat[pos_edge[0, pos_h_hop_gene_index], :], gene_feat[neg_edge[0, neg_h_hop_gene_index], :], pheo_feat[pos_edge[1, pos_h_hop_pheo_index]-pheo_index_start, :], pheo_feat[neg_edge[1, neg_h_hop_pheo_index]-pheo_index_start, :]), dim=0)  # shape=(gene_num+pheo_num, bio_dim)

        # 6. Concatenate
        concat_feat = torch.concat((topo_feat, bio_feat), dim=1).to(device)

        # 7. SortPooling
        x = self.SortPooling(topo_feat, concat_feat).reshape(1, -1)

        return x, state

    # 1. ES
    def subgraph_generate(self, pos_edge, neg_edge, gene_feat = None, pheo_feat = None, h=1):
        # we set h to 1 directly as described in paper: SGNNMD
        edge_embedding = torch.tensor([]).to(device)

        # pos_edge subgraph
        for gene, pheo in zip(pos_edge[0], pos_edge[1]):
            x, _ = self.node_labeling((gene, pheo), pos_edge, neg_edge, gene_feat, pheo_feat, 1, h)
            edge_embedding = torch.concat((edge_embedding, x), dim=0).to(device)
            print(f"\rsubgraph {edge_embedding.shape} done!", end="", flush=True)

        # neg_edge subgraph
        for gene, pheo in zip(neg_edge[0], neg_edge[1]):
            x, _ = self.node_labeling((gene, pheo), pos_edge, neg_edge, gene_feat, pheo_feat, 0, h)
            edge_embedding = torch.concat((edge_embedding, x), dim=0).to(device)
            print(f"\rsubgraph {edge_embedding.shape} done!", end="", flush=True)

        y = torch.concat((torch.ones(pos_edge.shape[1]), torch.zeros(neg_edge.shape[1]))).to(device)

        return edge_embedding, y

    # 3. GCN
    def GCN_embedding(self, topo_feat, pos_edge_index, neg_edge_index, out_channel):

        pos_encoder_1 = GCNConv(topo_feat.shape[1], out_channel).to(device)
        pos_encoder_2 = GCNConv(out_channel, out_channel).to(device)

        neg_encoder_1 = GCNConv(topo_feat.shape[1], out_channel).to(device)
        neg_encoder_2 = GCNConv(out_channel, out_channel).to(device)

        # layer 1
        pos_topo_feat = self.activation(pos_encoder_1(topo_feat, pos_edge_index)).to(device)
        neg_topo_feat = self.activation(neg_encoder_1(topo_feat, neg_edge_index)).to(device)

        # layer 2
        pos_topo_feat = self.activation(pos_encoder_2(pos_topo_feat, pos_edge_index)).to(device)
        neg_topo_feat = self.activation(neg_encoder_2(neg_topo_feat, neg_edge_index)).to(device)

        return pos_topo_feat + neg_topo_feat

    # 7. SortPooling
    def SortPooling(self, topo_feat, concat_feat):

        k = self.k if self.k <= topo_feat.shape[0] else topo_feat.shape[0]  # force every concat_feat has k rows

        sort_value = self.transform(topo_feat).to(device)
        _, top_k_index = sort_value.reshape(1, -1).topk(k)  # remain top(max) k
        sort_feat = concat_feat.index_select(0, top_k_index[0])  # shape=(k, concat_dim)

        return sort_feat.flatten().to(device)

    def compute_label_loss(self, score, y):
        pos_weight = torch.tensor([(y == 0).sum().item() / (y == 1).sum().item()] * y.shape[0]).to(device)
        return F.binary_cross_entropy_with_logits(score, y, pos_weight=pos_weight)

    @torch.no_grad()
    def test(self, pred_y, y):
        """test method, return acc auc f1"""
        pred = pred_y.cpu().numpy()
        test_y = y.cpu().numpy()

        # thresholds
        pred[pred >= 0] = 1
        pred[pred < 0] = 0

        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        micro_f1 = f1_score(test_y, pred, average="micro")
        macro_f1 = f1_score(test_y, pred, average="macro")

        return acc, auc, f1, micro_f1, macro_f1




class Predictor(nn.Module):
    
    def __init__(self, bio_dim, topo_dim, k, hidden_channels = 8):
        super().__init__()

        # 2-Linear MLP
        self.predictor = nn.Sequential(nn.Linear(k * (bio_dim + topo_dim), hidden_channels), 
                                       nn.ReLU(), 
                                       nn.Linear(hidden_channels, 1)).to(device)

    def forward(self, x):
        res = self.predictor(x).flatten()
        return res


percent_list = [30, 50, 60, 70, 80, 100]
seed_list = [114, 514, 1919, 810, 721]

if args.dataset == "napus":
    bio_dim = 2
else:
    bio_dim = 4
topo_dim = 16
k = 25
lr = 0.05
res_str = []

if __name__ == "__main__":
    # for per in range(6):
    # start per for
    res = []

    # percent = percent_list[per]
    percent = args.percent
    print(percent)
    print()

    # for times in range(5):
    # start times for
    # seed
    times = args.times - 1
    seed = seed_list[times]

    torch.random.manual_seed(seed)
    torch_geometric.seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    # train & test dataset ( trp trn tep ten )
    train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index = DataLoad(percent, times+1).load_data_format()
    print(f"time {times+1} train total {train_pos_edge_index.shape[1] + train_neg_edge_index.shape[1]}; test total {test_pos_edge_index.shape[1] + test_neg_edge_index.shape[1]}")

    pheo_index_start = train_pos_edge_index[1].min().item()

    # feature x
    gene_feat = DataLoad(percent, times+1).create_feature(1)
    if args.dataset == "cotton":
        pheo_feat = torch.tensor(np.loadtxt(f"../../data/cotton-data/p-p.csv", delimiter=","))
    elif args.dataset == "wheat":
        pheo_feat = torch.tensor(np.loadtxt(f"../../data/wheat/p-p.csv", delimiter=","))
    elif args.dataset == "napus":
        pheo_feat = torch.tensor(np.loadtxt(f"../../data/Brassica_napus-data/p-p.csv", delimiter=","))

    # 4. PCA
    pca = PCA(bio_dim, random_state=seed)
    gene_feat = torch.tensor(pca.fit_transform(gene_feat.cpu())).to(torch.float).to(device)
    pheo_feat = torch.tensor(pca.fit_transform(pheo_feat.cpu())).to(torch.float).to(device)

    model = MySGNNMD(bio_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(200):

        loss, _, _ = model(train_pos_edge_index, train_neg_edge_index, gene_feat, pheo_feat)
        print(f"\repoch {epoch} loss {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():

        _, score_test, y_test = model(test_pos_edge_index, test_neg_edge_index, gene_feat, pheo_feat)
        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

        print(f"\ntimes {times+1}: acc {acc}, auc {auc}, f1 {f1}, micro_f1 {micro_f1}, macro_f1 {macro_f1}")

    res.append([acc, auc, f1, micro_f1, macro_f1])
    # end times for

"""
    # calculate the avg of each times
    res = np.array(res)
    avg = res.mean(axis=0)
    std = res.std(axis=0)
    res_str.append(f"\npercent {args.percent}: acc {avg[0]:.3f}+{std[0]:.3f}; auc {avg[1]:.3f}+{std[1]:.3f}; f1 {avg[2]:.3f}+{std[2]:.3f}; micro_f1 {avg[3]:.3f}+{std[3]:.3f}; macro_f1 {avg[4]:.3f}+{std[4]:.3f}\n")
    # end per for
"""

"""
    for i in range(6):
        print(res_str[i])
"""
