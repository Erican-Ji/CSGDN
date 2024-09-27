import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os
from itertools import chain
import argparse

import sys
sys.path.append("..")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus", "cotton_80"], 
                    help='choose dataset')

args = parser.parse_args()

from CSGDN.utils import DataLoad

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

args.device = device
if args.dataset == "cotton":
    args.predictor = "2"  # cotton
    beta = 1e-2
    args.mask_ratio = 0.4
elif args.dataset == "napus":
    args.predictor = "1"  # napus
    beta = 0.1
    args.mask_ratio = 0.8
elif args.dataset == "cotton_80":
    args.predictor = "2"  # cotton
    beta = 1e-2
    args.mask_ratio = 0.4

seed_list = [1482, 1111, 490, 510, 197]

class Predictor(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.predictor == "1":
            # 1-Linear MLP
            self.predictor = nn.Linear(self.args.feature_dim * 2, 3).to(self.args.device)
        elif args.predictor == "2":
            # 2-Linear MLP
            self.predictor = nn.Sequential(nn.Linear(self.args.feature_dim * 2, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, 3)).to(self.args.device)
        elif args.predictor == "3":
            self.predictor = nn.Sequential(nn.Linear(self.args.feature_dim * 2, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, 3)).to(self.args.device)
        elif args.predictor == "4":
            self.predictor = nn.Sequential(nn.Linear(self.args.feature_dim * 2, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, self.args.feature_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(self.args.feature_dim, 3)).to(self.args.device)

    def forward(self, ux, vx):
        """link (u, v)"""

        x = torch.concat((ux, vx), dim=-1)
        # res = self.predictor(x).flatten()
        res = self.predictor(x)

        return res


class MyGRACE(nn.Module):

    def __init__(self, args, layer_num = 2, tau = 0.05) -> None:
        super().__init__()

        self.in_channels = args.feature_dim
        self.out_channels = args.feature_dim
        self.layer_num = layer_num
        self.tau = tau

        # transforms
        self.transforms = nn.Linear(2 * self.out_channels, self.out_channels).to(device)

        # activation
        self.activation = nn.ReLU()

        # predict
        self.predictor = Predictor(args)

    def encode(self, edge_index_a, edge_index_b, x):

        x_a, x_b = None, None

        for _ in range(self.layer_num):

            encoder = GCNConv(self.in_channels, self.out_channels).to(device)

            x_a = encoder(x, edge_index_a).to(device)
            x_a = self.activation(x_a).to(device)

            x_b = encoder(x, edge_index_b).to(device)
            x_b = self.activation(x_b).to(device)

        return x_a, x_b
        
    def forward(self, edge_index, x):

        view_a_pos, view_a_neg, view_b_pos, view_b_neg = edge_index

        pos_x_a, pos_x_b = self.encode(view_a_pos, view_b_pos, x)
        """
        neg_x_a, neg_x_b = self.encode(view_a_neg, view_b_neg, x)

        x_a = torch.concat((pos_x_a, neg_x_a), dim=1)
        x_a = self.activation(self.transforms(x_a))

        x_b = torch.concat((pos_x_b, neg_x_b), dim=1)
        x_b = self.activation(self.transforms(x_b))

        return x_a, x_b
        """
        return pos_x_a, pos_x_b

    def split_pos_neg(self, edge_index, edge_value):

        pos_edge_index = edge_index[:, edge_value > 0]
        neg_edge_index = edge_index[:, edge_value < 0]

        return pos_edge_index, neg_edge_index

    def remove_edges(self, edge_index, edge_value, mask_ratio = 0.1):

        mask = torch.rand(size=(1, edge_index.shape[1]))  # uniform distribution
        mask[mask < mask_ratio] = 0
        mask[mask >= mask_ratio] = 1

        mask = mask[0].bool().to(device)
        edge_value = edge_value.reshape(1, -1).to(device)

        return edge_index[:, mask], edge_value[:, mask][0]

    def generate_view(self, train_pos_edge_index, train_neg_edge_index):

        pos_num = train_pos_edge_index.shape[1]
        neg_num = train_neg_edge_index.shape[1]

        return self.split_pos_neg(*self.remove_edges(torch.concat((train_pos_edge_index, train_neg_edge_index), dim=1), torch.concat((torch.ones(pos_num), torch.zeros(neg_num))), mask_ratio = args.mask_ratio))

    def sim(self, x_a: torch.Tensor, x_b: torch.Tensor):
        x_a = F.normalize(x_a)
        x_b = F.normalize(x_b)
        return torch.mm(x_a, x_b.t())

    def semi_loss(self, x_a: torch.Tensor, x_b: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(x_a, x_a))
        between_sim = f(self.sim(x_a, x_b))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def label_loss(self, score, y):
        return F.cross_entropy(score, y)

    def constrative_loss(self, x_a, x_b, mean=True):
        l1 = self.semi_loss(x_a, x_b)
        l2 = self.semi_loss(x_b, x_a)
        return ((l1 + l2) * 0.5).mean() if mean else ((l1 + l2) * 0.5).sum()

    def predict(self, x_concat, src_id, dst_id):
        x_concat = self.transforms(x_concat)
        x_concat = F.normalize(x_concat)

        src_x = x_concat[src_id]
        dst_x = x_concat[dst_id]

        score = self.predictor(src_x, dst_x)

        return F.softmax(score, dim=1)

    @torch.no_grad()
    def test(self, pred_y, y):
        pred = pred_y.cpu().numpy()
        test_y = y.cpu().numpy()

        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        micro_f1 = f1_score(test_y, pred, average="micro")
        macro_f1 = f1_score(test_y, pred, average="macro")

        return acc, auc, f1, micro_f1, macro_f1


def test(model, x, test_pos_edge_index, test_neg_edge_index):

    model.eval()

    with torch.no_grad():

        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        prob = model.predict(x, test_src_id, test_dst_id).to(device)
        score_test = prob[:, (0, 2)].max(dim=1)[1]

        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

    return acc, auc, f1, micro_f1, macro_f1

def train():

    # train
    dataloader = DataLoad(args)
    train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()

    node_num = torch.max(train_pos_edge_index).item()
    x = dataloader.create_feature(node_num)
    original_x = x.clone()

    if args.dataset == "cotton":
        args.feature_dim = 64  # cotton
    elif args.dataset == "napus":
        args.feature_dim = 32  # napus
    linear_DR = nn.Linear(x.shape[1], args.feature_dim).to(device)
    model = MyGRACE(args)
    optimizer = torch.optim.Adam(chain.from_iterable([model.parameters(), linear_DR.parameters()]), lr=0.005, weight_decay=5e-4)

    # generate view
    view_a_pos, view_a_neg = model.generate_view(train_pos_edge_index, train_neg_edge_index)
    view_b_pos, view_b_neg = model.generate_view(train_pos_edge_index, train_neg_edge_index)

    edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1).to(args.device)

    best_acc, best_auc, best_f1, best_micro_f1, best_macro_f1 = 0, 0, 0, 0, 0
    best_model = None

    for epoch in range(400):

        x = linear_DR(original_x)

        x_a, x_b = model((view_a_pos, view_a_neg, view_b_pos, view_b_neg), x)

        x = torch.concat((x_a, x_b), dim=1)

        # predict
        none_edge_index = negative_sampling(edge_index, x.size(0))
        src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0], none_edge_index[0])).to(device)
        dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1], none_edge_index[1])).to(device)

        # pos: 001; neg: 100; none: 010
        pos_y = torch.zeros(train_pos_edge_index.shape[1], 3).to(device)
        pos_y[:, 2] = 1
        neg_y = torch.zeros(train_neg_edge_index.shape[1], 3).to(device)
        neg_y[:, 0] = 1
        none_y = torch.zeros(none_edge_index.shape[1], 3).to(device)
        none_y[:, 1] = 1
        y_train = torch.concat((pos_y, neg_y, none_y))

        score = model.predict(x, src_id, dst_id)

        con_loss = model.constrative_loss(x_a, x_b)
        label_loss = model.label_loss(score, y_train)
        loss = beta * con_loss + (1 - beta) * label_loss
        # print(f"\rloss{epoch} {loss.item()}", end="", flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, auc, f1, micro_f1, macro_f1 = test(model, x, val_pos_edge_index, val_neg_edge_index)

        if best_auc + best_f1 < auc + f1:
            best_acc, best_auc, best_f1, best_micro_f1, best_macro_f1 = acc, auc, f1, micro_f1, macro_f1
            best_model = model

        print(f"\rEpoch: {epoch+1:03d}, Loss: {loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}", end="", flush=True)

    print(f"\nbest val acc: {best_acc:.4f}, auc: {best_auc:.4f}, f1: {best_f1:.4f}, micro_f1: {best_micro_f1:.4f}, macro_f1: {best_macro_f1:.4f}")

    # test
    acc, auc, f1, micro_f1, macro_f1 = test(best_model, x, test_pos_edge_index, test_neg_edge_index)

    print(f"test acc: {acc:.4f}, auc: {auc:.4f}, f1: {f1:.4f}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}")

    return acc, auc, f1, micro_f1, macro_f1


res_str = []

if __name__ == "__main__":

    if not os.path.exists(f"./results/{args.dataset}/GRACE"):
        os.makedirs(f"./results/{args.dataset}/GRACE")

    # load period data
    period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)

    for period_name in period:

        args.period = period_name

        res = []

        for times in range(5):

            args.seed = seed_list[times]

            torch.random.manual_seed(args.seed)
            torch_geometric.seed_everything(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)

            # train
            acc, auc, f1, micro_f1, macro_f1 = train()

            res.append([acc, auc, f1, micro_f1, macro_f1])

        # calculate the avg of each times
        res = np.array(res)
        avg = res.mean(axis=0)
        std = res.std(axis=0)
        print(avg)
        print(std)
        print()

        with open(f"./results/{args.dataset}/GRACE/{period_name}_res.txt", "w") as f:
            for line in res.tolist():
                f.writelines(str(line))
                f.writelines("\n")
            f.writelines("\n")
            f.write(f"acc: {avg[0]:.4f}±{std[0]:.4f}\n")
            f.write(f"auc: {avg[1]:.4f}±{std[1]:.4f}\n")
            f.write(f"f1: {avg[2]:.4f}±{std[2]:.4f}\n")
            f.write(f"micro_f1: {avg[3]:.4f}±{std[3]:.4f}\n")
            f.write(f"macro_f1: {avg[4]:.4f}±{std[4]:.4f}\n")
            f.write("\n")
    
        break
