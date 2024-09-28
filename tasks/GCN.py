import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric import seed_everything
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
elif args.dataset == "napus":
    args.predictor = "1"  # napus
elif args.dataset == "cotton_80":
    args.predictor = "2"  # cotton_80
elif args.dataset == "wheat":
    args.predictor = "2"  # wheat

class MyGCN(nn.Module):

    def __init__(self, args, layer_num = 2) -> None:

        super().__init__()

        self.in_channels = args.feature_dim
        self.out_channels = args.feature_dim
        self.layer_num = layer_num

        self.activation = nn.ReLU()

        # predictor
        self.predictor = Predictor(args)

    def forward(self, edge_index, x):

        for _ in range(self.layer_num):
            encoder = GCNConv(self.in_channels, self.out_channels).to(device)
            x = encoder(x, edge_index).to(device)
            x = self.activation(x).to(device)

        return x


    def predict(self, x, src_id, dst_id):

        src_x = x[src_id]
        dst_x = x[dst_id]

        score = self.predictor(src_x, dst_x)

        return F.softmax(score, dim=1)


    def loss(self, score, y):
        """label loss"""
        return F.cross_entropy(score, y)


    @torch.no_grad()
    def test(self, pred_y, y):
        """test method, return acc auc f1"""
        pred = pred_y.cpu().numpy()
        test_y = y.cpu().numpy()

        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        micro_f1 = f1_score(test_y, pred, average="micro")
        macro_f1 = f1_score(test_y, pred, average="macro")

        return acc, auc, f1, micro_f1, macro_f1


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


def test(model, x, test_pos_edge_index, test_neg_edge_index):
    # test
    model.eval()

    with torch.no_grad():

        # test predict
        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        prob = model.predict(x, test_src_id, test_dst_id).to(device)
        score_test = prob[:, (0, 2)].max(dim=1)[1]

        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

    return acc, auc, f1, micro_f1, macro_f1


seed_list = [1482, 1111, 490, 510, 197]

if not os.path.exists(f"./results/{args.dataset}/GCN"):
    os.makedirs(f"./results/{args.dataset}/GCN")

# load period data
period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)

for period_name in period:

    args.period = period_name

    res = []

    for times in range(5):

        args.seed = seed_list[times]

        torch.random.manual_seed(args.seed)
        seed_everything(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        dataloader = DataLoad(args)
        train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()
        train_edge_index = torch.concat((train_pos_edge_index, train_neg_edge_index), dim=1).to(device)

        node_nums = torch.max(train_pos_edge_index).item()
        x = dataloader.create_feature(node_nums)
        original_x = x.clone()

        # Build and train model
        if args.dataset == "cotton":
            args.feature_dim = 64  # cotton
        elif args.dataset == "napus":
            args.feature_dim = 32  # napus
        elif args.dataset == "cotton_80":
            args.feature_dim = 16  # cotton_80
        elif args.dataset == "wheat":
            args.feature_dim = 64  # wheat
        linear_DR = nn.Linear(x.shape[1], args.feature_dim).to(device)
        model = MyGCN(args, layer_num=2)
        optimizer = torch.optim.Adam(chain.from_iterable([model.parameters(), linear_DR.parameters()]), lr=0.005, weight_decay=5e-4)

        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1).to(args.device)

        best_acc, best_auc, best_f1, best_micro_f1, best_macro_f1 = 0, 0, 0, 0, 0
        best_model = None

        for epoch in range(800):

            x = linear_DR(original_x)

            # GCN embedding x
            x = model(train_pos_edge_index, x)

            # predict train score
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

            # label loss
            loss = model.loss(score, y_train)

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
        if args.dataset == "napus":
            pass
        elif args.dataset == "cotton":
            acc, auc, f1, micro_f1, macro_f1 = test(best_model, x, test_pos_edge_index, test_neg_edge_index)

        print(f"test acc: {acc:.4f}, auc: {auc:.4f}, f1: {f1:.4f}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}")

        res.append((acc, auc, f1, micro_f1, macro_f1))

    res = np.array(res)
    print(res.mean(axis=0))
    print(res.std(axis=0))
    print()

    with open(f"./results/{args.dataset}/GCN/{args.period}_res.txt", "w") as f:
        for line in res.tolist():
            f.writelines(str(line))
            f.writelines("\n")
        f.writelines("\n")
        f.write(f"acc: {res.mean(axis=0)[0]:.4f}±{res.std(axis=0)[0]:.4f}\n")
        f.write(f"auc: {res.mean(axis=0)[1]:.4f}±{res.std(axis=0)[1]:.4f}\n")
        f.write(f"f1: {res.mean(axis=0)[2]:.4f}±{res.std(axis=0)[2]:.4f}\n")
        f.write(f"micro_f1: {res.mean(axis=0)[3]:.4f}±{res.std(axis=0)[3]:.4f}\n")
        f.write(f"macro_f1: {res.mean(axis=0)[4]:.4f}±{res.std(axis=0)[4]:.4f}\n")
        f.write("\n")

    break
        
