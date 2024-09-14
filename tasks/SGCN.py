from typing import Tuple
# import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np
from torch_geometric.nn import SignedGCN
from torch_geometric import seed_everything
import argparse
from itertools import chain
import os

import sys
sys.path.append("..")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus"], 
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

class MySignedGCN(SignedGCN):

    def __init__(self, x, in_channels: int, hidden_channels: int, num_layers: int, lamb: float = 5, bias: bool = True):
        super().__init__(in_channels, hidden_channels, num_layers, lamb, bias)

        # dimension reduce embeddings
        self.x = x
    
    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float, float]:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]

        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        micro_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        macro_f1 = f1_score(y, pred, average='macro') if pred.sum() > 0 else 0

        return acc, auc, f1, micro_f1, macro_f1


def train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    
    return model.test(z, test_pos_edge_index, test_neg_edge_index)


def napus_test(sgcn_model, dr_x, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index):
    sgcn_model.eval()

    with torch.no_grad():
        z = sgcn_model(dr_x, train_pos_edge_index , train_neg_edge_index)

    # 合并 train_src_id, train_dst_id 并且去重
    train_src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
    train_dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

    train_id = torch.concat((train_src_id, train_dst_id)).unique()
    train_original_x = x.detach()[train_id]
    train_final_x = z[train_id]

    # 通过 oringal_x 学习一个多层感知机映射到 final_x
    model = nn.Sequential(nn.Linear(x.shape[1], z.shape[1]), 
                          nn.ReLU(),
                          nn.Linear(z.shape[1], z.shape[1])).to(device)

    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()

    for epoch in range(400):
        optimizer.zero_grad()
        x_hat = model(train_original_x)
        loss = Loss(x_hat, train_final_x.detach())
        loss.backward()
        optimizer.step()
        print(f"\rmapping epoch {epoch+1} done", end="", flush=True)

    model.eval()

    # 将 test_pos_edge_index, test_neg_edge_index 中对应在 original 中的 test_original_x 映射到 final_x
    test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
    test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)
    test_id = torch.concat((test_src_id, test_dst_id)).unique()
    test_original_x = x[test_id]
    test_final_x = model(test_original_x)
    z[test_id] = test_final_x

    return sgcn_model.test(z, test_pos_edge_index, test_neg_edge_index)

""" Picture
plt.xlabel('epoch')
plt.ylabel('loss')
epochs = []
tmp = []
for epoch in range(101):
    loss = train()
    auc, f1, acc = test()
    epochs.append(epoch)
    tmp.append([loss, auc, f1, acc])
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}')

res = np.array(tmp).T
plt.plot(epochs, res[0, :], color=(57/255, 197/255, 187/255), label='loss')
# plt.plot(epochs, res[1, :], color=(255/255, 165/255, 0/255), label='auc')
# plt.plot(epochs, res[2, :], color=(153/255, 211/255, 0/255), label='f1')
plt.plot(epochs, res[3, :], color=(255/255, 192/255, 203/255), label='acc')
plt.grid()
plt.legend()
plt.show()
"""

# percent_list = [30, 50, 60, 70, 80, 100]
percent_list = [60]
seed_list = [1145]

if not os.path.exists(f"./results/{args.dataset}/SGCN"):
    os.makedirs(f"./results/{args.dataset}/SGCN")

# load period data
period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)

# for percent in percent_list:
for period_name in period:

    args.period = period_name

    # print(f"{percent} Start!")

    res = []

    for times in range(5):

        seed = seed_list[0]

        torch.random.manual_seed(seed)
        seed_everything(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataloader = DataLoad(args)
        train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()
        # x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)
        node_nums = torch.max(train_pos_edge_index).item()
        x = dataloader.create_feature(node_nums)
        original_x = x.clone()

        # Build and train model
        linear_DR = nn.Linear(x.shape[1], 32).to(device)
        model = MySignedGCN(x, 32, 32, num_layers=2, lamb=5).to(device)
        optimizer = torch.optim.Adam(chain.from_iterable([model.parameters(), linear_DR.parameters()]), lr=0.01, weight_decay=5e-4)

        best_acc, best_auc, best_f1, best_micro_f1, best_macro_f1 = 0, 0, 0, 0, 0
        best_model = None

        for epoch in range(200):
            x = linear_DR(original_x)
            loss = train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            acc, auc, f1, micro_f1, macro_f1 = test(model, x, train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index)
            if best_auc + best_f1 < auc + f1:
                best_acc, best_auc, best_f1, best_micro_f1, best_macro_f1 = acc, auc, f1, micro_f1, macro_f1
                best_model = model

            print(f"\rEpoch: {epoch+1:03d}, Loss: {loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}", end="", flush=True)

        print(f"\nbest val acc: {best_acc:.4f}, auc: {best_auc:.4f}, f1: {best_f1:.4f}, micro_f1: {best_micro_f1:.4f}, macro_f1: {best_macro_f1:.4f}")

        if args.dataset == "napus":
            acc, auc, f1, micro_f1, macro_f1 = napus_test(model, x, original_x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index)
        else:
            acc, auc, f1, micro_f1, macro_f1 = test(best_model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index)

        print(f"test acc: {acc:.4f}, auc: {auc:.4f}, f1: {f1:.4f}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}")
        print()

        res.append((acc, auc, f1, micro_f1, macro_f1))
        # print(res[times])

    res = np.array(res)
    print(res.mean(axis=0))
    print(res.std(axis=0))
    print()

    with open(f"./results/{args.dataset}/SGCN/{args.period}_res.txt", "w") as f:
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
