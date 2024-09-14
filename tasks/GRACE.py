import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import argparse

import sys
sys.path.append("..")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus"], 
                    help='choose dataset')

args = parser.parse_args()

if args.dataset == "cotton":
    from ShareMethod import DataLoad
elif args.dataset == "wheat":
    from wheat_dataloader import DataLoad
elif args.dataset == "napus":
    from napus_dataloader import DataLoad

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


percent_list = [30, 50, 60, 70, 80, 100]
seed_list = [114, 514, 1919, 810, 721]
lr = 0.01
beta = 5e-4


class Predictor(nn.Module):
    
    def __init__(self, in_channels = 64):
        super().__init__()

        # TODO add another methods ...

        # 2-Linear MLP
        self.predictor = nn.Sequential(nn.Linear(in_channels * 2, in_channels), 
                                       nn.ReLU(), 
                                       nn.Linear(in_channels, 1)).to(device)


    def forward(self, ux, vx):
        """link (u, v)"""

        x = torch.concat((ux, vx), dim=-1)
        res = self.predictor(x).flatten()

        return res

class MyGRACE(nn.Module):

    def __init__(self, x, in_channels = 32, out_channels = 32, layer_num = 2, tau = 0.5) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = layer_num
        self.tau = tau

        # transforms
        self.transforms = nn.Linear(2 * out_channels, out_channels).to(device)

        # activation
        self.activation = nn.ReLU()

        # predict
        self.predictor = Predictor(2 * out_channels)

        # dimention reduce embedding
        self.x = x
        self.linear_DR = nn.Linear(self.x.shape[1], in_channels).to(device)


    def dimension_reduction(self):
        """DR the init feature to the target dimensions (self.in_channels)"""
        return self.linear_DR(self.x)

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
        neg_x_a, neg_x_b = self.encode(view_a_neg, view_b_neg, x)

        x_a = torch.concat((pos_x_a, neg_x_a), dim=1)
        x_a = self.activation(self.transforms(x_a))

        x_b = torch.concat((pos_x_b, neg_x_b), dim=1)
        x_b = self.activation(self.transforms(x_b))

        return x_a, x_b

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

        return self.split_pos_neg(*self.remove_edges(torch.concat((train_pos_edge_index, train_neg_edge_index), dim=1), torch.concat((torch.ones(pos_num), torch.zeros(neg_num)))))

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
        pos_weight = torch.tensor([(y == 0).sum().item() / (y == 1).sum().item()] * y.shape[0]).to(device)
        return F.binary_cross_entropy_with_logits(score, y, pos_weight=pos_weight)

    def constrative_loss(self, x_a, x_b, mean=True):
        l1 = self.semi_loss(x_a, x_b)
        l2 = self.semi_loss(x_b, x_a)
        return ((l1 + l2) * 0.5).mean() if mean else ((l1 + l2) * 0.5).sum()

    def predict(self, x_concat, src_id, dst_id):
        src_x = x_concat[src_id]
        dst_x = x_concat[dst_id]

        return self.predictor(src_x, dst_x)

    @torch.no_grad()
    def test(self, pred_y, y):
        pred = pred_y.cpu().numpy()
        test_y = y.cpu().numpy()

        pred[pred >= 0] = 1
        pred[pred < 0] = 0

        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        micro_f1 = f1_score(test_y, pred, average="micro")
        macro_f1 = f1_score(test_y, pred, average="macro")

        return acc, auc, f1, micro_f1, macro_f1


def test(model, x_concat, test_pos_edge_index, test_neg_edge_index):

    model.eval()
    with torch.no_grad():

        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        score_test = model.predict(x_concat, test_src_id, test_dst_id).to(device)
        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)
        # print()
        print(f"\nacc {acc:.6f}; auc {auc:.6f}; f1 {f1:.6f}; micro_f1 {micro_f1:.6f}; macro_f1 {macro_f1:.6f}")

        return acc, auc, f1, micro_f1, macro_f1

def napus_test(grace_model, original_x, final_x, train_src_id, train_dst_id, test_pos_edge_index, test_neg_edge_index):
    grace_model.eval()

    # 合并 train_src_id, train_dst_id 并且去重
    train_id = torch.concat((train_src_id, train_dst_id)).unique()
    train_original_x = original_x[train_id]
    train_final_x = final_x[train_id]

    # 通过 oringal_x 学习一个多层感知机映射到 final_x
    model = nn.Sequential(nn.Linear(original_x.shape[1], final_x.shape[1]), 
                          nn.ReLU(),
                          nn.Linear(final_x.shape[1], final_x.shape[1])).to(device)

    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(400):
        model.train()
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
    test_original_x = original_x[test_id]
    test_final_x = model(test_original_x)
    final_x[test_id] = test_final_x

    with torch.no_grad():

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        # score_test = model.predict(model.x, test_src_id, test_dst_id).to(device)
        score_test = grace_model.predict(final_x, test_src_id, test_dst_id).to(device)

        acc, auc, f1, micro_f1, macro_f1 = grace_model.test(score_test, y_test)

        return acc, auc, f1, micro_f1, macro_f1

        # print(f"\nacc {acc:.6f}; auc {auc:.6f}; f1 {f1:.6f}; micro_f1 {micro_f1:.6f}; macro_f1 {macro_f1:.6f}")

    return acc, auc, f1, micro_f1, macro_f1

def train():

    # train
    train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index = DataLoad(percent, times+1).load_data_format()
    node_num = torch.max(train_pos_edge_index).item()
    x = DataLoad(percent, times+1).create_feature(node_num)
    original_x = x.clone()

    model = MyGRACE(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # generate view
    view_a_pos, view_a_neg = model.generate_view(train_pos_edge_index, train_neg_edge_index)
    view_b_pos, view_b_neg = model.generate_view(train_pos_edge_index, train_neg_edge_index)

    x_concat = None

    for epoch in range(400):

        x = model.dimension_reduction()

        x_a, x_b = model((view_a_pos, view_a_neg, view_b_pos, view_b_neg), x)

        x_concat = torch.concat((x_a, x_b), dim=1)

        # predict
        src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
        dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

        y_train = torch.concat((torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_neg_edge_index.shape[1]))).to(device)

        score = model.predict(x_concat, src_id, dst_id)

        con_loss = model.constrative_loss(x_a, x_b)
        label_loss = model.label_loss(score, y_train)
        loss = beta * con_loss + (1 - beta) * label_loss
        # print(f"\rloss{epoch} {loss.item()}", end="", flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    if args.dataset == "napus":
        acc, auc, f1, micro_f1, macro_f1 = napus_test(model, original_x, x_concat, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index)
    else:
        acc, auc, f1, micro_f1, macro_f1 = test(model, x_concat, test_pos_edge_index, test_neg_edge_index)

    return acc, auc, f1, micro_f1, macro_f1


res_str = []

if __name__ == "__main__":
    for percent in percent_list:
        res = []
        for times in range(5):
            # seed
            seed = seed_list[times]
            torch.random.manual_seed(seed)
            torch_geometric.seed_everything(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # train
            acc, auc, f1, micro_f1, macro_f1 = train()

            res.append([acc, auc, f1, micro_f1, macro_f1])

        # calculate the avg of each times
        res = np.array(res)
        avg = res.mean(axis=0)
        std = res.std(axis=0)
        res_str.append(f"percent {percent}: acc {avg[0]:.3f}+{std[0]:.3f}; auc {avg[1]:.3f}+{std[1]:.3f}; f1 {avg[2]:.3f}+{std[2]:.3f}; micro_f1 {avg[3]:.3f}+{std[3]:.3f}; macro_f1 {avg[4]:.3f}+{std[4]:.3f}\n")

    for i in range(6):
        print(f"percent {percent_list[i]}: {res_str[i]}")

