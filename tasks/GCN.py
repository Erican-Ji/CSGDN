import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric import seed_everything
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


class MyGCN(nn.Module):

    def __init__(self, x, in_channels = 32, out_channels = 32, layer_num = 2) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_num = layer_num

        self.activation = nn.ReLU()

        # predictor
        self.predictor = Predictor(out_channels)

        # dimension reduce embeddings
        self.linear_DR = nn.Linear(x.shape[1], in_channels).to(device)
        self.x = x


    def dimension_reduction(self):
        return self.linear_DR(self.x)


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

        return score


    def loss(self, score, y):
        """label loss"""
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
    
    def __init__(self, in_channels):
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


percent_list = [30, 50, 60, 70, 80, 100]
# percent_list = [100]
seed_list = [114, 514, 1919, 810, 721]

for percent in percent_list:

    print(f"{percent} Start!")

    res = []

    for times in range(5):

        seed = seed_list[times]

        torch.random.manual_seed(seed)
        seed_everything(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataloader = DataLoad(percent, times+1)
        train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()
        train_edge_index = torch.concat((train_pos_edge_index, train_neg_edge_index), dim=1).to(device)

        node_nums = torch.max(train_pos_edge_index).item()
        x = dataloader.create_feature(node_nums)

        model = MyGCN(x, in_channels=32, out_channels=32, layer_num=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_auc = 0
        best_acc = 0
        best_f1 = 0

        for epoch in range(400):

            x = model.dimension_reduction()

            # GCN embedding x
            x_new = model(train_edge_index, x)

            # predict train score
            src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
            dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

            y_train = torch.concat((torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_neg_edge_index.shape[1]))).to(device)

            score = model.predict(x_new, src_id, dst_id)

            # label loss
            loss = model.loss(score, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        model.eval()

        with torch.no_grad():

            # test predict
            test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
            test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

            y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

            score_test = model.predict(x_new, test_src_id, test_dst_id).to(device)

            acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

            # print(f"epoch {epoch+1}: acc {acc}; auc {auc}; f1 {f1}")

        res.append((acc, auc, f1, micro_f1, macro_f1))
        print(res[times])

        # print(f"best epoch {best_epoch}: best acc {best_acc}, best auc {best_auc}, best f1 {best_f1}")
    res = np.array(res)
    print(res.mean(axis=0))
    print(res.std(axis=0))
    print()

