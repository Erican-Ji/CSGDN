import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.utils import negative_sampling
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

class MySGCL(nn.Module):
    def __init__(self, args, in_channels=32, out_channels=32, layer_num=2) -> None:
        super().__init__()
        self.layer_num = layer_num
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.args = args

        # transform
        self.transform = nn.Linear(4 * out_channels, out_channels)

        # predictor
        self.predictor = Predictor(out_channels).to(device)

        self.activation = nn.ReLU()

        # dimension reduce embeddings
        self.linear_DR = nn.Linear(args.x.shape[1], out_channels).to(device)

    def dimension_reduction(self):
        return self.linear_DR(self.args.x)

    def drop_edges(self, edge_index, ratio=0.8):
        assert(0 <= ratio and ratio <= 1)
        M = edge_index.size(1)
        tM = int(M * ratio)
        permutation = torch.randperm(M)
        return edge_index[:, permutation[:tM]], edge_index[:, permutation[tM:]]

    def connectivity_perturbation(self, N, pos_edge_index, neg_edge_index, ratio=0.1):
        pos_tM = int(pos_edge_index.size(1) * ratio)
        res_pos_edge_index, _ = self.drop_edges(pos_edge_index, 1-ratio)
        neg_tM = int(pos_edge_index.size(1) * ratio)
        res_neg_edge_index, _ = self.drop_edges(neg_edge_index, 1-ratio)

        res_edge_index = torch.cat((res_pos_edge_index, res_neg_edge_index), dim=1)
        sample = negative_sampling(res_edge_index, N, pos_tM + neg_tM)
        pos_edge_index = torch.cat((res_pos_edge_index, sample[:, :pos_tM]), dim=1)
        neg_edge_index = torch.cat((res_neg_edge_index, sample[:, pos_tM:]), dim=1)
        return pos_edge_index, neg_edge_index
    
    def sign_perturbation(self, N, pos_edge_index, neg_edge_index, ratio=0.1):
        pos_edge_index, to_neg_edge_index = self.drop_edges(pos_edge_index, 1-ratio)
        neg_edge_index, to_pos_edge_index = self.drop_edges(neg_edge_index, 1-ratio)

        pos_edge_index = torch.cat((pos_edge_index, to_pos_edge_index), dim=1)
        neg_edge_index = torch.cat((neg_edge_index, to_neg_edge_index), dim=1)
        return pos_edge_index, neg_edge_index

    def generate_view(self, N, pos_edge_index, neg_edge_index):
        con_pos_edge_index, con_neg_edge_index = self.connectivity_perturbation(N, pos_edge_index, neg_edge_index, self.args.aug_ratio)
        sig_pos_edge_index, sig_neg_edge_index = self.sign_perturbation(N, pos_edge_index, neg_edge_index, self.args.aug_ratio)
        return con_pos_edge_index, con_neg_edge_index, sig_pos_edge_index, sig_neg_edge_index

    def encode(self, edge_index_a, edge_index_b, x):

        x_a, x_b = None, None

        for _ in range(self.layer_num):

            # encoder = GATConv(self.in_channels, self.out_channels).to(device)
            encoder = GCNConv(self.in_channels, self.out_channels).to(device)

            x_a = encoder(x, edge_index_a).to(device)
            x_a = self.activation(x_a).to(device)

            x_b = encoder(x, edge_index_b).to(device)
            x_b = self.activation(x_b).to(device)

        return x_a, x_b

    def forward(self, x, N, pos_edge_index, neg_edge_index):
        con_pos_edge_index, con_neg_edge_index, sig_pos_edge_index, sig_neg_edge_index = self.generate_view(N, pos_edge_index, neg_edge_index)

        pos_x_con, pos_x_sig = self.encode(con_pos_edge_index, sig_pos_edge_index, x)
        neg_x_con, neg_x_sig = self.encode(con_neg_edge_index, sig_neg_edge_index, x)

        x_concat = torch.concat((pos_x_con, pos_x_sig, neg_x_con, neg_x_sig), dim=1)
        return x_concat, pos_x_con, pos_x_sig, neg_x_con, neg_x_sig

    def similarity_score(self, x_a, x_b):
        """compute the similarity score : exp(\frac{sim_{imim'}}{\tau})"""

        sim_score = torch.bmm(x_a.view(x_a.shape[0], 1, x_a.shape[1]),
                x_b.view(x_b.shape[0], x_b.shape[1], 1))

        return torch.exp(torch.div(sim_score, self.args.tau))


    def compute_per_loss(self, x_a, x_b):
        """inter-contrastive"""

        numerator = self.similarity_score(x_a, x_b)  # exp(\frac{sim_{imim'}}{\tau})

        denominator = torch.mm(x_a.view(x_a.shape[0], x_a.shape[1]), x_b.transpose(0, 1))  # similarity value for (im, jm')
    
        denominator[np.arange(x_a.shape[0]), np.arange(x_a.shape[0])] = 0  # (im, im') = 0

        denominator = torch.sum(torch.exp(torch.div(denominator, self.args.tau)), dim=1)  # \sum_j exp(\frac{sim_{imjm'}}{\tau})

        # -\frac{1}{I} \sum_i log(\frac{numerator}{denominator})
        return torch.mean(-torch.log(torch.div(numerator, denominator)))


    def compute_cross_loss(self, x, pos_x_a, pos_x_b, neg_x_a, neg_x_b):
        """intra-contrastive"""

        pos = self.similarity_score(x, pos_x_a) + self.similarity_score(x, pos_x_b)  # numerator

        neg = self.similarity_score(x, neg_x_a) + self.similarity_score(x, neg_x_b)  # denominator

        # -\frac{1}{I} \sum_i log(\frac{numerator}{denominator})
        return torch.mean(-torch.log(torch.div(pos, neg)))

    def compute_contrastive_loss(self, x, pos_x_con, pos_x_sig, neg_x_con, neg_x_sig):
        """contrastive-loss"""

        # x reduce dimention to feature_dim
        # self.x = self.transform(x.to(torch.float32)).to(device)
        self.x = self.transform(x).to(device)

        # Normalization
        self.x = F.normalize(self.x, p=2, dim=1)

        pos_x_con = F.normalize(pos_x_con, p=2, dim=1)
        pos_x_sig = F.normalize(pos_x_sig, p=2, dim=1)

        neg_x_con = F.normalize(neg_x_con, p=2, dim=1)
        pos_x_sig = F.normalize(pos_x_sig, p=2, dim=1)

        # inter-loss
        inter_loss_train_pos = self.compute_per_loss(pos_x_con, pos_x_sig)
        inter_loss_train_neg = self.compute_per_loss(neg_x_con, pos_x_sig)

        inter_loss = inter_loss_train_pos + inter_loss_train_neg

        # intra-loss
        intra_loss_train = self.compute_cross_loss(self.x, pos_x_con, pos_x_sig, neg_x_con, pos_x_sig)

        intra_loss = intra_loss_train

        # (1-\alpha) inter + \alpha intra
        return (1 - self.args.alpha) * inter_loss + self.args.alpha * intra_loss

    def predict(self, x_concat, src_id, dst_id):
        src_x = x_concat[src_id]
        dst_x = x_concat[dst_id]

        return self.predictor(src_x, dst_x)

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


# percent_list = [30, 50, 60, 70, 80, 100]
percent_list = [60]
seed_list = [114, 514, 1919, 810, 721]
lr = 0.005
weight_decay = 5e-4

args.alpha = 0.2
args.beta = 0.0001
args.tau = 0.05
args.aug_ratio = 0.1


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
        N = torch.max(train_pos_edge_index).item()
        x = dataloader.create_feature(N)
        args.x = x

        model = MySGCL(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1000):

            x = model.dimension_reduction()

            x_concat, *other_x = model(x, N, train_pos_edge_index, train_neg_edge_index)

            # loss
            contrastive_loss = model.compute_contrastive_loss(x_concat, *other_x)

            # train predict
            src_id = torch.concat((train_pos_edge_index[0], train_neg_edge_index[0])).to(device)
            dst_id = torch.concat((train_pos_edge_index[1], train_neg_edge_index[1])).to(device)

            y_train = torch.concat((torch.ones(train_pos_edge_index.shape[1]), torch.zeros(train_neg_edge_index.shape[1]))).to(device)

            score = model.predict(model.x, src_id, dst_id)

            label_loss = model.compute_label_loss(score, y_train)

            loss = args.beta * contrastive_loss + label_loss

            print(f"\repoch {epoch+1}: {loss}", end="", flush=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # test
        print()
        model.eval()

        with torch.no_grad():
            x_concat, *other_x = model(x, N, train_pos_edge_index, train_neg_edge_index)

            # test predict
            test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
            test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

            y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

            score_test = model.predict(model.x, test_src_id, test_dst_id).to(device)

            acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

            print(f"\nacc {acc:.6f}; auc {auc:.6f}; f1 {f1:.6f}; micro_f1 {micro_f1:.6f}; macro_f1 {macro_f1:.6f}")


        res.append((acc, auc, f1, micro_f1, macro_f1))
        print(res[times])

    res = np.array(res)
    print(res.mean(axis=0))
    print(res.std(axis=0))

