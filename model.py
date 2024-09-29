import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class CSGDN(nn.Module):

    def __init__(self, args, layer_num = 2) -> None:

        super().__init__()

        self.args = args
        self.in_channels = args.feature_dim
        self.out_channels = args.feature_dim
        self.layer_num = layer_num

        # activation function: \sigma(.)
        self.activation = nn.ReLU()

        # concated x reduce dimention to feature_dim
        self.transform = nn.Sequential(nn.Linear(8 * args.feature_dim, args.feature_dim)).to(self.args.device)

        # predict
        self.predictor = Predictor(self.args)


    def encode(self, edge_index_a, edge_index_b, x):

        x_a, x_b = None, None
        for _ in range(self.layer_num):

            encoder = GATConv(self.in_channels, self.out_channels).to(self.args.device)
            # encoder = GCNConv(self.in_channels, self.out_channels).to(self.args.device)

            # for the graph a
            for _ in range(2):
                x_a = encoder(x, edge_index_a)
                x_a = self.activation(x_a)
            # x_a = encoder(x, edge_index_a).to(self.args.device)
            # x_a = self.activation(x_a).to(self.args.device)

            # for the graph b
            for _ in range(2):
                x_b = encoder(x, edge_index_b)
                x_b = self.activation(x_b)
            # x_b = encoder(x, edge_index_b).to(self.args.device)
            # x_b = self.activation(x_b).to(self.args.device)

        return x_a, x_b


    def forward(self, edge_index, x):

        train_pos_edge_index_a, train_neg_edge_index_a, train_pos_edge_index_b, train_neg_edge_index_b, diff_pos_edge_index_a, diff_neg_edge_index_a, diff_pos_edge_index_b, diff_neg_edge_index_b = edge_index

        # train pos & train neg
        train_pos_x_a, train_pos_x_b = self.encode(train_pos_edge_index_a, train_pos_edge_index_b, x)
        train_neg_x_a, train_neg_x_b = self.encode(train_neg_edge_index_a, train_neg_edge_index_b, x)

        # diffusion pos & diffusion neg
        diff_pos_x_a, diff_pos_x_b = self.encode(diff_pos_edge_index_a, diff_pos_edge_index_b, x)
        diff_neg_x_a, diff_neg_x_b = self.encode(diff_neg_edge_index_a, diff_neg_edge_index_b, x)

        return train_pos_x_a, train_pos_x_b, \
               diff_pos_x_a, diff_pos_x_b, \
               train_neg_x_a, train_neg_x_b, \
               diff_neg_x_a, diff_neg_x_b


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


    def compute_contrastive_loss(self, x, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b):
        """contrastive-loss"""

        # x reduce dimention to feature_dim
        self.x = self.transform(x).to(self.args.device)

        # Normalization
        self.x = F.normalize(self.x, p=2, dim=1)

        train_pos_x_a = F.normalize(train_pos_x_a, p=2, dim=1)
        train_pos_x_b = F.normalize(train_pos_x_b, p=2, dim=1)

        train_neg_x_a = F.normalize(train_neg_x_a, p=2, dim=1)
        train_neg_x_b = F.normalize(train_neg_x_b, p=2, dim=1)

        diff_pos_x_a = F.normalize(diff_pos_x_a, p=2, dim=1)
        diff_pos_x_b = F.normalize(diff_pos_x_b, p=2, dim=1)

        diff_neg_x_a = F.normalize(diff_neg_x_a, p=2, dim=1)
        diff_neg_x_b = F.normalize(diff_neg_x_b, p=2, dim=1)

        # inter-loss
        inter_loss_train_pos = self.compute_per_loss(train_pos_x_a, train_pos_x_b)
        inter_loss_train_neg = self.compute_per_loss(train_neg_x_a, train_neg_x_b)

        inter_loss_diff_pos = self.compute_per_loss(diff_pos_x_a, diff_pos_x_b)
        inter_loss_diff_neg = self.compute_per_loss(diff_neg_x_a, diff_neg_x_b)

        inter_loss = inter_loss_train_pos + inter_loss_train_neg + inter_loss_diff_pos + inter_loss_diff_neg

        # intra-loss
        intra_loss_train = self.compute_cross_loss(self.x, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b)
        intra_loss_diff = self.compute_cross_loss(self.x, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b)

        intra_loss = intra_loss_train + intra_loss_diff

        # (1-\alpha) inter + \alpha intra
        return (1 - self.args.alpha) * inter_loss + self.args.alpha * intra_loss


    def predict(self, x, edge_index):
        """predict training dataset"""
        score = self.predictor(x[edge_index[0]], x[edge_index[1]])

        # return F.softmax(score, dim=1)
        return torch.log_softmax(score, dim=1)


    def compute_label_loss(self, x, train_pos_edge_index, train_neg_edge_index):
        """label-loss"""
        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1).to(self.args.device)
        none_edge_index = negative_sampling(edge_index, x.size(0))

        pos_score = self.predict(x, train_pos_edge_index)
        neg_score = self.predict(x, train_neg_edge_index)
        none_score = self.predict(x, none_edge_index)

        nll_loss = 0
        nll_loss += F.nll_loss(
            pos_score,
            train_pos_edge_index.new_full((train_pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            neg_score,
            train_neg_edge_index.new_full((train_neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            none_score,
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        """
        nll_loss += F.cross_entropy(
            pos_score,
            train_pos_edge_index.new_full((train_pos_edge_index.size(1), ), 0))
        nll_loss += F.cross_entropy(
            neg_score,
            train_neg_edge_index.new_full((train_neg_edge_index.size(1), ), 1))
        nll_loss += F.cross_entropy(
            none_score,
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        """

        return nll_loss / 3.0
    
    def loss(self, x, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b, train_pos_edge_index_a, train_neg_edge_index_a):
        """loss function"""
        # contrastive loss
        contrastive_loss = self.compute_contrastive_loss(x, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b)

        # label loss
        label_loss = self.compute_label_loss(self.x, train_pos_edge_index_a, train_neg_edge_index_a)

        return self.args.beta * contrastive_loss + (1 - self.args.beta) * label_loss


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
        res = self.predictor(x)

        return res
