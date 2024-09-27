import os
import numpy as np
import torch

class DataLoad:

    def __init__(self, args) -> None:
        self.args = args

    def load_data_format(self):
        """
        load dataset and format the data to pos & neg edge index
        return train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
        """
    
        train_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_training.txt", delimiter="\t", dtype=int)).to(self.args.device)
        val_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_validation.txt", delimiter="\t", dtype=int)).to(self.args.device)
        test_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_test.txt", delimiter="\t", dtype=int)).to(self.args.device)

        # train
        train_data_idx = train_data[:, :2].T  # (m, 3) -> (2, m) only edge index
        train_dataset = train_data[:, 2]  # (m, 3) -> (m, ) only edge value

        train_pos_edge_index = train_data_idx[:, train_dataset > 0]
        train_neg_edge_index = train_data_idx[:, train_dataset < 0]

        # val
        val_data_idx = val_data[:, :2].T
        val_dataset = val_data[:, 2]

        val_pos_edge_index = val_data_idx[:, val_dataset > 0]
        val_neg_edge_index = val_data_idx[:, val_dataset < 0]

        # test
        test_data_idx = test_data[:, :2].T
        test_dataset = test_data[:, 2]

        test_pos_edge_index = test_data_idx[:, test_dataset > 0]
        test_neg_edge_index = test_data_idx[:, test_dataset < 0]

        return train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index

    def load_data(self):
        """return directly"""
        
        train_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_training.txt", delimiter="\t", dtype=int)).to(self.args.device)
        val_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_validation.txt", delimiter="\t", dtype=int)).to(self.args.device)
        test_data = torch.tensor(np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_test.txt", delimiter="\t", dtype=int)).to(self.args.device)

        # train
        train_data_idx = train_data[:, :2].T  # (m, 3) -> (2, m) only edge index
        train_dataset = train_data[:, 2]  # (m, 3) -> (m, ) only edge value

        # val
        val_data_idx = val_data[:, :2].T
        val_dataset = val_data[:, 2]

        # test
        test_data_idx = test_data[:, :2].T
        test_dataset = test_data[:, 2]

        return train_data_idx, train_dataset, val_data_idx, val_dataset, test_data_idx, test_dataset


    def create_feature(self, node_num):
        """
        load feature matrix from the file
        return feature matrix: torch tensor
        """

        # node_num can be used to create spectral_features
        if self.args.dataset == "wheat":
            from data_generator import generate_feature
            if not os.path.exists(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_feature.txt"):
                generate_feature(self.args.period, self.args.feature_dim)
            sim_matrix = np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_feature.txt", delimiter="\t")

        else:
            sim_matrix = np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_feature.txt", delimiter="\t")

        return torch.tensor(sim_matrix, dtype=torch.float32).to(self.args.device)


    def load_diffusion_data(self):
        """
        load the diffusion training data and split the pos and neg
        return data_index, data_value
        """

        diffusion_graph = np.loadtxt(f"./data/{self.args.dataset}/{self.args.dataset}_{self.args.period}_diffusion.txt", delimiter="\t")

        diffusion_graph = torch.tensor(diffusion_graph, dtype=torch.int).to(self.args.device)
        
        data_index = diffusion_graph[:, :2].T
        data_value = diffusion_graph[:, 2]

        """ return pos & neg edge index
        diff_pos_edge_index = data_index[:, data_value > 0]
        diff_neg_edge_index = data_index[:, data_value < 0]

        return diff_pos_edge_index, diff_neg_edge_index
        """

        return data_index, data_value

    def load_backup_dict(self):

        gene2idx = np.load(f"./data/{self.args.dataset}/{self.args.dataset}_gene2idx.npy", allow_pickle=True).item()
        idx2gene = np.load(f"./data/{self.args.dataset}/{self.args.dataset}_idx2gene.npy", allow_pickle=True).item()

        return gene2idx, idx2gene

# ========================================================================================================================================================================================

    def load_gene_simMat(self):
        """
        load data in original adjacency files ( g-g-{percent}.csv )
        return adjmat: np array
        """
        return np.loadtxt(f"../data/cotton-data/g-g-{self.percent}.csv", delimiter=",")


def remove_edges(args, edge_index, edge_value, mask_ratio = 0.1):

    mask = torch.rand(size=(1, edge_index.shape[1])).to(args.device)  # uniform distribution
    mask[mask < mask_ratio] = 0
    mask[mask >= mask_ratio] = 1

    mask = mask[0].bool()

    return edge_index[:, mask], edge_value.reshape(1, -1)[:, mask][0]
    

def split_pos_neg(edge_index, edge_value):

    pos_edge_index = edge_index[:, edge_value > 0]
    neg_edge_index = edge_index[:, edge_value < 0]

    return pos_edge_index, neg_edge_index


def generate_view(args):

    # load original graph's edge index data
    dataloader = DataLoad(args)
    train_data_index, train_data_value, _, _, _, _ = dataloader.load_data()
    diff_data_index, diff_data_value = dataloader.load_diffusion_data()

    # random remove edges ( ori & diff -> ori_a, ori_b, diff_a, diff_b ) and
    # split to pos & neg
    train_pos_edge_index_a, train_neg_edge_index_a = split_pos_neg(*remove_edges(args, train_data_index, train_data_value))
    train_pos_edge_index_b, train_neg_edge_index_b = split_pos_neg(*remove_edges(args, train_data_index, train_data_value))

    if args.ablation:
        # w/o diffusion
        diff_pos_edge_index_a, diff_neg_edge_index_a = split_pos_neg(*remove_edges(args, train_data_index, train_data_value))
        diff_pos_edge_index_b, diff_neg_edge_index_b = split_pos_neg(*remove_edges(args, train_data_index, train_data_value))
    else:
        diff_pos_edge_index_a, diff_neg_edge_index_a = split_pos_neg(*remove_edges(args, diff_data_index, diff_data_value))
        diff_pos_edge_index_b, diff_neg_edge_index_b = split_pos_neg(*remove_edges(args, diff_data_index, diff_data_value))

    return (train_pos_edge_index_a, train_neg_edge_index_a, train_pos_edge_index_b, train_neg_edge_index_b, 
            diff_pos_edge_index_a, diff_neg_edge_index_a, diff_pos_edge_index_b, diff_neg_edge_index_b)
            # node_id_selected)
