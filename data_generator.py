import numpy as np
import pandas as pd
import torch
import os
import argparse
from diffusion import Diffusion
from torch_geometric.nn import SignedGCN
from utils import DataLoad

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus"], 
                    help='choose dataset')

args = parser.parse_args()

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

args.device = device

dataset = pd.read_excel(f"./data/TWAS_out_240913.xlsx", sheet_name=args.dataset)
seed = 114514
torch.manual_seed(seed)

if not os.path.exists(f"./data/{args.dataset}"):
    os.makedirs(f"./data/{args.dataset}")

# Phenotype	Stage	GeneID	TWAS.Zscore
period = dataset["Stage"].unique()
pheo_name = dataset["Phenotype"].unique()
gene_name = dataset["GeneID"].unique()  # for all stage gene

# str type -> float type ( gene and pheo name -> index ) ( e.g. for cotton  0: Ghir_A01G002290 )
# NOTE: although named gene, but it also contains pheo
idx2gene = {idx: gene for idx, gene in enumerate(gene_name)}
gene2idx = {gene: idx for idx, gene in enumerate(gene_name)}
max_gene_idx = max(idx2gene.keys())
if args.dataset == "cotton":
    for idx, pheo in enumerate(pheo_name):
        idx2gene[idx+max_gene_idx+1] = pheo
        gene2idx[pheo] = idx+max_gene_idx+1
elif args.dataset == "napus":
    for idx, pheo in enumerate(period):  # different period of SOC are seen as different pheo
        idx2gene[idx+max_gene_idx+1] = pheo
        gene2idx[pheo] = idx+max_gene_idx+1

def generate_graph():

    for period_name in period:

        # find all index of current preiod ( e.g 4PDA )
        if args.dataset == "cotton":
            cur_period_index = dataset["Stage"] == period_name
            # extract current period dataset
            cur_period_dataset = dataset[cur_period_index].iloc[:, [0, 2, 3]]
        elif args.dataset == "napus":
            cur_period_index = dataset["Phenotype"] == "SOC"
            cur_period_dataset = dataset[cur_period_index].iloc[:, [1, 2, 3]]

        # reindex
        if args.dataset == "cotton":
            cur_period_dataset["Phenotype"] = cur_period_dataset["Phenotype"].map(gene2idx)
        elif args.dataset == "napus":
            cur_period_dataset["Stage"] = cur_period_dataset["Stage"].map(gene2idx)
        cur_period_dataset["GeneID"] = cur_period_dataset["GeneID"].map(gene2idx)

        # to tensor
        data = torch.tensor(cur_period_dataset.values).to(device)

        data[data[:, 2] > 0, 2] = 1
        data[data[:, 2] < 0, 2] = -1

        # gene pheo sign_val
        data = data[:, (1, 0, 2)].to(torch.int32)

        # split train and test
        shuffle = torch.randperm(data.size(0))
        data = data[shuffle]

        train_data = data[: int(data.size(0)*0.7)]
        val_data = data[int(data.size(0)*0.7): int(data.size(0)*0.8)]
        test_data = data[int(data.size(0)*0.8):]

        """
        # remove the node which does not appear in the train data
        visited_dict = {node: 1 for node in train_data[:, 0]}
        
        val_mask = torch.tensor([True if visited_dict.get(node) else False for node in val_data[:, 0]])
        test_mask = torch.tensor([True if visited_dict.get(node) else False for node in test_data[:, 0]])

        train_data = torch.concat((train_data, val_data[~val_mask], test_data[~test_mask]), dim=0)
        test_data = test_data[test_mask]

        # resize the val and test data, val:test = 1:2
        val_test = torch.concat((val_data, test_data), dim=0)
        val_data = val_test[: int(val_test.size(0)*0.3)]
        test_data = val_test[int(val_test.size(0)*0.3):]
        """

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_training.txt", train_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_validation.txt", val_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_test.txt", test_data.cpu().numpy(), fmt='%d', delimiter='\t')


def generate_feature(period = "4DPA", feature_dim = 64):

    print("generating similarity adjacency matrix...")

    if args.dataset == "cotton":

        # use the similarity matrix among genes 

        gene_sim_data = pd.read_table(f"./data/{args.dataset}/ori_sim.txt", header=None, usecols=(0, 1, 2))
        triad_data = []
        for each in gene_sim_data.itertuples():
            # the gene which our dataset does not contain
            if not (gene2idx.get(each[1]) and gene2idx.get(each[2])):
                continue
            triad_data.append([gene2idx[each[1]], gene2idx[each[2]], each[3]])

        # TODO triad_data have a lot of duplicate data
        triad_data = torch.tensor(triad_data).to(device)

        N = len(gene2idx)

        sim_adjmat = torch.ones((N, N))

        for sim_a, sim_b, sim_score in triad_data:
            sim_a = int(sim_a)
            sim_b = int(sim_b)
            sim_adjmat[sim_a, sim_b] = sim_score / 100
            sim_adjmat[sim_b, sim_a] = sim_score / 100

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_feature.txt", sim_adjmat.cpu().numpy(), fmt='%.2f', delimiter='\t')

    elif args.dataset == "napus":

        # use the spectral feature

        args.period = period
        args.feature_dim = feature_dim

        train_pos_edge_index, train_neg_edge_index, _, _, _, _ = DataLoad(args).load_data_format()
        model = SignedGCN(args.feature_dim, args.feature_dim, num_layers=2, lamb=5).to(device)
        x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

        np.savetxt(f"./data/napus/{args.dataset}_{args.period}_feature.txt", x.cpu().numpy(), delimiter="\t", fmt="%.2f")
        

if __name__ == "__main__":
    # save the dict
    """
    """
    np.save(f"./data/{args.dataset}/{args.dataset}_period.npy", period)
    np.save(f"./data/{args.dataset}/{args.dataset}_idx2gene.npy", idx2gene)
    np.save(f"./data/{args.dataset}/{args.dataset}_gene2idx.npy", gene2idx)

    generate_graph()

    # generate_feature()

    period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)
    for period_name in period:
        args.period = period_name
        Diffusion(args).generate_diffusion_graph()


