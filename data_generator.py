import numpy as np
import pandas as pd
import torch
import os
import argparse
from diffusion import Diffusion
from torch_geometric.nn import SignedGCN
from utils import DataLoad, remove_edges

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus", "cotton_80"], 
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

dataset = pd.read_excel(f"./data/TWAS.xlsx", sheet_name=args.dataset)
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
if args.dataset == "napus":
    for idx, pheo in enumerate(period):  # different period of SOC are seen as different pheo
        idx2gene[idx+max_gene_idx+1] = pheo
        gene2idx[pheo] = idx+max_gene_idx+1
else:
    for idx, pheo in enumerate(pheo_name):
        idx2gene[idx+max_gene_idx+1] = pheo
        gene2idx[pheo] = idx+max_gene_idx+1

def wheat_graph(p_value = 0.05):

    # read the correlation matrix
    cor = ["BM_cor", "RL_cor", "RA_cor", "RV_cor"]
    p = ["BM_pvalue", "RL_pvalue", "RA_pvalue", "RV_pvalue"]

    df = pd.read_excel(f"./data/wheat_correlation_matrix.xlsx").fillna(p_value)

    cor_data = torch.tensor([df[col] for col in cor]).T.to(args.device)
    p_data = torch.tensor([df[col] for col in p]).T.to(args.device)

    idx = torch.where(p_data < p_value)  # ph idx 0 1 2 3 not +70199 yet
    cor_data = cor_data[idx]

    cor_data[cor_data < 0] = -1
    cor_data[cor_data > 0] = 1

    cor_data = torch.cat((idx[0].reshape(-1, 1), idx[1].reshape(-1, 1), cor_data.reshape(-1, 1)), dim=1).to(torch.int)

    # in cor_data, gene name: gene-xxx, pheo name: xxx_cor
    cor_idx2gene = {idx: gene[5:] for idx, gene in enumerate(df["Gene"])}
    cor_gene2idx = {gene[5:]: idx for idx, gene in enumerate(df["Gene"])}
    cor_idx2pheo = {idx: pheo[: 2] for idx, pheo in enumerate(cor)}
    cor_pheo2idx = {pheo[: 2]: idx for idx, pheo in enumerate(cor)}

    # read the TWAS matrix
    for period_name in period:
        
        args.period = period_name
        cur_period_index = dataset["Stage"] == period_name
        # extract current period dataset
        cur_period_dataset = dataset[cur_period_index].iloc[:, [2, 0, 3]]

        gene_set = set()
        pheo_set = set()
        cur_dict = {}
        for cur_period_data in cur_period_dataset.itertuples():
            triad = cur_period_data[1: ]
            gene_name = triad[0]
            pheo_name = triad[1]
            gene_set.add(gene_name)
            pheo_set.add(pheo_name)
            if cur_dict.get(gene_name):
                cur_dict[gene_name].update({pheo_name: triad[2]})
            else:
                cur_dict[gene_name] = {pheo_name: triad[2]}
        
        train_data = pd.DataFrame(columns=["GeneID", "Phenotype", "TWAS.Zscore"])
        cur_num = cur_period_dataset.shape[0]
        train_num = int(cur_num * 8 // 3)
        M = train_num
        i = 0
        while train_num:
            triad = cor_data[i]
            gene_name = cor_idx2gene[triad[0].item()]
            pheo_name = cor_idx2pheo[triad[1].item()]
            if not (cur_dict.get(gene_name) and cur_dict[gene_name].get(pheo_name)):
                gene_set.add(gene_name)
                pheo_set.add(pheo_name)
                train_data = train_data.merge(pd.DataFrame([[gene_name, pheo_name, triad[2].item()]], columns=["GeneID", "Phenotype", "TWAS.Zscore"]), how="outer")
                train_num -= 1
            i += 1

        idx2gene = {idx: gene for idx, gene in enumerate(gene_set)}
        gene2idx = {gene: idx for idx, gene in enumerate(gene_set)}
        max_gene_idx = max(idx2gene.keys())
        for idx, pheo in enumerate(pheo_set):
            idx2gene[idx+max_gene_idx+1] = pheo
            gene2idx[pheo] = idx+max_gene_idx+1

        # merge cur_period_dataset and train_data, they have the same columns header
        cur_period_dataset = pd.concat([train_data, cur_period_dataset], axis=0, join='outer')
        # row index is not continuous, so we need to reset the index
        cur_period_dataset = cur_period_dataset.reset_index(drop=True)

        cur_period_dataset["Phenotype"] = cur_period_dataset["Phenotype"].map(gene2idx)
        cur_period_dataset["GeneID"] = cur_period_dataset["GeneID"].map(gene2idx)

        data = torch.tensor(cur_period_dataset.values).to(device)

        data[data[:, 2] > 0, 2] = 1
        data[data[:, 2] < 0, 2] = -1

        train_data = data[: M]
        val_test_data = data[M: ]

        # split val and test
        val_data = val_test_data[: int(cur_num * 0.3)]
        test_data = val_test_data[int(cur_num * 0.3): ]

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_training.txt", train_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_validation.txt", val_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_test.txt", test_data.cpu().numpy(), fmt='%d', delimiter='\t')

        return idx2gene, gene2idx

def generate_graph():

    for period_name in period:

        args.period = period_name

        # find all index of current preiod ( e.g 4PDA )
        if args.dataset == "napus":
            cur_period_index = dataset["Phenotype"] == "SOC"
            cur_period_dataset = dataset[cur_period_index].iloc[:, [1, 2, 3]]
        else:
            cur_period_index = dataset["Stage"] == period_name
            # extract current period dataset
            cur_period_dataset = dataset[cur_period_index].iloc[:, [0, 2, 3]]

        # reindex
        if args.dataset != "cotton_80":

            if args.dataset == "napus":
                cur_period_dataset["Stage"] = cur_period_dataset["Stage"].map(gene2idx)
            else:
                cur_period_dataset["Phenotype"] = cur_period_dataset["Phenotype"].map(gene2idx)
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

        # if the dataset is cotton_80, we regard them as the training data, and then we use the cotton dataset data as the val and test data. train: val: test = 7:1:2
        else:
            cur_period_dataset["Phenotype"] = cur_period_dataset["Phenotype"].map(gene2idx)
            cur_period_dataset["GeneID"] = cur_period_dataset["GeneID"].map(gene2idx)
            # to tensor
            data = torch.tensor(cur_period_dataset.values).to(device)
            data[data[:, 2] > 0, 2] = 1
            data[data[:, 2] < 0, 2] = -1
            data = data[:, (1, 0, 2)].to(torch.int32)
            M = data.size(0)

            # load the 100% data
            args.dataset = "cotton"
            dataloader = DataLoad(args)
            train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = dataloader.load_data_format()
            gene2idx_100, idx2gene_100 = dataloader.load_backup_dict()
            data_100 = torch.concat((train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index), dim=1)
            data_100_value = torch.cat([torch.ones(train_pos_edge_index.size(1)), -torch.ones(train_neg_edge_index.size(1)), torch.ones(val_pos_edge_index.size(1)), -torch.ones(val_neg_edge_index.size(1)), torch.ones(test_pos_edge_index.size(1)), -torch.ones(test_neg_edge_index.size(1))], dim=0).to(args.device).reshape(1, -1)
            data_100 = torch.cat([data_100, data_100_value], dim=0).T
            
            mask = torch.randperm(data_100.size(0))
            data_100 = data_100[mask]

            data_dict = {}
            for i in range(M):
                gene_name = idx2gene[data[i][0].item()]
                pheo_name = idx2gene[data[i][1].item()]
                if data_dict.get(gene_name):
                    data_dict[gene_name].update({pheo_name: data[i][2]})
                    # data_dict[data[i][0]].update({data[i][1]: [i, data[i][2]]})
                else:
                    data_dict[gene_name] = {pheo_name: data[i][2]}
                    # data_dict[data[i][0]] = {data[i][1]: [i, data[i][2]]}

            args.dataset = "cotton_80"

            val_len = int(M / 8)
            test_len = int(M / 8) * 2
            val_data = torch.tensor([]).to(args.device)
            test_data = torch.tensor([]).to(args.device)
            for i in range(data_100.size(0)):
                gene_name = idx2gene_100[data_100[i][0].item()]
                pheo_name = idx2gene_100[data_100[i][1].item()]
                if val_data.size(0) == val_len and test_data.size(0) == test_len:
                    break
                if (data_dict.get(gene_name) and data_dict[gene_name].get(pheo_name)) or not gene2idx.get(gene_name):
                    continue

                triad = torch.tensor([gene2idx[gene_name], gene2idx[pheo_name], data_100[i][2]]).to(args.device).reshape(1, -1)
                if val_data.size(0) < val_len:
                    val_data = torch.concat([val_data, triad], dim=0)
                else:
                    test_data = torch.concat([test_data, triad], dim=0)

            train_data = data

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_training.txt", train_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_validation.txt", val_data.cpu().numpy(), fmt='%d', delimiter='\t')
        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{period_name}_test.txt", test_data.cpu().numpy(), fmt='%d', delimiter='\t')


def generate_feature(period, feature_dim = 32):

    print("generating similarity adjacency matrix...")

    if True:

        # use the similarity matrix among genes 

        gene_sim_data = pd.read_table(f"./data/{args.dataset}/ori_sim.txt", header=None, usecols=(0, 1, 2))
        triad_data = []
        for each in gene_sim_data.itertuples():
            # the gene which our dataset does not contain
            if not (gene2idx.get(each[1]) and gene2idx.get(each[2])):
                continue
            triad_data.append([gene2idx[each[1]], gene2idx[each[2]], each[3]])

        triad_data = torch.tensor(triad_data).to(device)

        N = len(gene2idx)

        sim_adjmat = torch.eye(N)

        for sim_a, sim_b, sim_score in triad_data:
            sim_a = int(sim_a)
            sim_b = int(sim_b)
            sim_adjmat[sim_a, sim_b] = sim_score / 100
            sim_adjmat[sim_b, sim_a] = sim_score / 100

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_feature.txt", sim_adjmat.cpu().numpy(), fmt='%.2f', delimiter='\t')

    """
    # use the spectral feature
    else:

        args.period = period
        args.feature_dim = feature_dim

        train_pos_edge_index, train_neg_edge_index, _, _, _, _ = DataLoad(args).load_data_format()
        model = SignedGCN(args.feature_dim, args.feature_dim, num_layers=2, lamb=5).to(device)
        x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

        np.savetxt(f"./data/{args.dataset}/{args.dataset}_{args.period}_feature.txt", x.cpu().numpy(), delimiter="\t", fmt="%.2f")
    """

def ptb_graph(ptb_ratio = 0.1, period = "4DPA"):

    args.period = period

    train_data_index, train_data_value, _, _, _, _ = DataLoad(args).load_data()
    train_data_index, train_data_value, to_another_index, to_another_value = remove_edges(args, train_data_index, train_data_value, ptb_ratio)

    # train data
    to_another_value = - to_another_value
    train_data_index = torch.cat([train_data_index, to_another_index], dim=1)
    train_data_value = torch.cat([train_data_value, to_another_value], dim=0)
    train_data = torch.cat([train_data_index, train_data_value.unsqueeze(0)], dim=0).T

    # save
    np.savetxt(f"./data/{args.dataset}/{args.dataset}_{args.period}_training.txt", train_data.cpu().numpy(), delimiter="\t", fmt="%d")

    Diffusion(args).generate_diffusion_graph()
        

if __name__ == "__main__":
    # save the dict
    if args.dataset == "wheat":
        idx2gene, gene2idx = wheat_graph()
    else:
        generate_graph()

    generate_feature(args.period)

    np.save(f"./data/{args.dataset}/{args.dataset}_period.npy", period)
    np.save(f"./data/{args.dataset}/{args.dataset}_idx2gene.npy", idx2gene)
    np.save(f"./data/{args.dataset}/{args.dataset}_gene2idx.npy", gene2idx)

    period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)
    for period_name in period:
        args.period = period_name
        Diffusion(args).generate_diffusion_graph()