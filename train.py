import torch
import torch_geometric
from torch_geometric.utils import negative_sampling
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
import utils
from utils import DataLoad
from model import CSGDN
from itertools import chain
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus"], 
                    help='choose dataset')
parser.add_argument('--times', type=int, default=1,
                    help='Random seed. ( seed = seed_list[args.times] )')
parser.add_argument('--mask_ratio', type=float, default=0.1,
                    help='random mask ratio')
parser.add_argument('--tau', type=float, default=0.05,
                    help='temperature parameter')
parser.add_argument('--beta', type=float, default=1e-4,
                    help='control contribution of loss contrastive')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='control the contribution of inter and intra loss')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--feature_dim', type=int, default=64,
                    help='initial embedding size of node')
parser.add_argument('--epochs', type=int, default=200,
                    help='initial embedding size of node')
parser.add_argument('--predictor', type=str, default="4", 
                    help='predictor method (1-4 Linear)')
parser.add_argument('--ablation', action="store_true",)

args = parser.parse_args()

# cuda / mps / cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

args.device = device

# seed
seed_list = [114]
seed = seed_list[args.times-1]
args.seed = seed

def test(model, test_pos_edge_index, test_neg_edge_index, see_prob=False):

    model.eval()

    with torch.no_grad():

        # test predict
        test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
        test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        prob = model.predict(model.x, test_src_id, test_dst_id).to(device)
        score_test = prob[:, (0, 2)].max(dim=1)[1]

        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

        # print the original gene name
        if args.dataset == "cotton" and see_prob:
            # back to gene name
            gene2idx, idx2gene = DataLoad(args).load_backup_dict()

            test_src_id = test_src_id.cpu().numpy().tolist()
            test_dst_id = test_dst_id.cpu().numpy().tolist()
            y_test = y_test.to(torch.int).cpu().numpy().tolist()
            score_test = score_test.cpu().numpy().tolist()

            test_src_geneId = [idx2gene.get(i) for i in test_src_id]
            test_dst_geneId = [idx2gene.get(i) for i in test_dst_id]

            df = pd.DataFrame({"src": test_src_geneId, "dst": test_dst_geneId, "true": y_test, "predict": score_test, "neg_prob": prob[:, 0].cpu().numpy(), "none_prob": prob[:, 1].cpu().numpy(), "pos_prob": prob[:, 2].cpu().numpy()})
            df.to_csv(f"./results/{args.dataset}/CSGDN/{args.period}_{args.times}.csv", index=False)

        # print(f"\nacc {acc:.6f}; auc {auc:.6f}; f1 {f1:.6f}; micro_f1 {micro_f1:.6f}; macro_f1 {macro_f1:.6f}")

    return acc, auc, f1, micro_f1, macro_f1

def napus_test(csgdn_model, original_x, final_x, train_src_id, train_dst_id, test_pos_edge_index, test_neg_edge_index):
    csgdn_model.eval()

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

    print(loss)

    model.eval()

    # 将 test_pos_edge_index, test_neg_edge_index 中对应在 original 中的 test_original_x 映射到 final_x
    test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
    test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)
    test_id = torch.concat((test_src_id, test_dst_id)).unique()
    test_original_x = original_x[test_id]
    test_final_x = model(test_original_x)
    final_x[test_id] = test_final_x
    print(test_original_x)
    print(test_final_x)

    with torch.no_grad():

        y_test = torch.concat((torch.ones(test_pos_edge_index.shape[1]), torch.zeros(test_neg_edge_index.shape[1]))).to(device)

        # score_test = model.predict(model.x, test_src_id, test_dst_id).to(device)
        if args.predictor == "dot":
            score_test = csgdn_model.predict(final_x, test_src_id, test_dst_id).to(device)
        else:
            score_test = csgdn_model.predict(final_x, test_src_id, test_dst_id)[:, : 2].max(dim=1)[1]

        acc, auc, f1, micro_f1, macro_f1 = csgdn_model.test(score_test, y_test)

        return acc, auc, f1, micro_f1, macro_f1

        # print(f"\nacc {acc:.6f}; auc {auc:.6f}; f1 {f1:.6f}; micro_f1 {micro_f1:.6f}; macro_f1 {macro_f1:.6f}")

    return acc, auc, f1, micro_f1, macro_f1


def train(args):
    
    # train & test dataset ( trp trn tep ten )
    train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = DataLoad(args).load_data_format()

    # original graph & diffusion graph ( tpa tna tpb tnb; dpa dna dpb dnb; id )
    train_pos_edge_index_a, train_neg_edge_index_a, train_pos_edge_index_b, train_neg_edge_index_b, \
            diff_pos_edge_index_a, diff_neg_edge_index_a, diff_pos_edge_index_b, diff_neg_edge_index_b = utils.generate_view(args)
            # node_id_selected = utils.generate_view(args.percent, args.times)

    # In the present case, 4 phenotype must exist and their reidx
    node_num = torch.max(train_pos_edge_index_a).item()

    # feature x
    x = DataLoad(args).create_feature(node_num)
    original_x = x.clone()

    # dimention reduce embedding
    linear_DR = nn.Linear(x.shape[1], args.feature_dim).to(device)

    # def model & optimizer
    model = CSGDN(args)
    optimizer = torch.optim.Adam(chain.from_iterable([model.parameters(), linear_DR.parameters()]), lr=args.lr, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[70, 100, 130, 160], gamma=0.2)

    edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1).to(args.device)

    best_acc, best_auc, best_f1, best_mricro_f1, best_macro_f1 = 0, 0, 0, 0, 0
    best_model = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # x = model.dimension_reduction()
        x = linear_DR(original_x)

        # embedding feature ( tpxa tpxb dpxa dpxb; tnxa tnxb dnxa dnxb )
        train_pos_x_a, train_pos_x_b, diff_pos_x_a, diff_pos_x_b, \
        train_neg_x_a, train_neg_x_b, diff_neg_x_a, diff_neg_x_b \
            = model((train_pos_edge_index_a, train_neg_edge_index_a, train_pos_edge_index_b, train_neg_edge_index_b, 
                    diff_pos_edge_index_a, diff_neg_edge_index_a, diff_pos_edge_index_b, diff_neg_edge_index_b), x)

        # concat x
        x_concat = torch.concat((train_pos_x_a, train_pos_x_b, diff_pos_x_a, diff_pos_x_b, 
                                train_neg_x_a, train_neg_x_b, diff_neg_x_a, diff_neg_x_b), dim=1)

        # compute contrastive loss
        contrastive_loss = model.compute_contrastive_loss(x_concat, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b)

        # predict
        # sample the none edges ( src -> dst 0 )
        none_edge_index = negative_sampling(edge_index, model.x.size(0))

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

        score = model.predict(model.x, src_id, dst_id)
        # score = model.predict(model.x, src_id, dst_id)[:, : 2].max(dim=1)[1].to(torch.float)

        # compute label loss
        label_loss = model.compute_label_loss(score, y_train)

        loss = args.beta * contrastive_loss + (1 - args.beta) * label_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc, auc, f1, micro_f1, macro_f1 = test(model, val_pos_edge_index, val_neg_edge_index)
        print(f"\rtimes {args.times} epoch {epoch+1} done! loss {loss.item()} acc {acc}, auc {auc}, f1 {f1}", end="", flush=True)

        if auc + f1 > best_auc + best_f1:
            best_acc, best_auc, best_f1, best_mricro_f1, best_macro_f1 = acc, auc, f1, micro_f1, macro_f1
            best_model = model

    print(f"\nbest val acc {best_acc} auc {best_auc}, best f1 {best_f1}, micro_f1 {best_mricro_f1}, macro_f1 {best_macro_f1}")

    # test
    if args.dataset == "napus":
        acc, auc, f1, micro_f1, macro_f1 = napus_test(model, original_x, model.x, src_id, dst_id, test_pos_edge_index, test_neg_edge_index)
    else:
        acc, auc, f1, micro_f1, macro_f1 = test(best_model, test_pos_edge_index, test_neg_edge_index, see_prob=True)
        # acc, auc, f1, micro_f1, macro_f1 = test(best_model, test_pos_edge_index, test_neg_edge_index)

    return acc, auc, f1, micro_f1, macro_f1


best = {"4DPA": {'mask_ratio': 0.5, 'alpha': 0.8, 'beta': 0.0001, 'tau': 0.05, 'predictor': '4', 'feature_dim': 64}, 
            50: {'mask_ratio': 0.4, 'alpha': 0.8, 'beta': 0.01, 'tau': 0.1, 'predictor': '2', 'feature_dim': 64}, 
            60: {'mask_ratio': 0.4, 'alpha': 0.2, 'beta': 1e-04, 'tau': 0.05, 'predictor': '4', 'feature_dim': 64}, 
            70: {'mask_ratio': 0.1, 'alpha': 0.2, 'beta': 0.01, 'tau': 0.05, 'predictor': '2', 'feature_dim': 64}, 
            80: {'mask_ratio': 0.3, 'alpha': 0.6000000000000001, 'beta': 9.999999999999999e-05, 'tau': 0.1, 'predictor': 'dot', 'feature_dim': 16}, 
            100: {'mask_ratio': 0.3, 'alpha': 0.4, 'beta': 9.999999999999999e-06, 'tau': 0.05, 'predictor': 'dot', 'feature_dim': 16}}

napus = {'mask_ratio': 0.2, 'alpha': 0.2, 'beta': 0.01, 'tau': 0.05, 'predictor': '2', 'feature_dim': 16}

res_str = []

if __name__ == "__main__":

    if not os.path.exists(f"./results/{args.dataset}/CSGDN"):
        os.makedirs(f"./results/{args.dataset}/CSGDN")

    # load period data
    period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)

    for period_name in period:

        res = []
        args.period = period_name

        # hyper params
        args.mask_ratio = best.get(args.period).get("mask_ratio")
        args.alpha = best.get(args.period).get("alpha")
        args.beta = best.get(args.period).get("beta")
        args.tau = best.get(args.period).get("tau")
        args.predictor = best.get(args.period).get("predictor")
        args.feature_dim = best.get(args.period).get("feature_dim")
        """
        """

        print(args)

        for times in range(5):
            # seed
            args.seed = seed_list[0]
            args.times = times + 1

            torch.random.manual_seed(args.seed)
            torch_geometric.seed_everything(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)

            acc, auc, f1, micro_f1, macro_f1 = train(args)
            print(f"times {times+1}: acc {acc}, auc {auc}, f1 {f1}, micro_f1 {micro_f1}, macro_f1 {macro_f1}")
            print()

            res.append([acc, auc, f1, micro_f1, macro_f1])
        # print()

        # calculate the avg of each times
        res = np.array(res)
        avg = res.mean(axis=0)
        std = res.std(axis=0)
        res_str.append(f"Stage {args.period}: acc {avg[0]:.3f}+{std[0]:.3f}; auc {avg[1]:.3f}+{std[1]:.3f}; f1 {avg[2]:.3f}+{std[2]:.3f}; micro_f1 {avg[3]:.3f}+{std[3]:.3f}; macro_f1 {avg[4]:.3f}+{std[4]:.3f}\n")

        with open(f"./results/{args.dataset}/CSGDN/{args.period}_res.txt", "w") as f:
            for line in res.tolist():
                f.writelines(str(line))
                f.writelines("\n")
            f.writelines("\n")
            f.writelines(res_str[-1])
        """
        """

        break

    for each in res_str:
        print(each)