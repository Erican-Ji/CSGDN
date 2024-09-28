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

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus", "cotton_80"], 
                    help='choose dataset')
parser.add_argument('--times', type=int, default=1,
                    help='Random seed. ( seed = seed_list[args.times] )')
parser.add_argument('--mask_ratio', type=float, default=0.4,
                    help='random mask ratio')
parser.add_argument('--tau', type=float, default=0.05,
                    help='temperature parameter')
parser.add_argument('--beta', type=float, default=0.01,
                    help='control contribution of loss contrastive')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='control the contribution of inter and intra loss')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=5e-4,
                    help='dropout rate.')
parser.add_argument('--feature_dim', type=int, default=64,
                    help='initial embedding size of node')
parser.add_argument('--epochs', type=int, default=500,
                    help='initial embedding size of node')
parser.add_argument('--predictor', type=str, default="2", 
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
# 11451
# 41919
# 81007
# 21
seed_list = [1482, 1111, 490, 510, 197]
seed_list = [114]
seed = seed_list[args.times-1]
args.seed = seed

def test(model, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, see_prob=False):

    model.eval()

    edge_idx = torch.concat([train_pos_edge_index, train_neg_edge_index], dim=1).unique().to(device)

    # mapping model: map the original feature to the final feature
    mapping_model = nn.Sequential(nn.Linear(model.x.shape[1], model.x.shape[1]),
                                  nn.ReLU(),
                                  nn.Linear(model.x.shape[1], model.x.shape[1])).to(device)
    mapping_loss = nn.MSELoss()
    mapping_optimizer = torch.optim.Adam(mapping_model.parameters(), lr=0.01, weight_decay=args.dropout)

    x_original = model.x[edge_idx].detach()
    
    for epoch in range(50):
        mapping_model.train()
        mapping_optimizer.zero_grad()
        x_hat = mapping_model(x_original)
        loss = mapping_loss(x_hat, x_original)
        loss.backward()
        mapping_optimizer.step()
        # print(f"\rmapping epoch {epoch+1} done: loss {loss}", end="", flush=True)

    mapping_model.eval()

    with torch.no_grad():

        # original feature to final feature
        test_edge_idx = torch.concat([test_pos_edge_index, test_neg_edge_index], dim=1).unique().to(device)
        model.x[test_edge_idx] = mapping_model(model.x[test_edge_idx])

        pos_log_prob = model.predict(model.x, test_pos_edge_index)
        pos_score = pos_log_prob[:, :2].max(dim=1)[1]
        neg_log_prob = model.predict(model.x, test_neg_edge_index)
        neg_score = neg_log_prob[:, :2].max(dim=1)[1]
        score_test = (1 - torch.cat([pos_score, neg_score]))

        y_test = torch.cat(
            [score_test.new_ones((pos_score.size(0))),
             score_test.new_zeros(neg_score.size(0))])

        acc, auc, f1, micro_f1, macro_f1 = model.test(score_test, y_test)

        # print the original gene name
        if args.dataset == "cotton" and see_prob:

            # test predict
            test_src_id = torch.concat((test_pos_edge_index[0], test_neg_edge_index[0])).to(device)
            test_dst_id = torch.concat((test_pos_edge_index[1], test_neg_edge_index[1])).to(device)
            
            pos_prob = torch.exp(pos_log_prob)
            neg_prob = torch.exp(neg_log_prob)
            prob = torch.concat([pos_prob, neg_prob], dim=0)
            # back to gene name
            gene2idx, idx2gene = DataLoad(args).load_backup_dict()

            test_src_id = test_src_id.cpu().numpy().tolist()
            test_dst_id = test_dst_id.cpu().numpy().tolist()
            y_test = y_test.to(torch.int).cpu().numpy().tolist()
            score_test = score_test.cpu().numpy().tolist()

            test_src_geneId = [idx2gene.get(i) for i in test_src_id]
            test_dst_geneId = [idx2gene.get(i) for i in test_dst_id]

            df = pd.DataFrame({"src": test_src_geneId, "dst": test_dst_geneId, "true": y_test, "predict": score_test, "neg_prob": prob[:, 1].cpu().numpy(), "none_prob": prob[:, 2].cpu().numpy(), "pos_prob": prob[:, 0].cpu().numpy()})
            df.to_csv(f"./results/{args.dataset}/CSGDN/{args.period}_{args.times}.csv", index=False)

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
    # scheduler = MultiStepLR(optimizer=optimizer, milestones=[100], gamma=0.02)

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

        loss = model.loss(x_concat, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b, diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b, train_pos_edge_index, train_neg_edge_index)

        loss.backward()
        optimizer.step()
        # scheduler.step()

        acc, auc, f1, micro_f1, macro_f1 = test(model, train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index)
        print(f"\rtimes {args.times} epoch {epoch+1} done! loss {loss.item()} acc {acc}, auc {auc}, f1 {f1}", end="", flush=True)

        # if auc > best_auc:
        # if auc + f1 > best_auc + best_f1:
        if auc + f1 >= best_auc + best_f1:
            best_acc, best_auc, best_f1, best_mricro_f1, best_macro_f1 = acc, auc, f1, micro_f1, macro_f1
            best_model = model

    print(f"\nbest val acc {best_acc} auc {best_auc}, best f1 {best_f1}, micro_f1 {best_mricro_f1}, macro_f1 {best_macro_f1}")

    # test
    # acc, auc, f1, micro_f1, macro_f1 = test(best_model, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, see_prob=True)
    acc, auc, f1, micro_f1, macro_f1 = test(best_model, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, see_prob=False)

    return acc, auc, f1, micro_f1, macro_f1


cotton = {"4DPA": {'mask_ratio': 0.8, 'alpha': 0.2, 'beta': 0.01, 'tau': 0.05, 'predictor': '1', 'feature_dim': 128},
# best = {"4DPA": {'mask_ratio': 0, 'alpha': 0.2, 'beta': 0.01, 'tau': 0.1, 'predictor': '2', 'feature_dim': 64},
# cotton = {"4DPA": {'mask_ratio': 0.4, 'alpha': 0.8, 'beta': 0.01, 'tau': 0.05, 'predictor': '2', 'feature_dim': 64},  # GAT best 0.781
# best = {"4DPA": {'mask_ratio': 0.4, 'alpha': 0.8, 'beta': 0.0001, 'tau': 0.05, 'predictor': '1', 'feature_dim': 16},  # GCN 751
            50: {'mask_ratio': 0.4, 'alpha': 0.8, 'beta': 0.01, 'tau': 0.1, 'predictor': '2', 'feature_dim': 64}, 
            60: {'mask_ratio': 0.4, 'alpha': 0.2, 'beta': 1e-04, 'tau': 0.05, 'predictor': '4', 'feature_dim': 64}, 
            70: {'mask_ratio': 0.1, 'alpha': 0.2, 'beta': 0.01, 'tau': 0.05, 'predictor': '2', 'feature_dim': 64}, 
            80: {'mask_ratio': 0.3, 'alpha': 0.6000000000000001, 'beta': 9.999999999999999e-05, 'tau': 0.1, 'predictor': 'dot', 'feature_dim': 16}, 
            100: {'mask_ratio': 0.3, 'alpha': 0.4, 'beta': 9.999999999999999e-06, 'tau': 0.05, 'predictor': 'dot', 'feature_dim': 16}}

napus = {"20DPA": {'mask_ratio': 0.8, 'alpha': 0.2, 'beta': 0.1, 'tau': 0.05, 'predictor': '1', 'feature_dim': 32}, }

cotton_80 = {"4DPA": {'mask_ratio': 0.2, 'alpha': 0.2, 'beta': 0.001, 'tau': 0.05, 'predictor': '2', 'feature_dim': 16}}

if args.dataset == "cotton":
    best = cotton
elif args.dataset == "napus":
    best = napus
elif args.dataset == "wheat":
    pass
    # args.feature_dim = 16
    # args.lr = 0.001
    # args.dropout = 0.1
    # args.mask_ratio = 0
elif args.dataset == "cotton_80":
    best = cotton_80

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
        """
        args.mask_ratio = best.get(args.period).get("mask_ratio")
        args.alpha = best.get(args.period).get("alpha")
        args.beta = best.get(args.period).get("beta")
        args.tau = best.get(args.period).get("tau")
        args.predictor = best.get(args.period).get("predictor")
        args.feature_dim = best.get(args.period).get("feature_dim")
        """

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
        res_str.append(f"Stage {args.period}: acc {avg[0]:.4f}+{std[0]:.4f}; auc {avg[1]:.4f}+{std[1]:.4f}; f1 {avg[2]:.4f}+{std[2]:.4f}; micro_f1 {avg[4]:.4f}+{std[4]:.4f}; macro_f1 {avg[4]:.4f}+{std[4]:.4f}\n")

        """
        """
        with open(f"./results/{args.dataset}/CSGDN/{args.period}_res.txt", "w") as f:
            for line in res.tolist():
                f.writelines(str(line))
                f.writelines("\n")
            f.writelines("\n")
            f.writelines(res_str[-1])

        break

    for each in res_str:
        print(each)

"""
times 1 epoch 500 done! loss -0.19210490584373474 acc 0.7014925373134329, auc 0.7164246823956442, f1 0.705882352941176504
best val acc 0.8656716417910447 auc 0.8693284936479129, best f1 0.8524590163934426, micro_f1 0.8656716417910447, macro_f1 0.864585672580283
times 1: acc 0.7744360902255639, auc 0.7837552742616035, f1 0.75, micro_f1 0.7744360902255639, macro_f1 0.7722602739726028

times 2 epoch 500 done! loss -0.21233895421028137 acc 0.8507462686567164, auc 0.8480036297640653, f1 0.82758620689655176
best val acc 0.9402985074626866 auc 0.9432849364791289, best f1 0.9333333333333333, micro_f1 0.9402985074626866, macro_f1 0.9396396396396396
times 2: acc 0.8270676691729323, auc 0.80168776371308, f1 0.7578947368421053, micro_f1 0.8270676691729323, macro_f1 0.8116959064327485

times 3 epoch 500 done! loss -0.1913270503282547 acc 0.746268656716418, auc 0.764065335753176, f1 0.7536231884057971035
best val acc 0.9253731343283582 auc 0.9219600725952813, best f1 0.9122807017543859, micro_f1 0.9253731343283582, macro_f1 0.9236728184096605
times 3: acc 0.7593984962406015, auc 0.779887482419128, f1 0.75, micro_f1 0.7593984962406015, macro_f1 0.7590579710144927

times 4 epoch 500 done! loss -0.23301105201244354 acc 0.7164179104477612, auc 0.7254990925589837, f1 0.70769230769230774
best val acc 0.8805970149253731 auc 0.882486388384755, best f1 0.8666666666666667, micro_f1 0.8805970149253731, macro_f1 0.8792792792792793
times 4: acc 0.7744360902255639, auc 0.7691045475855602, f1 0.7272727272727273, micro_f1 0.7744360902255639, macro_f1 0.7674825174825175

times 5 epoch 500 done! loss -0.12970617413520813 acc 0.7910447761194029, auc 0.8035390199637025, f1 0.78787878787878789
best val acc 0.8805970149253731 auc 0.8865698729582576, best f1 0.8709677419354839, micro_f1 0.8805970149253731, macro_f1 0.8799283154121864
times 5: acc 0.7669172932330827, auc 0.7774261603375529, f1 0.743801652892562, micro_f1 0.7
"""