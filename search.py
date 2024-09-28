import torch
import torch_geometric
import numpy as np
from train import train
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cotton", choices = ["cotton", "wheat", "napus", "cotton_80"],
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
seed_list = [1482, 1111, 490, 510, 197]
seed = seed_list[args.times-1]
args.seed = seed

# =======================================================================================================================================

grid = {"mask_ratio": np.array([0.2, 0.4, 0.6, 0.8]), 
        "alpha": np.array([0.2, 0.4, 0.6, 0.8]),
        "beta": np.array([1e-3, 1e-2, 1e-1])}

res_str = []

def search(args):
    best = {}
    best["acc"] = 0
    best["auc"] = 0
    best["f1"] = 0
    best["micro_f1"] = 0
    best["macro_f1"] = 0

    for mask_ratio in grid.get("mask_ratio"):
        for alpha in grid.get("alpha"):
            for beta in grid.get("beta"):

                args.mask_ratio = mask_ratio
                args.alpha = alpha
                args.beta = beta

                print(args)

                res = []
                for times in range(5):
                    # seed
                    args.seed = seed_list[times]
                    args.times = times + 1

                    torch.random.manual_seed(args.seed)
                    torch_geometric.seed_everything(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(args.seed)

                    acc, auc, f1, micro_f1, macro_f1 = train(args)

                    res.append([acc, auc, f1, micro_f1, macro_f1])

                # calculate the avg of each times
                res = np.array(res)
                avg = res.mean(axis=0)
                std = res.std(axis=0)
                res_str = f"Stage {args.period}: acc {avg[0]:.3f}+{std[0]:.3f}; auc {avg[1]:.3f}+{std[1]:.3f}; f1 {avg[2]:.3f}+{std[2]:.3f}; micro_f1 {avg[3]:.3f}+{std[3]:.3f}; macro_f1 {avg[4]:.3f}+{std[4]:.3f}\n"

                if best["auc"] + best["f1"] < avg[1] + avg[2]:
                    best["mask_ratio"] = mask_ratio
                    best["alpha"] = alpha
                    best["beta"] = beta
                    best["tau"] = args.tau
                    best["predictor"] = args.predictor
                    best["feature_dim"] = args.feature_dim
                    best["acc"] = avg[0]
                    best["auc"] = avg[1]
                    best["f1"] = avg[2]
                    best["micro_f1"] = avg[3]
                    best["macro_f1"] = avg[4]
                    best["res"] = res_str

    return best

if __name__ == "__main__":

    # load period data
    period = np.load(f"./data/{args.dataset}/{args.dataset}_period.npy", allow_pickle=True)

    for period_name in period:

        args.period = period_name

        best = search(args)

        print(best)

        print(best.get("auc"))

        break

# best:
# 40: mask_ratio: 0.1, alpha: 0.2, beta: 0.01, tau: 0.05, predictor: 2, feature_dim: 32
