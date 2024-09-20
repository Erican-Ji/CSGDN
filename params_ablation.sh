#!/bin/sh
mask_values=(0 0.2 0.4 0.6 0.8)
alpha_valus=(0.2 0.4 0.6 0.8 1.0)
beta_values=(1e-1 1e-2 1e-3 1e-4 1e-5)
tau_values=(0.05 0.1 0.2 0.4 0.8)
predictor_values=("1" "2" "3" "4")
feature_dim_values=(8 16 32 64 128)

export CUDA_VISIBLE_DEVICES=0
for mask in ${mask_values[@]}; do
    nohup python -u train.py --dataset cotton --mask $mask > results/cotton/CSGDN/params_ablation/mask_${mask}.log 2>&1 &
done
for alpha in ${alpha_values[@]}; do
    nohup python -u train.py --dataset cotton --alpha $alpha > results/cotton/CSGDN/params_ablation/alpha_${alpha}.log 2>&1 &
done

export CUDA_VISIBLE_DEVICES=1
for beta in ${beta_values[@]}; do
    nohup python -u train.py --dataset cotton --beta $beta > results/cotton/CSGDN/params_ablation/beta_${beta}.log 2>&1 &
done
for tau in ${tau_values[@]}; do
    nohup python -u train.py --dataset cotton --tau $tau > results/cotton/CSGDN/params_ablation/tau_${tau}.log 2>&1 &
done

export CUDA_VISIBLE_DEVICES=2
for predictor in ${predictor_values[@]}; do
    nohup python -u train.py --dataset cotton --predictor $predictor > results/cotton/CSGDN/params_ablation/predictor_${predictor}.log 2>&1 &
done
for feature_dim in ${feature_dim_values[@]}; do
    nohup python -u train.py --dataset cotton --feature_dim $feature_dim > results/cotton/CSGDN/params_ablation/feature_dim_${feature_dim}.log 2>&1 &
done