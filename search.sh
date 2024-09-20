#!/bin/sh

# Define the parameter settings
tau_values=(0.05 0.1 0.2)
predictor_values=("1" "2" "3" "4")
feature_dim_values=(8 16 32 64 128)

Loop through each combination of parameter settings
for tau in "${tau_values[@]}"; do
    if [ "$tau" == "0.05" ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$tau" == "0.1" ]; then
        export CUDA_VISIBLE_DEVICES=1
    elif [ "$tau" == "0.2" ]; then
        export CUDA_VISIBLE_DEVICES=2
    fi
    for predictor in "${predictor_values[@]}"; do
        for feature_dim in "${feature_dim_values[@]}"; do
            nohup python -u search.py --dataset cotton --tau "$tau" --predictor "$predictor" --feature_dim "$feature_dim" > ./results/cotton/CSGDN/tmp_search/cotton_tau_"$tau"_predictor_"$predictor"_feature_dim_"$feature_dim".txt 2>&1 &
        done
    done
done
