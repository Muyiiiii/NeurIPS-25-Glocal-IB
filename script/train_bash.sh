#!/bin/bash

# datasets=("beijing_air_quality" "pems_traffic" "electricity" "traffic" "weather" "metr_la")
datasets=("ETTh1" )

# models=("SAITS" "Transformer" "DLinear" "tcn" "TimesNet" "FreTS" "PatchTST" "SCINet" "iTransformer" "CSDI" "GPVAE" "TimeMixer" "USGAN" "CSDI" "saits_my" "timesnet_my" "dlinear_my" "tcn_my")
models=("saits")


# missing_rates=(0.1 0.3 0.5 0.7 0.9)
missing_rates=(0.1)

# iters=("1" "2" "3" "4" "5")
iters=("1")

# patterns=("point" "subseq" "block")
patterns=("point")

# set training epochs and learning rate
train_epoch=5
learning_rate=0.001

csv_path="./results/results.csv"

# outer loop
for dataset in "${datasets[@]}"; do
    # inner loop
    for model in "${models[@]}"; do
        for missing_rate in "${missing_rates[@]}"; do
            # create a log file
            log_file="./log/${dataset}__training_${train_epoch}_${model}_${missing_rate}.log"
            # clear the old log file or create a new one
            > "$log_file"
            for pattern in "${patterns[@]}"; do
                echo "Starting model $model training on $dataset with missing rate $missing_rate with $pattern missing pattern, train_epoch $train_epoch" | tee -a $log_file

                python train_EN.py --dataset $dataset --model $model --missing_rate $missing_rate --pattern $pattern --epochs $train_epoch --save_emb 0 --save_res 1 --csv_path $csv_path 2>&1 | tee -a $log_file

                echo "Finished model $model training on $dataset with missing rate $missing_rate with $pattern missing pattern, train_epoch $train_epoch" | tee -a $log_file
            done
        done
    done
done