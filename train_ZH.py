import random
import os
import numpy as np
import pandas as pd
import torch
import time
import argparse
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, calc_missing_rate
from pypots.nn.functional import calc_mae, calc_mse
import muyi.gpu as mug
import muyi.utils as muu

from utils import parse_args, set_seed, get_data, get_train_val_test_data, get_model

args = parse_args()

set_seed(args.seed)

train_start_time = time.time()

data, n_features = get_data(args) # X is missed values, X_ori is original complete values

train_set, val_set, test_set, test_X_ori, indicating_mask = get_train_val_test_data(data)

model = get_model(args, n_features=n_features)

model.fit(train_set, val_set)  # 在数据集上训练模型

train_end_time = time.time()
train_time = train_end_time-train_start_time

gpu_memory_info = mug.get_gpu_memory_usage()
gpu_usage = gpu_memory_info[0][2]



imputation = model.impute(
    test_set
).squeeze()  # 对测试集中原始缺失和人为缺失的值进行填补

mae = calc_mae(
    imputation, np.nan_to_num(test_X_ori), indicating_mask
)  # 在人为添加的缺失位置上计算 MAE（对比填补结果与真实值）
mse = calc_mse(
    imputation, np.nan_to_num(test_X_ori), indicating_mask
)  # 在人为添加的缺失位置上计算 MAE（对比填补结果与真实值）
muu.color_print(
    f"seed: {args.seed}, MAE: {mae:.4f}, MSE: {mse:.4f}", bg_color="bg_green"
)  # 打印 MAE 和 MSE 结果

if args.save_emb:
    all_info = model.get_all_info(data_set=train_set)  # 获取模型的所有信息
    enc_out = all_info["enc_out"]  # 获取编码器输出

    muu.color_print(f"{enc_out.shape=}", bg_color="bg_yellow")  # 打印 enc_out 的形状
    data_save = enc_out

    file_idx = 1
    data_type = "Embedding"
    path = f"./emb/train{args.epochs}/{args.model}_{data_type}_{args.dataset}_{args.missing_rate}_{args.epochs}.pt"

    while os.path.exists(path):
        print(f"File {path} already exists.")
        file_idx += 1
        path = f"./emb/{args.model}_{data_type}_{args.dataset}_{args.missing_rate}_v{file_idx}.pt"

    torch.save(data_save, path)

if args.save_res:
    if args.csv_path == "":
        muu.color_print("Please provide a path to save the results.", bg_color="bg_red")
    else:
        res = {
            "Setting": [
                f"{args.dataset}_missing_{args.missing_rate}_{args.pattern}_training_{args.epochs}_{args.model}_{args.seed}"
            ],
            "Model": [args.model],
            "Pattern": [args.pattern],
            "Missing_Rate": [args.missing_rate],
            "Epochs": [args.epochs],
            "Seed": [args.seed],
            "Embed_Size": [args.d_model],
            "N_Steps": [args.n_steps],
            "Align_Type": [args.align_type],
            "MAE": [format(mae, ".4f")],
            "MSE": [format(mse, ".4f")],
            "GPU_Usage": [gpu_usage],
            "Train_Time": [f"{train_time:.2f}s"],
        }

        df = pd.DataFrame(res)
        csv_file = args.csv_path

        try:
            # 如果文件存在，不需要写入列名（header）
            df_existing = pd.read_csv(csv_file)
            df.to_csv(csv_file, mode="a", index=False, header=False)
        except FileNotFoundError:
            # 如果文件不存在，写入列名（header）
            df.to_csv(csv_file, mode="w", index=False)

        print("Results have been appended to:", csv_file)
