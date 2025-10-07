import numpy as np
from pygrinder import mcar, calc_missing_rate

import sys
sys.path.append("../")

from data import (
    preprocess_electricity,
    preprocess_traffic,
    preprocess_weather,
    preprocess_illness,
    preprocess_exchange_rate,
    preprocess_pems_bay,
    preprocess_metr_la,
)

from benchpots.datasets import (
    preprocess_beijing_air_quality,
    preprocess_physionet2012,
    preprocess_ett,
    preprocess_electricity_load_diagrams,
    preprocess_pems_traffic,
)

def get_data(args):
    if args.dataset == "physionet2012":
        data = preprocess_physionet2012(subset="all", rate=args.missing_rate, pattern=args.pattern)
    elif args.dataset == "ETTh1":
        data = preprocess_ett(subset="ETTh1", rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "ETTh2":
        data = preprocess_ett(subset="ETTh2", rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "ETTm1":
        data = preprocess_ett(subset="ETTm1", rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "ETTm2":
        data = preprocess_ett(subset="ETTm2", rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "electricity_load_diagrams":
        data = preprocess_electricity_load_diagrams(
            rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern
        )
    elif args.dataset == "beijing_air_quality":
        data = preprocess_beijing_air_quality(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "pems_traffic":
        data = preprocess_pems_traffic(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "electricity":
        data = preprocess_electricity(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "traffic":
        data = preprocess_traffic(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "weather":
        data = preprocess_weather(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "illness":
        data = preprocess_illness(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "exchange_rate":
        data = preprocess_exchange_rate(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "pems_bay":
        data = preprocess_pems_bay(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    elif args.dataset == "metr_la":
        data = preprocess_metr_la(rate=args.missing_rate, n_steps=args.n_steps, pattern=args.pattern)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")          

    n_features = data["train_X"].shape[2]

    return data, n_features

def get_train_val_test_data(data):
    train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]  # [B, T, N]
    print(f"train shape: {train_X.shape}")  # (n_samples, n_steps, n_features) [B, T, N]
    print(f"val shape: {val_X.shape}")
    print(f"train_X missing rate: {calc_missing_rate(train_X):.1%}")

    train_set = {"X": train_X}  # 训练集只需包含不完整时间序列
    val_set = {
        "X": val_X,
        "X_ori": data["val_X_ori"],  # val_X_ori is original complete values
    }
    test_set = {"X": test_X}  # 测试集仅提供待填补的不完整时间序列
    test_X_ori = data["test_X_ori"]  # test_X_ori 包含用于最终评估的真实值
    indicating_mask = np.isnan(test_X) ^ np.isnan(
        test_X_ori
    )  # indicating_mask is the mask of the missing values in the test set

    return train_set, val_set, test_set, test_X_ori, indicating_mask