import argparse
import random
import os
import numpy as np
import torch


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument("--dataset", type=str, default="ETTh1", choices=["physionet2012", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity_load_diagrams", "beijing_air_quality", "pems_traffic", "electricity", "traffic", "weather", "illness", "exchange_rate", "pems_bay", "metr_la",], help="dataset name",)
    parser.add_argument("--pattern", type=str, default="block", choices=["point", "subseq", "block",], help="missing pattern",)
    parser.add_argument("--missing_rate", type=float, default=0.1, help="missing rate",)
    parser.add_argument("--n_steps", type=int, default=96, help="number of time steps",)

    # training settings
    parser.add_argument("--epochs", type=int, default=30, help="training epochs",)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size",)
    parser.add_argument("--seed", type=int, default=1, help="model seed",)
    parser.add_argument("--save_emb", type=int, default=0, help="save the embedding of input",)
    parser.add_argument("--save_res", type=int, default=0, help="save the final result",)
    parser.add_argument("--iter", type=int, default=1, help="iteration number",)
    parser.add_argument("--csv_path", type=str, default="", help="path to save results",)

    # Glocal-IB settings
    parser.add_argument("--loss_type", type=str, default="123", choices=["1", "2", "3", "12", "13", "23", "123"], help="loss type of my model",)
    parser.add_argument("--mse_weight", type=float, default=1, help="mse loss weight",)
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="kl loss weight",)
    parser.add_argument("--align_weight", type=float, default=1, help="alignment loss weight",)
    parser.add_argument("--align_type", type=str, default="contras_2", choices=["contras_1", "contras_2", "FM_align"], help="alignment loss type",)

    # model settings
    parser.add_argument("--model", type=str, default="tcn_my", choices=["SAITS", "Transformer", "DLinear", "TimesNet", "FreTS", "PatchTST", "SCINet", "iTransformer", "CSDI", "GPVAE", "tcn", "TimeMixer", "USGAN","saits_my", "timesnet_my", "gpvae_my", "dlinear_my", "tcn_my",], help="model name",)
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers",)
    parser.add_argument("--d_model", type=int, default=256, help="embedding size",)
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads",)
    parser.add_argument("--d_k", type=int, default=64, help="dimension of key",)
    parser.add_argument("--d_v", type=int, default=64, help="dimension of value",)
    parser.add_argument("--d_ffn", type=int, default=128, help="dimension of feedforward network",)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate",)
    parser.add_argument("--moving_avg_window_size", type=int, default=3, help="moving average window size",)
    parser.add_argument("--individual", type=int, default=0, help="individual of DLinear",)
    parser.add_argument("--n_levels", type=int, default=2, help="number of levels of TCN",)
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size of TCN",)
    parser.add_argument("--top_k_timesnet", type=int, default=3, help="top k of TimesNet",)
    parser.add_argument("--n_kernels", type=int, default=5, help="number of kernels of TimesNet",)
    parser.add_argument("--patch_size", type=int, default=16, help="patch size of PatchTST",)
    parser.add_argument("--patch_stride", type=int, default=16, help="patch stride of PatchTST",)
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout rate of PatchTST",)
    parser.add_argument("--n_stacks", type=int, default=2, help="number of stacks of SCINet",)
    parser.add_argument("--n_groups", type=int, default=1, help="number of groups of SCINet",)
    parser.add_argument("--n_decoder_layers", type=int, default=2, help="number of decoder layers of SCINet",)
    parser.add_argument("--n_channels", type=int, default=64, help="number of channels of CSDI",)
    parser.add_argument("--d_time_embedding", type=int, default=32, help="dimension of time embedding of CSDI",)
    parser.add_argument("--d_feature_embedding", type=int, default=32, help="dimension of feature embedding of CSDI",)
    parser.add_argument("--d_diffusion_embedding", type=int, default=32, help="dimension of diffusion embedding of CSDI",)
    parser.add_argument("--n_diffusion_steps", type=int, default=50, help="number of diffusion steps of CSDI",)
    parser.add_argument("--top_k_time_mixer", type=int, default=100, help="top k of TimeMixer",)
    parser.add_argument("--prior_type", type=str, default="norm", help="prior type",)
    
    

    args = parser.parse_args(args)

    return args

def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
  