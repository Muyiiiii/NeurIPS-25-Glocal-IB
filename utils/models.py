from pypots.imputation import (
    SAITS,
    Transformer,
    DLinear,
    TCN,
    TimesNet,
    FreTS,
    PatchTST,
    SCINet,
    iTransformer,
    CSDI,
    GPVAE,
    TimeMixer,
    USGAN,
)

import sys
sys.path.append("../")
from otherModel import SAITS_MY, TimesNet_MY, GPVAE_MY, DLinear_MY, TCN_MY

def get_model(args, n_features):
    if args.model == "SAITS":
        return SAITS(n_steps=args.n_steps, 
                    n_features=n_features,
                    n_layers=args.n_layers,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    d_k=args.d_k,
                    d_v=args.d_v,
                    d_ffn=args.d_ffn,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    epochs=args.epochs)
    elif args.model == "Transformer":
        return Transformer(n_steps=args.n_steps, 
                           n_features=n_features,
                           n_layers=args.n_layers,
                           d_model=args.d_model,
                           n_heads=args.n_heads,
                           d_k=args.d_k,
                           d_v=args.d_v,
                           d_ffn=args.d_ffn,
                           dropout=args.dropout,
                           batch_size=args.batch_size,
                           epochs=args.epochs)
    elif args.model == "DLinear":
        return DLinear(n_steps=args.n_steps, 
                       n_features=n_features,
                       moving_avg_window_size=args.moving_avg_window_size,
                       individual=args.individual,
                       d_model=args.d_model,
                       batch_size=args.batch_size,
                       epochs=args.epochs)
    elif args.model == "tcn":
        return TCN(n_steps=args.n_steps, 
                   n_features=n_features,
                   n_levels=args.n_levels,
                   d_hidden=args.d_model,
                   kernel_size=args.kernel_size,
                   dropout=args.dropout,
                   batch_size=args.batch_size,
                   epochs=args.epochs)
    elif args.model == "TimesNet":
        return TimesNet(n_steps=args.n_steps, 
                        n_features=n_features,
                        n_layers=args.n_layers,
                        top_k=args.top_k_timesnet,
                        d_model=args.d_model,
                        d_ffn=args.d_ffn,
                        n_kernels=args.n_kernels,
                        dropout=args.dropout,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    elif args.model == "FreTS":
        return FreTS(n_steps=args.n_steps, 
                     n_features=n_features,
                     embed_size=args.d_model,
                     hidden_size=args.d_model,
                     batch_size=args.batch_size,
                     epochs=args.epochs)
    elif args.model == "PatchTST":
        return PatchTST(n_steps=args.n_steps, 
                        n_features=n_features,
                        patch_size=args.patch_size,
                        patch_stride=args.patch_stride,
                        n_layers=args.n_layers,
                        d_model=args.d_model,
                        n_heads=args.n_heads,
                        d_k=args.d_k,
                        d_v=args.d_v,
                        d_ffn=args.d_ffn,
                        dropout=args.dropout,
                        attn_dropout=args.attn_dropout,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    elif args.model == "SCINet":
        return SCINet(n_steps=args.n_steps, 
                      n_features=n_features,
                      n_stacks=args.n_stacks,
                      n_levels=args.n_levels,
                      n_groups=args.n_groups,
                      n_decoder_layers=args.n_decoder_layers,
                      d_hidden=args.d_model,
                      batch_size=args.batch_size,
                      epochs=args.epochs)
    elif args.model == "iTransformer":
        return iTransformer(n_steps=args.n_steps, 
                            n_features=n_features,
                            n_layers=args.n_layers,
                            d_model=args.d_model,
                            n_heads=args.n_heads,
                            d_k=args.d_k,
                            d_v=args.d_v,
                            d_ffn=args.d_ffn,
                            dropout=args.dropout,
                            batch_size=args.batch_size,
                            epochs=args.epochs)
    elif args.model == "CSDI":
        return CSDI(n_steps=args.n_steps, 
                    n_features=n_features,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    n_channels=args.n_channels,
                    d_time_embedding=args.d_time_embedding,
                    d_feature_embedding=args.d_feature_embedding,
                    d_diffusion_embedding=args.d_diffusion_embedding,
                    n_diffusion_steps=args.n_diffusion_steps,
                    batch_size=args.batch_size,
                    epochs=args.epochs)
    elif args.model == "GPVAE":
        return GPVAE(n_steps=args.n_steps, 
                    n_features=n_features,
                    latent_size=args.d_model,
                    window_size=args.n_steps,
                    batch_size=args.batch_size,
                    epochs=args.epochs)
    elif args.model == "TimeMixer":
        return TimeMixer(n_steps=args.n_steps, 
                        n_features=n_features,
                        n_layers=args.n_layers,
                        top_k=args.top_k_time_mixer,
                        d_model=args.d_model,
                        d_ffn=args.d_ffn,
                        dropout=args.dropout,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    elif args.model == "USGAN":
        return USGAN(n_steps=args.n_steps, 
                    n_features=n_features,
                    rnn_hidden_size=args.d_model,
                    batch_size=args.batch_size,
                    epochs=args.epochs)
    elif args.model == "saits_my":
        return SAITS_MY(loss_type=args.loss_type, 
                        loss_weight=[args.mse_weight, args.kl_weight, args.align_weight], 
                        align_type=args.align_type, 
                        n_steps=args.n_steps, 
                        n_features=n_features,
                        n_layers=args.n_layers,
                        d_model=args.d_model,
                        n_heads=args.n_heads,
                        d_k=args.d_k,
                        d_v=args.d_v,
                        d_ffn=args.d_ffn,
                        dropout=args.dropout,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    elif args.model == "timesnet_my":
        return TimesNet_MY(loss_type=args.loss_type, 
                           loss_weight=[args.mse_weight, args.kl_weight, args.align_weight], 
                           align_type=args.align_type, 
                           n_steps=args.n_steps, 
                           n_features=n_features,
                           n_layers=args.n_layers,
                           top_k=args.top_k_timesnet,
                           d_model=args.d_model,
                           d_ffn=args.d_ffn,
                           n_kernels=args.n_kernels,
                           dropout=args.dropout,
                           batch_size=args.batch_size,
                           epochs=args.epochs)
    elif args.model == "gpvae_my":
        return GPVAE_MY(loss_type=args.loss_type, 
                        loss_weight=[args.mse_weight, args.kl_weight, args.align_weight], 
                        align_type=args.align_type, 
                        n_steps=args.n_steps, 
                        n_features=n_features,
                        latent_size=args.d_model,
                        window_size=args.n_steps,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
    elif args.model == "dlinear_my":
        return DLinear_MY(loss_type=args.loss_type, 
                          loss_weight=[args.mse_weight, args.kl_weight, args.align_weight], 
                          align_type=args.align_type, 
                          n_steps=args.n_steps, 
                          n_features=n_features,
                          moving_avg_window_size=args.moving_avg_window_size,
                          individual=args.individual,
                          d_model=args.d_model,
                          batch_size=args.batch_size,
                          epochs=args.epochs)
    elif args.model == "tcn_my":
        return TCN_MY(loss_type=args.loss_type, 
                      loss_weight=[args.mse_weight, args.kl_weight, args.align_weight], 
                      align_type=args.align_type, 
                      n_steps=args.n_steps, 
                      n_features=n_features,
                      n_levels=args.n_levels,
                      d_hidden=args.d_model,
                      kernel_size=args.kernel_size,
                      dropout=args.dropout,
                      batch_size=args.batch_size,
                      epochs=args.epochs)
    else:
        raise ValueError(f"Model {args.model} not found")