""" """

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from pypots.nn.functional import nonstationary_norm, nonstationary_denorm
from pypots.nn.modules import ModelCore
from pypots.nn.modules.loss import Criterion
from pypots.nn.modules.timesnet import BackboneTimesNet
from pypots.nn.modules.transformer.embedding import DataEmbedding

import subprocess
import os
from transformers import AutoModelForCausalLM

from ..loss import (
    MyContrastiveLoss_v1,
    MyContrastiveLoss_v2,
    MyAlignmentLoss,
)

import muyi.utils as muu


class _TimesNet(ModelCore):
    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_layers,
        n_steps,
        n_features,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        dropout,
        apply_nonstationary_norm,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.align_type = align_type
        self.n_features = n_features
        self.seq_len = n_steps
        self.n_layers = n_layers
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            n_max_steps=n_steps,
        )
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            0,  # n_pred_steps should be 0 for the imputation task
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(d_model, n_features)

        self.contrastive_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            # nn.BatchNorm1d(n_steps),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if self.align_type == "FM_align":
            muu.color_print(
                f"!!!!!!!!!! Using foundation model (Time-MoE 50M Frozen) !!!!!!!!!!"
            )
            # self.foundation_model = AutoModelForCausalLM.from_pretrained(
            #     "Maple728/TimeMoE-50M",
            #     device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            #     trust_remote_code=True,
            # )
            self.foundation_model = AutoModelForCausalLM.from_pretrained(
                "../Time-MoE/TimeMoE-50M",
                device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )

            self.foundation_model.requires_grad_(False)

            self.alignment_projection = nn.Sequential(
                nn.Linear(d_model, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_features),
            )
        else:
            self.foundation_model = None
            self.alignment_projection = None

        self.contrastive_loss_v1 = MyContrastiveLoss_v1()
        self.contrastive_loss_v2 = MyContrastiveLoss_v2()
        self.alignment_loss = MyAlignmentLoss()

    def forward(self, inputs: dict) -> dict:
        if self.training:
            X, X_ori, missing_mask = (
                inputs["X"],
                inputs["X_ori"],
                inputs["missing_mask"],
            )  # [B, T, N]
        else:
            X, missing_mask = (
                inputs["X"],
                inputs["missing_mask"],
            )  # [B, T, N]

        if self.align_type == "FM_align":
            X_foundation = self.foundation_model.generate(
                X.reshape(-1, self.n_features), max_new_tokens=self.n_features
            )[:, -self.n_features :].reshape(X.shape[0], -1, self.n_features)
        else:
            X_foundation = None

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # embedding
        input_X = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        enc_out = self.model(input_X)

        X_obs_z_contras = enc_out
        X_obs_p_contras = self.contrastive_projection(X_obs_z_contras)
        if self.training:
            X_ori_z_contras = self.model(self.enc_embedding(X_ori))
            X_ori_p_contras = self.contrastive_projection(X_ori_z_contras)
        else:
            X_ori_z_contras = None
            X_ori_p_contras = None
        if self.align_type == "FM_align":
            X_ori_align = self.alignment_projection(X_obs_z_contras)  # [B, N, T]
        else:
            X_ori_align = None

        # project back the original data space
        reconstruction = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "X_obs_z_contras": X_obs_z_contras,  # [B, T, D]
            "X_obs_p_contras": X_obs_p_contras,  # [B, T, D]
            "X_ori_z_contras": X_ori_z_contras,  # [B, T, D]
            "X_ori_p_contras": X_ori_p_contras,  # [B, T, D]
            "X_foundation": X_foundation,  # [B, N, T]
            "X_ori_align": X_ori_align,  # [B, N, T]
            "enc_out": enc_out,
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        reconstruction = results["reconstruction"]

        if (
            self.training
        ):  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            loss = self.training_loss(reconstruction, X, missing_mask)

            results["Contrastive_loss_v1"] = self.contrastive_loss_v1(results=results)
            results["Contrastive_loss_v2"] = self.contrastive_loss_v2(results=results)
            if self.align_type == "FM_align":
                results["Alignment_loss"] = self.alignment_loss(results=results)
            else:
                results["Alignment_loss"] = 0

            results["loss"] = 0.0
            if "1" in self.loss_type:
                results["loss"] += loss * self.loss_weight[0]
            if "3" in self.loss_type:
                if self.align_type == "contras_1":
                    results["loss"] += (
                        results["Contrastive_loss_v1"] * self.loss_weight[2]
                    )
                elif self.align_type == "contras_2":
                    results["loss"] += (
                        results["Contrastive_loss_v2"] * self.loss_weight[2]
                    )
                elif self.align_type == "FM_align":
                    results["loss"] += results["Alignment_loss"] * self.loss_weight[2]
        else:  # if in the eval mode (the validation stage), return metric result from validation_metric
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            results["metric"] = self.validation_metric(
                reconstruction, X_ori, indicating_mask
            )

        return results
