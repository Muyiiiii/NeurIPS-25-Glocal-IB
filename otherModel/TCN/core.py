"""
The core wrapper assembles the submodules of TCN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from pypots.nn.modules import ModelCore
from pypots.nn.modules.loss import Criterion
from pypots.nn.modules.saits import SaitsLoss, SaitsEmbedding

from .backbone import BackboneTCN

import subprocess
import os
from transformers import AutoModelForCausalLM

from ..loss import (
    MyContrastiveLoss_v1,
    MyContrastiveLoss_v2,
    MyAlignmentLoss,
)

import muyi.utils as muu


class _TCN(ModelCore):
    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_steps: int,
        n_features: int,
        n_levels: int,
        d_hidden: int,
        kernel_size: int,
        dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.align_type = align_type
        self.n_features = n_features

        self.n_steps = n_steps
        channel_sizes = [d_hidden] * n_levels

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
            dropout=dropout,
        )
        self.backbone = BackboneTCN(
            n_features,
            channel_sizes,
            kernel_size,
            dropout,
        )

        # for the imputation task, the output dim is the same as input dim
        self.output_projection = nn.Linear(channel_sizes[-1], n_features)
        # apply SAITS loss function to TCN on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.contrastive_projection = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            # nn.BatchNorm1d(n_steps),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        if self.align_type == "FM_align":
            muu.color_print(
                f"!!!!!!!!!! Using foundation model (Time-MoE 50M Frozen) !!!!!!!!!!"
            )
            self.foundation_model = AutoModelForCausalLM.from_pretrained(
                "../Time-MoE/TimeMoE-50M",
                device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )

            self.foundation_model.requires_grad_(False)

            self.alignment_projection = nn.Sequential(
                nn.Linear(d_hidden, n_features),
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

        # WDU: the original TCN paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out = enc_out.permute(0, 2, 1)

        # TCN encoder processing
        enc_out = self.backbone(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        # project back the original data space
        reconstruction = self.output_projection(enc_out)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        X_obs_z_contras = enc_out
        X_obs_p_contras = self.contrastive_projection(X_obs_z_contras)
        if self.training:
            X_ori_z_contras = self.backbone(self.saits_embedding(X_ori, torch.ones_like(missing_mask)).permute(0, 2, 1)).permute(0, 2, 1)
            X_ori_p_contras = self.contrastive_projection(X_ori_z_contras)
        else:
            X_ori_z_contras = None
            X_ori_p_contras = None
        if self.align_type == "FM_align":
            X_ori_align = self.alignment_projection(X_obs_z_contras)  # [B, N, T]
        else:
            X_ori_align = None

        results = {
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
            "enc_out": enc_out,
            "X_obs_z_contras": X_obs_z_contras,
            "X_obs_p_contras": X_obs_p_contras,
            "X_ori_z_contras": X_ori_z_contras,
            "X_ori_p_contras": X_ori_p_contras,
            "X_ori_align": X_ori_align,
            "X_foundation": X_foundation,
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)

        X_ori, indicating_mask, missing_mask = inputs["X_ori"], inputs["indicating_mask"], inputs["missing_mask"]
        reconstruction = results["reconstruction"]

        if self.training:  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

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
            results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
