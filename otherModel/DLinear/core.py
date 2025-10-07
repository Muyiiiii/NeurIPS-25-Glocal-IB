"""
The core wrapper assembles the submodules of DLinear imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

from .backbone import BackboneDLinear

from pypots.nn.modules import ModelCore
from pypots.nn.modules.autoformer import SeriesDecompositionBlock
from pypots.nn.modules.loss import Criterion
from pypots.nn.modules.saits import SaitsLoss, SaitsEmbedding

import subprocess
import os
from transformers import AutoModelForCausalLM

from ..loss import (
    MyContrastiveLoss_v1,
    MyContrastiveLoss_v2,
    MyAlignmentLoss,
)

import muyi.utils as muu

class _DLinear(ModelCore):
    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_steps: int,
        n_features: int,
        moving_avg_window_size: int,
        individual: bool,
        d_model: Optional[int],
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.align_type = align_type

        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual

        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.backbone = BackboneDLinear(n_steps, n_features, individual, d_model)

        self.contrastive_projection = nn.Sequential(
            nn.Linear(n_features, n_features),
            # nn.BatchNorm1d(n_steps),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
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
                nn.Linear(n_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_features),
            )
        else:
            self.foundation_model = None
            self.alignment_projection = None

        self.contrastive_loss_v1 = MyContrastiveLoss_v1()
        self.contrastive_loss_v2 = MyContrastiveLoss_v2()
        self.alignment_loss = MyAlignmentLoss()

        if not individual:
            self.seasonal_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.trend_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)

        # apply SAITS loss function to DLinear on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

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

        # input preprocessing and embedding for DLinear
        seasonal_init, trend_init = self.series_decomp(X)

        if not self.individual:
            # WDU: the original DLinear paper isn't proposed for imputation task. Hence the model doesn't take
            # the missing mask into account, which means, in the process, the model doesn't know which part of
            # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
            # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
            # the output layers to project the seasonal and trend from the hidden space to the original space.
            # But this is only for the non-individual mode.
            seasonal_init = self.seasonal_saits_embedding(seasonal_init, missing_mask)
            trend_init = self.trend_saits_embedding(trend_init, missing_mask)

        seasonal_output, trend_output = self.backbone(seasonal_init, trend_init)

        if not self.individual:
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)

        reconstruction = seasonal_output + trend_output

        X_obs_z_contras = reconstruction
        X_obs_p_contras = self.contrastive_projection(X_obs_z_contras)
        if self.training:
            seasonal_init_ori, trend_init_ori = self.series_decomp(X_ori)
            if not self.individual:
                seasonal_init_ori = self.seasonal_saits_embedding(seasonal_init_ori, torch.ones_like(missing_mask))
                trend_init_ori = self.trend_saits_embedding(trend_init_ori, torch.ones_like(missing_mask))
            seasonal_output_ori, trend_output_ori = self.backbone(seasonal_init_ori, trend_init_ori)
            if not self.individual:
                seasonal_output_ori = self.linear_seasonal_output(seasonal_output_ori)
                trend_output_ori = self.linear_trend_output(trend_output_ori)
            reconstruction_ori = seasonal_output_ori + trend_output_ori
            X_ori_z_contras = reconstruction_ori
            X_ori_p_contras = self.contrastive_projection(X_ori_z_contras)
        else: 
            X_ori_z_contras = None
            X_ori_p_contras = None
        if self.align_type == "FM_align":
            X_ori_align = self.alignment_projection(X_obs_z_contras)  # [B, N, T]
        else:
            X_ori_align = None

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
            "enc_out": X_obs_z_contras,
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
