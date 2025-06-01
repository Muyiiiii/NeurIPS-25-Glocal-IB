"""
The core wrapper assembles the submodules of SAITS imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from pypots.nn.modules import ModelCore
from pypots.nn.modules.loss import Criterion

from .backbone import BackboneSAITS

import subprocess
import os
from transformers import AutoModelForCausalLM

from ..loss import (
    MyContrastiveLoss_v1,
    MyContrastiveLoss_v2,
    MyAlignmentLoss,
)

import muyi.utils as muu


class _SAITS(ModelCore):
    def __init__(
        self,
        loss_type: str,
        loss_weight: list,
        align_type: str,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool,
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
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.encoder = BackboneSAITS(
            n_steps,
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )

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

    def forward(
        self,
        inputs: dict,
        diagonal_attention_mask: bool = True,
    ) -> dict:
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

        # determine the attention mask
        if (self.training and self.diagonal_attention_mask) or (
            (not self.training) and diagonal_attention_mask
        ):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        # SAITS processing
        (
            enc_output_1,
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(X, missing_mask, diagonal_attention_mask)

        X_obs_z_contras = enc_output_1
        X_obs_p_contras = self.contrastive_projection(X_obs_z_contras)
        if self.training:
            # only get the embedding of X_ori
            X_ori_z_contras = self.encoder(
                X_ori, torch.ones_like(missing_mask), diagonal_attention_mask
            )[0]
            X_ori_p_contras = self.contrastive_projection(X_ori_z_contras)
        else:
            X_ori_z_contras = None
            X_ori_p_contras = None
        if self.align_type == "FM_align":
            X_ori_align = self.alignment_projection(X_obs_z_contras)  # [B, N, T]
        else:
            X_ori_align = None

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3

        # ensemble the results as a dictionary for return
        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
            "X_tilde_1": X_tilde_1,
            "X_tilde_2": X_tilde_2,
            "X_tilde_3": X_tilde_3,
            "enc_out": enc_output_1,
            "X_obs_z_contras": X_obs_z_contras,  # [B, T, D]
            "X_obs_p_contras": X_obs_p_contras,  # [B, T, D]
            "X_ori_z_contras": X_ori_z_contras,  # [B, T, D]
            "X_ori_p_contras": X_ori_p_contras,  # [B, T, D]
            "X_foundation": X_foundation,  # [B, N, T]
            "X_ori_align": X_ori_align,  # [B, N, T]
        }

        return results

    def calc_criterion(self, inputs: dict) -> dict:
        results = self.forward(inputs)
        X_tilde_1, X_tilde_2, X_tilde_3 = (
            results["X_tilde_1"],
            results["X_tilde_2"],
            results["X_tilde_3"],
        )
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if (
            self.training
        ):  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]

            # calculate loss for the observed reconstruction task (ORT)
            # this calculation is more complicated that pypots.nn.modules.saits.SaitsLoss because
            # SAITS model structure has three parts of representation
            ORT_loss = 0
            ORT_loss += self.training_loss(X_tilde_1, X, missing_mask)
            ORT_loss += self.training_loss(X_tilde_2, X, missing_mask)
            ORT_loss += self.training_loss(X_tilde_3, X, missing_mask)
            ORT_loss /= 3
            ORT_loss = self.ORT_weight * ORT_loss

            # calculate loss for the masked imputation task (MIT)
            MIT_loss = self.MIT_weight * self.training_loss(
                X_tilde_3, X_ori, indicating_mask
            )
            # `loss` is always the item for backward propagating to update the model
            loss = ORT_loss + MIT_loss

            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss

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
                X_tilde_3, X_ori, indicating_mask
            )

        return results
