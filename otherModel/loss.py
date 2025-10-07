import torch
import torch.nn as nn

from pypots.nn.modules.loss import Criterion, MAE, MSE, RMSE, MRE

import muyi.utils as muu


class MyBasicLoss(nn.Module):
    def __init__(
        self,
        ORT_weight,  # ORT (Observed Reconstruction Task)
        MIT_weight,  # MIT (Masked Imputation Task)
        loss_calc_func: Criterion = MSE(),
    ):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_calc_func = loss_calc_func

    def forward(self, reconstruction, X_ori, missing_mask, indicating_mask):
        # calculate loss for the observed reconstruction task (ORT)
        ORT_loss = self.ORT_weight * self.loss_calc_func(
            reconstruction, X_ori, missing_mask
        )
        # calculate loss for the masked imputation task (MIT)
        MIT_loss = self.MIT_weight * self.loss_calc_func(
            reconstruction, X_ori, indicating_mask
        )
        # calculate the loss to back propagate for model updating
        loss = ORT_loss + MIT_loss
        return loss, ORT_loss, MIT_loss


class MyContrastiveLoss_v1(nn.Module):
    def __init__(
        self,
        loss_calc_func: Criterion = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.loss_calc_func = loss_calc_func

    def forward(self, results):
        # calculate loss for the observed reconstruction task (ORT)
        X_obs_z, X_obs_p, X_ori_z, X_ori_p = (
            results["X_obs_z_contras"],
            results["X_obs_p_contras"],
            results["X_ori_z_contras"],
            results["X_ori_p_contras"],
        )

        X_obs_p = nn.functional.normalize(X_obs_p, dim=-1)
        X_ori_p = nn.functional.normalize(X_ori_p, dim=-1)
        X_ori_p = X_ori_p.detach()

        logit_pos_neg = torch.matmul(X_obs_p, X_ori_p.transpose(-1, -2))
        label_pos_neg = (
            torch.arange(X_obs_p.shape[1])
            .to(X_obs_p.device)
            .repeat(logit_pos_neg.shape[0], 1)
        )

        ContrastiveLoss = self.loss_calc_func(logit_pos_neg, label_pos_neg)

        # calculate the loss to back propagate for model updating
        return ContrastiveLoss


class MyContrastiveLoss_v2(nn.Module):
    def __init__(
        self,
        loss_calc_func: Criterion = nn.CosineSimilarity(eps=1e-8, dim=1),
    ):
        super().__init__()
        self.loss_calc_func = loss_calc_func

    def forward(self, results):
        # calculate loss for the observed reconstruction task (ORT)
        ContrastiveLoss = (
            1
            - (
                self.loss_calc_func(
                    results["X_obs_p_contras"],
                    results["X_ori_z_contras"].detach(),
                    # results["X_obs_z_contras"],
                    # results["X_ori_z_contras"].detach(),
                ).mean()
                # + self.loss_calc_func(
                #     results["X_ori_p"], results["X_obs_z"].detach()
                # ).mean()
            )
            # * 0.5
        )

        # calculate the loss to back propagate for model updating
        return ContrastiveLoss


class MyAlignmentLoss(nn.Module):
    def __init__(
        self,
        loss_calc_func: Criterion = nn.CosineSimilarity(eps=1e-8, dim=1),
    ):
        super().__init__()
        self.loss_calc_func = loss_calc_func

    def forward(self, results):
        # calculate loss for the observed reconstruction task (ORT)
        AlignmentLoss = 1 - (
            self.loss_calc_func(
                results["X_foundation"].detach(), results["X_ori_align"]
            ).mean()
        )

        # calculate the loss to back propagate for model updating
        return AlignmentLoss
