import torch
import torch.nn as nn

from .basic_loss import LossCalculator, calc_mae, calc_mse, calc_rmse

class PredictionLoss(nn.Module):
    def __init__(
        self,
        ORT_weight,  # ORT (Observed Reconstruction Task)
        MIT_weight,  # MIT (Masked Imputation Task)
        loss_calc_func,
    ):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_calc_func = LossCalculator(loss_calc_func)

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


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        loss_calc_func = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.loss_calc_func = loss_calc_func

    def forward(self, X_obs_p, X_ori_z):
        # calculate loss for the observed reconstruction task (ORT)
        X_obs_p = nn.functional.normalize(X_obs_p, dim=-1)
        X_ori_z = nn.functional.normalize(X_ori_z, dim=-1)
        X_ori_z = X_ori_z.detach()

        logit_pos_neg = torch.matmul(X_obs_p, X_ori_z.transpose(-1, -2))
        label_pos_neg = (
            torch.arange(X_obs_p.shape[1])
            .to(X_obs_p.device)
            .repeat(logit_pos_neg.shape[0], 1)
        )

        contrastive_loss = self.loss_calc_func(logit_pos_neg, label_pos_neg)

        # calculate the loss to back propagate for model updating
        return contrastive_loss


class CosAlignLoss(nn.Module):
    def __init__(
        self,
        loss_calc_func = nn.CosineSimilarity(eps=1e-8, dim=1),
    ):
        super().__init__()
        self.loss_calc_func = loss_calc_func

    def forward(self, X_obs_p, X_ori_z):
        # calculate loss for the observed reconstruction task (ORT)
        cos_align_loss = (
            1
            - (
                self.loss_calc_func(
                    X_obs_p,
                    X_ori_z.detach(),
                ).mean()
            )
        )

        # calculate the loss to back propagate for model updating
        return cos_align_loss
