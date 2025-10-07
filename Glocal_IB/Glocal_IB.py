import torch
import torch.nn as nn
import torch.nn.functional as F

from .align_loss import PredictionLoss, ContrastiveLoss, CosAlignLoss

ALIGN_LOSS_DICT = {
    "contrastive": ContrastiveLoss,
    "cos_align": CosAlignLoss,
}

class Glocal_IB(nn.Module):
    """
    A PEFT-like wrapper for time series imputation models.

    This class wraps a base imputation model to add a new objective during training:
    aligning the latent embeddings of masked and complete time series data.

    Args:
        base_model (nn.Module): The time series imputation model to be wrapped. 
                                It is expected that its forward pass returns a tuple:
                                (imputation_output, intermediate_embedding).
        embedding_dim (int): The dimension of the embedding.
        align_loss_type (str): The type of the alignment loss.
        align_model_type (str): The type of the alignment model.
        align_weight (float, optional): The weight to apply to the alignment loss. 
        foundation_embedding (torch.Tensor, optional): The embedding of the foundation model.
    """
    def __init__(self, base_model: nn.Module, embedding_dim: int, align_loss_type: str, align_model_type: str = "self", align_weight: float = 1.0, foundation_embedding: torch.Tensor = None):
        if align_model_type not in ["self", "foundation"]:
            raise ValueError("align_model_type must be one of ['self', 'foundation']")
        if align_loss_type not in ["contrastive", "cos_align"]:
            raise ValueError("align_loss_type must be one of ['contrastive', 'cos_align']")
        
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.align_loss_type = align_loss_type
        self.align_model_type = align_model_type
        self.align_loss_fn = ALIGN_LOSS_DICT[align_loss_type]()
        self.align_weight = align_weight
        self.foundation_embedding = foundation_embedding
        
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, x_masked: torch.Tensor, x_complete: torch.Tensor = None, **kwargs):
        """
        Defines the computation performed at every call.

        - In training mode (`self.training` is True and `x_complete` is provided):
          It performs two forward passes to get embeddings for both masked and
          complete data, calculates the alignment loss, and returns both the
          imputation result and the alignment loss.
        - In evaluation mode (`self.training` is False or `x_complete` is None):
          It performs a single forward pass with the masked data and returns
          only the imputation result.

        Args:
            x_masked (torch.Tensor): The input tensor with missing values.
            x_complete (torch.Tensor, optional): The ground truth tensor without
                                                 missing values. Required for training.
                                                 Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base model's
                      forward method.

        Returns:
            - dict: During training, a dictionary {'output': tensor, 'alignment_loss': tensor}.
            - torch.Tensor: During evaluation, the imputed output tensor.
        """
        # Always get the primary output and embedding from the masked input
        imputation_output, emb_masked = self.base_model(x_masked, **kwargs)

        # If in training mode and the complete data is available, compute alignment loss
        if self.training and x_complete is not None:
            # Pass complete data through the model to get its embedding
            # We don't need the output, so we use '_'
            with torch.no_grad(): # Optional: if you don't want to compute gradients for the complete pass
                if self.align_model_type == "foundation":
                    emb_complete = self.foundation_embedding
                else:
                    _, emb_complete = self.base_model(x_complete, **kwargs)
            
            emb_masked = self.projection(emb_masked)

            # Calculate the specific alignment loss
            alignment_loss = self.align_loss_fn(emb_masked, emb_complete)
            
            return {
                'output': imputation_output,
                'alignment_loss': self.align_weight * alignment_loss
            }
        else:
            # In evaluation mode, just return the imputation result
            return imputation_output

    def __getattr__(self, name: str):
        """
        Forward attribute access to the base model.
        
        This makes it so that you can call any method of the base model directly
        on the wrapper. For example, `wrapper.get_config()` will call
        `wrapper.base_model.get_config()`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)