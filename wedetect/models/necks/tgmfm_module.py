# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class TGMFM(nn.Module):
    """Text-Guided Multi-scale Feature Modulation.

    A lightweight, non-fusion feature modulation module that uses text
    semantics to generate channel-wise modulation weights for visual features.
    Text features only produce scalar channel weights (no pixel-level
    interaction), preserving the retrieval paradigm.

    Args:
        text_dim: Dimension of text embeddings (e.g., 768 for XLM-RoBERTa).
        feat_channels: List of visual feature channel numbers at each stage.
        hidden_ratio: Ratio to compute hidden dim from text_dim. Default 0.5.
        use_residual: Whether to use residual modulation (F_out = F_vis * w + F_vis).
                      If False, F_out = F_vis * w.
    """

    def __init__(self,
                 text_dim: int = 768,
                 feat_channels: list = [256, 128, 256, 512],
                 hidden_ratio: float = 0.5,
                 use_residual: bool = True) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.feat_channels = feat_channels
        self.use_residual = use_residual
        hidden_dim = int(text_dim * hidden_ratio)

        # One lightweight MLP per stage: text_dim -> hidden_dim -> feat_channels[i]
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, ch),
            ) for ch in feat_channels
        ])

    def forward(self, visual_feats: list, text_feats: torch.Tensor) -> list:
        """Apply text-guided channel modulation to multi-scale visual features.

        Args:
            visual_feats: List of visual feature tensors, each (B, C_i, H_i, W_i).
            text_feats: Text feature tensor (B, K, D) from text encoder.

        Returns:
            List of modulated visual features, same shapes as input.
        """
        if text_feats is None:
            return visual_feats

        # Aggregate text features: mean over category dimension K -> (B, D)
        txt_avg = text_feats.mean(dim=1)  # (B, D)

        out_feats = []
        for feat, mlp in zip(visual_feats, self.mlps):
            # Generate channel weights: (B, D) -> (B, C)
            w = mlp(txt_avg)  # (B, C)
            w = torch.sigmoid(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

            if self.use_residual:
                feat = feat * w + feat  # residual modulation
            else:
                feat = feat * w
            out_feats.append(feat)

        return out_feats

    def forward_single(self, feat: torch.Tensor, text_feats: torch.Tensor,
                       stage_idx: int) -> torch.Tensor:
        """Apply modulation to a single stage feature.

        Args:
            feat: Visual feature tensor (B, C, H, W).
            text_feats: Text feature tensor (B, K, D).
            stage_idx: Index of the stage MLP to use.

        Returns:
            Modulated feature tensor, same shape as input.
        """
        if text_feats is None:
            return feat

        txt_avg = text_feats.mean(dim=1)  # (B, D)
        w = self.mlps[stage_idx](txt_avg)  # (B, C)
        w = torch.sigmoid(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        if self.use_residual:
            return feat * w + feat
        else:
            return feat * w
