"""Mamba-BAN model for DTI prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.nn.utils.parametrizations import weight_norm

try:
    from ban import BANLayer
except ImportError:
    from .ban import BANLayer


class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, kernel_sizes=(3, 7, 15), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        out = 0
        for conv in self.convs:
            out = out + conv(x)
        out = out.transpose(1, 2)
        out = self.dropout(self.norm(self.activation(out)))
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class MambaBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        for layer in self.layers:
            residual = x
            x = self.dropout(layer(x))
            x = x + residual
            if mask is not None:
                x = x * mask.unsqueeze(-1)

        x = self.norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class ParallelFusionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_mamba_layers=2,
        dropout=0.1,
        cnn_kernel_sizes=(3, 7, 15),
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.cnn_branch = MultiScaleCNN(
            input_dim,
            hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
        )
        self.mamba_branch = MambaBranch(input_dim, hidden_dim, num_mamba_layers, dropout=dropout)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 2), nn.Softmax(dim=-1))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        x = self.input_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        cnn_feat = self.cnn_branch(x, mask)
        mamba_feat = self.mamba_branch(x, mask)
        concat_feat = torch.cat([cnn_feat, mamba_feat], dim=-1)
        fusion_weights = self.fusion_gate(concat_feat)
        fused = fusion_weights[..., 0:1] * cnn_feat + fusion_weights[..., 1:2] * mamba_feat
        out = self.output_proj(concat_feat) + fused
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out, fusion_weights


class CEClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, interaction):
        interaction = torch.nan_to_num(interaction, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = self.net(interaction)
        prob = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "prob": prob,
        }


class CNNAttentionDTI(nn.Module):
    def __init__(
        self,
        drug_dim=768,
        protein_dim=1280,
        cnn_dim=256,
        mamba_layers=2,
        dropout=0.3,
        ban_glimpses=4,
        ban_k=3,
        drug_kernel_sizes=(3, 5, 7),
        protein_kernel_sizes=(3, 7, 15),
    ):
        super().__init__()
        self.drug_encoder = ParallelFusionEncoder(
            drug_dim,
            cnn_dim,
            mamba_layers,
            dropout,
            cnn_kernel_sizes=drug_kernel_sizes,
        )
        self.prot_encoder = ParallelFusionEncoder(
            protein_dim,
            cnn_dim,
            mamba_layers,
            dropout,
            cnn_kernel_sizes=protein_kernel_sizes,
        )
        self.ban = weight_norm(
            BANLayer(
                v_dim=cnn_dim,
                q_dim=cnn_dim,
                h_dim=cnn_dim,
                h_out=ban_glimpses,
                act="ReLU",
                dropout=dropout,
                k=ban_k,
            ),
            name="h_mat",
            dim=None,
        )
        self.drug_ban_norm = nn.LayerNorm(cnn_dim)
        self.prot_ban_norm = nn.LayerNorm(cnn_dim)
        self.interaction_norm = nn.LayerNorm(cnn_dim)
        self.classifier = CEClassifier(cnn_dim, cnn_dim, 2, dropout)

    def forward(self, drug_feat, prot_feat, drug_mask=None, prot_mask=None, return_attention=False):
        if drug_mask is not None:
            drug_feat = drug_feat * drug_mask.unsqueeze(-1)
        if prot_mask is not None:
            prot_feat = prot_feat * prot_mask.unsqueeze(-1)

        drug_encoded, drug_fusion_weights = self.drug_encoder(drug_feat, drug_mask)
        prot_encoded, prot_fusion_weights = self.prot_encoder(prot_feat, prot_mask)
        drug_encoded = F.normalize(self.drug_ban_norm(drug_encoded), dim=-1)
        prot_encoded = F.normalize(self.prot_ban_norm(prot_encoded), dim=-1)
        if drug_mask is not None:
            drug_encoded = drug_encoded * drug_mask.unsqueeze(-1)
        if prot_mask is not None:
            prot_encoded = prot_encoded * prot_mask.unsqueeze(-1)

        interaction, ban_attn = self.ban(
            drug_encoded,
            prot_encoded,
            softmax=True,
            v_mask=drug_mask,
            q_mask=prot_mask,
        )
        interaction = torch.nan_to_num(interaction, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        interaction = self.interaction_norm(interaction)

        output = self.classifier(interaction)
        output.update(
            {
                "fused_repr": interaction,
                "drug_branch_weights": drug_fusion_weights,
                "prot_branch_weights": prot_fusion_weights,
            }
        )
        if return_attention:
            output["ban_attn"] = ban_attn
        return output

    def predict_prob(self, drug_feat, prot_feat, drug_mask=None, prot_mask=None):
        output = self.forward(drug_feat, prot_feat, drug_mask, prot_mask)
        return output["prob"][:, 1]
