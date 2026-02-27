"""Multi-modal fusion module for audio-visual features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module for combining audio and visual features."""
    
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        method: str = "late"
    ):
        """Initialize multi-modal fusion module.
        
        Args:
            audio_dim: Audio feature dimension.
            visual_dim: Visual feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of fusion layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            method: Fusion method ("early", "late", "attention").
        """
        super().__init__()
        
        self.method = method
        self.hidden_dim = hidden_dim
        
        if method == "early":
            # Early fusion: concatenate features before processing
            self.input_projection = nn.Linear(audio_dim + visual_dim, hidden_dim)
            self.fusion_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
        elif method == "late":
            # Late fusion: process modalities separately then combine
            self.audio_projection = nn.Linear(audio_dim, hidden_dim)
            self.visual_projection = nn.Linear(visual_dim, hidden_dim)
            
            self.audio_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
            self.visual_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
            # Cross-modal attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Final fusion
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
            
        elif method == "attention":
            # Attention-based fusion
            self.audio_projection = nn.Linear(audio_dim, hidden_dim)
            self.visual_projection = nn.Linear(visual_dim, hidden_dim)
            
            # Cross-modal attention
            self.audio_to_visual = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.visual_to_audio = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Fusion layers
            self.fusion_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through fusion module.
        
        Args:
            audio_features: Audio features of shape (batch, time, audio_dim).
            visual_features: Visual features of shape (batch, time, visual_dim).
            audio_lengths: Length of each audio sequence.
            visual_lengths: Length of each visual sequence.
            
        Returns:
            Fused features of shape (batch, time, hidden_dim).
        """
        if self.method == "early":
            return self._early_fusion(audio_features, visual_features, audio_lengths, visual_lengths)
        elif self.method == "late":
            return self._late_fusion(audio_features, visual_features, audio_lengths, visual_lengths)
        elif self.method == "attention":
            return self._attention_fusion(audio_features, visual_features, audio_lengths, visual_lengths)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def _early_fusion(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Early fusion: concatenate features before processing."""
        # Concatenate audio and visual features
        fused = torch.cat([audio_features, visual_features], dim=-1)
        
        # Project to hidden dimension
        fused = self.input_projection(fused)
        fused = self.dropout(fused)
        
        # Create padding mask
        mask = None
        if audio_lengths is not None:
            batch_size, max_len = fused.size(0), fused.size(1)
            mask = torch.arange(max_len, device=fused.device).expand(batch_size, max_len) >= audio_lengths.unsqueeze(1)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused = layer(fused, src_key_padding_mask=mask)
        
        return fused
    
    def _late_fusion(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Late fusion: process modalities separately then combine."""
        # Project features to hidden dimension
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Create padding masks
        audio_mask = None
        visual_mask = None
        
        if audio_lengths is not None:
            batch_size, max_len = audio_proj.size(0), audio_proj.size(1)
            audio_mask = torch.arange(max_len, device=audio_proj.device).expand(batch_size, max_len) >= audio_lengths.unsqueeze(1)
        
        if visual_lengths is not None:
            batch_size, max_len = visual_proj.size(0), visual_proj.size(1)
            visual_mask = torch.arange(max_len, device=visual_proj.device).expand(batch_size, max_len) >= visual_lengths.unsqueeze(1)
        
        # Process audio features
        for layer in self.audio_layers:
            audio_proj = layer(audio_proj, src_key_padding_mask=audio_mask)
        
        # Process visual features
        for layer in self.visual_layers:
            visual_proj = layer(visual_proj, src_key_padding_mask=visual_mask)
        
        # Cross-modal attention
        audio_attended, _ = self.cross_attention(
            query=audio_proj,
            key=visual_proj,
            value=visual_proj,
            key_padding_mask=visual_mask
        )
        
        visual_attended, _ = self.cross_attention(
            query=visual_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=audio_mask
        )
        
        # Combine attended features
        combined = torch.cat([audio_attended, visual_attended], dim=-1)
        fused = self.fusion_layer(combined)
        
        return fused
    
    def _attention_fusion(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attention-based fusion."""
        # Project features to hidden dimension
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Create padding masks
        audio_mask = None
        visual_mask = None
        
        if audio_lengths is not None:
            batch_size, max_len = audio_proj.size(0), audio_proj.size(1)
            audio_mask = torch.arange(max_len, device=audio_proj.device).expand(batch_size, max_len) >= audio_lengths.unsqueeze(1)
        
        if visual_lengths is not None:
            batch_size, max_len = visual_proj.size(0), visual_proj.size(1)
            visual_mask = torch.arange(max_len, device=visual_proj.device).expand(batch_size, max_len) >= visual_lengths.unsqueeze(1)
        
        # Cross-modal attention
        audio_attended, _ = self.audio_to_visual(
            query=audio_proj,
            key=visual_proj,
            value=visual_proj,
            key_padding_mask=visual_mask
        )
        
        visual_attended, _ = self.visual_to_audio(
            query=visual_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=audio_mask
        )
        
        # Combine features
        fused = audio_attended + visual_attended
        fused = self.dropout(fused)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused = layer(fused, src_key_padding_mask=audio_mask)
        
        return fused
