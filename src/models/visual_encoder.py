"""Visual encoder implementation for lip reading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisualEncoder(nn.Module):
    """Visual encoder for lip reading using CNN and Transformer."""
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """Initialize visual encoder.
        
        Args:
            input_channels: Number of input channels.
            hidden_dim: Hidden dimension size.
            num_layers: Number of CNN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # CNN layers for spatial feature extraction
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - 1 - i))
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            ])
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        self.output_size = self._get_conv_output_size()
        
        # Temporal modeling with Transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.output_size,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Final projection layer
        self.projection = nn.Linear(self.output_size, hidden_dim)
    
    def _get_conv_output_size(self) -> int:
        """Calculate output size after convolution layers."""
        # Create dummy input to calculate output size
        dummy_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return output.view(1, -1).size(1)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through visual encoder.
        
        Args:
            x: Input visual features of shape (batch, time, channels, height, width).
            lengths: Length of each visual sequence.
            
        Returns:
            Encoded visual features of shape (batch, time, hidden_dim).
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        
        # Apply CNN layers
        conv_out = self.conv_layers(x)
        
        # Flatten spatial dimensions
        flattened = conv_out.view(batch_size * seq_len, -1)
        
        # Reshape back to sequence format
        visual_features = flattened.view(batch_size, seq_len, -1)
        
        # Apply temporal Transformer
        if lengths is not None:
            # Create padding mask
            max_len = visual_features.size(1)
            mask = torch.arange(max_len, device=visual_features.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
            visual_features = self.temporal_transformer(visual_features, src_key_padding_mask=mask)
        else:
            visual_features = self.temporal_transformer(visual_features)
        
        # Final projection
        visual_features = self.projection(visual_features)
        
        return visual_features
