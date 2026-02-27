"""Conformer encoder implementation for audio processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Attention mask.
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.out_linear(context)
        
        return output


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, hidden_dim: int, kernel_size: int = 31, dropout: float = 0.1):
        """Initialize convolution module.
        
        Args:
            hidden_dim: Hidden dimension size.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim * 2, hidden_dim * 2, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=hidden_dim * 2
        )
        self.pointwise_conv2 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution module.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        residual = x
        
        # Transpose for convolution (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise convolution
        x = self.pointwise_conv1(x)
        
        # GLU activation
        x = F.glu(x, dim=1)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Pointwise convolution
        x = self.pointwise_conv2(x)
        
        # Transpose back (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(self, hidden_dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        """Initialize feed-forward module.
        
        Args:
            hidden_dim: Hidden dimension size.
            expansion_factor: Expansion factor for hidden layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        expanded_dim = hidden_dim * expansion_factor
        
        self.linear1 = nn.Linear(hidden_dim, expanded_dim)
        self.linear2 = nn.Linear(expanded_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward module.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        residual = x
        
        x = self.layer_norm(x)
        x = F.swish(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class ConformerBlock(nn.Module):
    """Conformer block combining attention, convolution, and feed-forward modules."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """Initialize Conformer block.
        
        Args:
            hidden_dim: Hidden dimension size.
            num_heads: Number of attention heads.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.ff1 = FeedForwardModule(hidden_dim, dropout=dropout)
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.conv = ConvolutionModule(hidden_dim, kernel_size, dropout)
        self.ff2 = FeedForwardModule(hidden_dim, dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Attention mask.
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        # Feed-forward 1
        x = self.ff1(x)
        
        # Multi-head self-attention
        residual = x
        x = self.layer_norm(x)
        x = self.self_attn(x, mask)
        x = x + residual
        
        # Convolution module
        x = self.conv(x)
        
        # Feed-forward 2
        x = self.ff2(x)
        
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """Initialize Conformer encoder.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of Conformer blocks.
            num_heads: Number of attention heads.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Conformer encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            lengths: Length of each sequence.
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
        
        # Apply Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x
