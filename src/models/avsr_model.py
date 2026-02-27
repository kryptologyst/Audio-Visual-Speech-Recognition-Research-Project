"""Audio-Visual Speech Recognition model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from .conformer import ConformerEncoder
from .visual_encoder import VisualEncoder
from .fusion import MultiModalFusion


class AVSRModel(nn.Module):
    """Audio-Visual Speech Recognition model with Conformer architecture."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        audio_input_dim: int = 80,
        visual_input_dim: int = 3,
        audio_hidden_dim: int = 256,
        visual_hidden_dim: int = 256,
        fusion_hidden_dim: int = 512,
        num_audio_layers: int = 6,
        num_visual_layers: int = 4,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = "late",
        blank_token: int = 0
    ):
        """Initialize AVSR model.
        
        Args:
            vocab_size: Vocabulary size for output.
            audio_input_dim: Input dimension for audio features.
            visual_input_dim: Input channels for visual features.
            audio_hidden_dim: Hidden dimension for audio encoder.
            visual_hidden_dim: Hidden dimension for visual encoder.
            fusion_hidden_dim: Hidden dimension for fusion layer.
            num_audio_layers: Number of layers in audio encoder.
            num_visual_layers: Number of layers in visual encoder.
            num_fusion_layers: Number of layers in fusion module.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            fusion_method: Fusion method ("early", "late", "attention").
            blank_token: Blank token for CTC.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.fusion_method = fusion_method
        self.blank_token = blank_token
        
        # Audio encoder (Conformer)
        self.audio_encoder = ConformerEncoder(
            input_dim=audio_input_dim,
            hidden_dim=audio_hidden_dim,
            num_layers=num_audio_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Visual encoder (CNN)
        self.visual_encoder = VisualEncoder(
            input_channels=visual_input_dim,
            hidden_dim=visual_hidden_dim,
            num_layers=num_visual_layers,
            dropout=dropout
        )
        
        # Fusion module
        self.fusion = MultiModalFusion(
            audio_dim=audio_hidden_dim,
            visual_dim=visual_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            num_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            method=fusion_method
        )
        
        # Output projection
        self.output_projection = nn.Linear(fusion_hidden_dim, vocab_size)
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_token, reduction='mean', zero_infinity=True)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            audio_features: Audio features of shape (batch, time, features).
            visual_features: Visual features of shape (batch, time, channels, height, width).
            audio_lengths: Length of each audio sequence.
            visual_lengths: Length of each visual sequence.
            labels: Ground truth labels for training.
            label_lengths: Length of each label sequence.
            
        Returns:
            Dictionary containing logits, loss (if training), and other outputs.
        """
        batch_size = audio_features.size(0)
        
        # Encode audio features
        audio_encoded = self.audio_encoder(audio_features, audio_lengths)
        
        # Encode visual features
        visual_encoded = self.visual_encoder(visual_features, visual_lengths)
        
        # Fuse audio and visual features
        fused_features = self.fusion(audio_encoded, visual_encoded, audio_lengths, visual_lengths)
        
        # Project to vocabulary
        logits = self.output_projection(fused_features)
        
        outputs = {"logits": logits}
        
        # Compute loss if training
        if labels is not None and label_lengths is not None:
            # Transpose logits for CTC loss (time, batch, vocab)
            logits_t = logits.transpose(0, 1)
            
            # Compute CTC loss
            loss = self.ctc_loss(logits_t, labels, audio_lengths, label_lengths)
            outputs["loss"] = loss
        
        return outputs
    
    def transcribe(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        length_penalty: float = 1.0
    ) -> List[str]:
        """Transcribe audio-visual input to text.
        
        Args:
            audio_features: Audio features.
            visual_features: Visual features.
            audio_lengths: Length of each audio sequence.
            visual_lengths: Length of each visual sequence.
            beam_size: Beam size for beam search.
            length_penalty: Length penalty for beam search.
            
        Returns:
            List of transcribed texts.
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(audio_features, visual_features, audio_lengths, visual_lengths)
            logits = outputs["logits"]
            
            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to text (simplified - in practice, you'd use a proper CTC decoder)
            transcripts = []
            for pred in predictions:
                # Remove consecutive duplicates and blank tokens
                filtered_pred = []
                prev_token = None
                for token in pred:
                    if token != self.blank_token and token != prev_token:
                        filtered_pred.append(token.item())
                    prev_token = token
                
                # Convert token IDs to text (simplified)
                transcript = " ".join([str(token) for token in filtered_pred])
                transcripts.append(transcript)
        
        return transcripts
    
    def get_visual_contribution(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Analyze the contribution of visual modality.
        
        Args:
            audio_features: Audio features.
            visual_features: Visual features.
            audio_lengths: Length of each audio sequence.
            visual_lengths: Length of each visual sequence.
            
        Returns:
            Dictionary containing visual contribution metrics.
        """
        self.eval()
        
        with torch.no_grad():
            # Full model prediction
            full_outputs = self.forward(audio_features, visual_features, audio_lengths, visual_lengths)
            full_logits = full_outputs["logits"]
            
            # Audio-only prediction (zero out visual features)
            zero_visual = torch.zeros_like(visual_features)
            audio_only_outputs = self.forward(audio_features, zero_visual, audio_lengths, visual_lengths)
            audio_only_logits = audio_only_outputs["logits"]
            
            # Calculate contribution metrics
            logit_diff = torch.mean(torch.abs(full_logits - audio_only_logits))
            confidence_diff = torch.mean(torch.softmax(full_logits, dim=-1) - torch.softmax(audio_only_logits, dim=-1))
            
            return {
                "logit_difference": logit_diff.item(),
                "confidence_difference": torch.mean(torch.abs(confidence_diff)).item(),
                "visual_contribution_ratio": logit_diff.item() / (torch.mean(torch.abs(full_logits)).item() + 1e-8)
            }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'AVSRModel':
        """Create model from configuration dictionary.
        
        Args:
            config: Model configuration dictionary.
            
        Returns:
            Initialized AVSR model.
        """
        return cls(**config)
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            optimizer: Optimizer state to save.
            epoch: Current epoch number.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'model_config': {
                'vocab_size': self.vocab_size,
                'fusion_method': self.fusion_method,
                'blank_token': self.blank_token
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> Tuple['AVSRModel', Dict]:
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
            device: Device to load model on.
            
        Returns:
            Tuple of (model, checkpoint_info).
        """
        checkpoint = torch.load(path, map_location=device)
        
        model_config = checkpoint.get('model_config', {})
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict')
        }
        
        return model, checkpoint_info
