"""Unit tests for AVSR project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.avsr_model import AVSRModel
from models.conformer import ConformerEncoder
from models.visual_encoder import VisualEncoder
from models.fusion import MultiModalFusion
from features.audio_features import extract_mel_spectrogram, normalize_audio
from features.visual_features import VisualFeatureExtractor
from utils import set_seed, get_device, count_parameters
from metrics import AVSRMetrics


class TestAVSRModel:
    """Test cases for AVSR model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = AVSRModel(
            vocab_size=100,
            audio_input_dim=80,
            visual_input_dim=3,
            audio_hidden_dim=256,
            visual_hidden_dim=256,
            fusion_hidden_dim=512
        )
        
        assert model.vocab_size == 100
        assert model.fusion_method == "late"
        assert model.blank_token == 0
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = AVSRModel(
            vocab_size=100,
            audio_input_dim=80,
            visual_input_dim=3,
            audio_hidden_dim=128,
            visual_hidden_dim=128,
            fusion_hidden_dim=256
        )
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        visual_features = torch.randn(batch_size, seq_len, 3, 64, 64)
        
        outputs = model(audio_features, visual_features)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, 100)
    
    def test_model_training_mode(self):
        """Test model in training mode."""
        model = AVSRModel(vocab_size=100)
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        visual_features = torch.randn(batch_size, seq_len, 3, 64, 64)
        labels = torch.randint(0, 100, (batch_size, 20))
        label_lengths = torch.randint(10, 20, (batch_size,))
        
        outputs = model(
            audio_features=audio_features,
            visual_features=visual_features,
            labels=labels,
            label_lengths=label_lengths
        )
        
        assert 'loss' in outputs
        assert outputs['loss'].item() >= 0
    
    def test_model_transcribe(self):
        """Test model transcription."""
        model = AVSRModel(vocab_size=100)
        model.eval()
        
        batch_size = 1
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        visual_features = torch.randn(batch_size, seq_len, 3, 64, 64)
        
        predictions = model.transcribe(audio_features, visual_features)
        
        assert isinstance(predictions, list)
        assert len(predictions) == batch_size


class TestConformerEncoder:
    """Test cases for Conformer encoder."""
    
    def test_conformer_initialization(self):
        """Test Conformer encoder initialization."""
        encoder = ConformerEncoder(
            input_dim=80,
            hidden_dim=256,
            num_layers=4,
            num_heads=8
        )
        
        assert encoder.input_projection.in_features == 80
        assert encoder.input_projection.out_features == 256
        assert len(encoder.conformer_blocks) == 4
    
    def test_conformer_forward(self):
        """Test Conformer encoder forward pass."""
        encoder = ConformerEncoder(
            input_dim=80,
            hidden_dim=256,
            num_layers=2,
            num_heads=8
        )
        
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 80)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_conformer_with_lengths(self):
        """Test Conformer encoder with sequence lengths."""
        encoder = ConformerEncoder(
            input_dim=80,
            hidden_dim=256,
            num_layers=2,
            num_heads=8
        )
        
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([80, 60])
        
        output = encoder(x, lengths)
        
        assert output.shape == (batch_size, seq_len, 256)


class TestVisualEncoder:
    """Test cases for visual encoder."""
    
    def test_visual_encoder_initialization(self):
        """Test visual encoder initialization."""
        encoder = VisualEncoder(
            input_channels=3,
            hidden_dim=256,
            num_layers=4
        )
        
        assert encoder.conv_layers is not None
        assert encoder.temporal_transformer is not None
        assert encoder.projection.out_features == 256
    
    def test_visual_encoder_forward(self):
        """Test visual encoder forward pass."""
        encoder = VisualEncoder(
            input_channels=3,
            hidden_dim=256,
            num_layers=2
        )
        
        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 3, 64, 64)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 256)


class TestMultiModalFusion:
    """Test cases for multi-modal fusion."""
    
    def test_fusion_initialization(self):
        """Test fusion module initialization."""
        fusion = MultiModalFusion(
            audio_dim=256,
            visual_dim=256,
            hidden_dim=512,
            method="late"
        )
        
        assert fusion.method == "late"
        assert fusion.hidden_dim == 512
    
    def test_late_fusion(self):
        """Test late fusion method."""
        fusion = MultiModalFusion(
            audio_dim=256,
            visual_dim=256,
            hidden_dim=512,
            method="late"
        )
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 256)
        visual_features = torch.randn(batch_size, seq_len, 256)
        
        output = fusion(audio_features, visual_features)
        
        assert output.shape == (batch_size, seq_len, 512)
    
    def test_early_fusion(self):
        """Test early fusion method."""
        fusion = MultiModalFusion(
            audio_dim=256,
            visual_dim=256,
            hidden_dim=512,
            method="early"
        )
        
        batch_size = 2
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 256)
        visual_features = torch.randn(batch_size, seq_len, 256)
        
        output = fusion(audio_features, visual_features)
        
        assert output.shape == (batch_size, seq_len, 512)


class TestAudioFeatures:
    """Test cases for audio feature extraction."""
    
    def test_extract_mel_spectrogram(self):
        """Test mel-spectrogram extraction."""
        # Create dummy audio
        audio = torch.randn(16000)  # 1 second at 16kHz
        
        mel_spec = extract_mel_spectrogram(
            audio,
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_spec.shape[1] > 0  # time frames
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = torch.randn(1000)
        
        # Test RMS normalization
        normalized = normalize_audio(audio, method="rms")
        assert torch.allclose(torch.sqrt(torch.mean(normalized ** 2)), torch.tensor(1.0), atol=1e-6)
        
        # Test peak normalization
        normalized = normalize_audio(audio, method="peak")
        assert torch.max(torch.abs(normalized)) <= 1.0 + 1e-6


class TestVisualFeatures:
    """Test cases for visual feature extraction."""
    
    def test_visual_feature_extractor_initialization(self):
        """Test visual feature extractor initialization."""
        extractor = VisualFeatureExtractor(
            image_size=(64, 64),
            crop_margin=0.1
        )
        
        assert extractor.image_size == (64, 64)
        assert extractor.crop_margin == 0.1
        assert extractor.lip_detector is not None


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_avsr_metrics_initialization(self):
        """Test AVSR metrics initialization."""
        metrics = AVSRMetrics()
        
        assert metrics.total_wer == 0.0
        assert metrics.total_cer == 0.0
        assert metrics.total_samples == 0
    
    def test_wer_calculation(self):
        """Test WER calculation."""
        metrics = AVSRMetrics()
        
        # Perfect match
        wer = metrics._calculate_wer("hello world", "hello world")
        assert wer == 0.0
        
        # Substitution error
        wer = metrics._calculate_wer("hello world", "hello there")
        assert wer == 0.5  # 1 substitution out of 2 words
    
    def test_cer_calculation(self):
        """Test CER calculation."""
        metrics = AVSRMetrics()
        
        # Perfect match
        cer = metrics._calculate_cer("hello", "hello")
        assert cer == 0.0
        
        # Substitution error
        cer = metrics._calculate_cer("hello", "helpo")
        assert cer == 0.2  # 1 substitution out of 5 characters
    
    def test_metrics_update(self):
        """Test metrics update."""
        metrics = AVSRMetrics()
        
        predictions = ["hello world", "good morning"]
        references = ["hello world", "good evening"]
        
        metrics.update(predictions, references)
        
        assert metrics.total_samples == 2
        assert metrics.total_wer > 0  # Should have some errors
        assert metrics.total_cer > 0  # Should have some errors


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(1)
        np_rand1 = np.random.rand(1)
        
        # Set seed again
        set_seed(42)
        
        # Generate again
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        # Should be the same
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        
        assert param_count == 55  # 10*5 + 5 bias


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline."""
        # Create model
        model = AVSRModel(
            vocab_size=100,
            audio_input_dim=80,
            visual_input_dim=3,
            audio_hidden_dim=128,
            visual_hidden_dim=128,
            fusion_hidden_dim=256
        )
        model.eval()
        
        # Create dummy data
        batch_size = 1
        seq_len = 100
        audio_features = torch.randn(batch_size, seq_len, 80)
        visual_features = torch.randn(batch_size, seq_len, 3, 64, 64)
        
        # Run inference
        with torch.no_grad():
            predictions = model.transcribe(audio_features, visual_features)
        
        assert isinstance(predictions, list)
        assert len(predictions) == batch_size
    
    def test_model_checkpoint_save_load(self):
        """Test model checkpoint saving and loading."""
        # Create model
        model = AVSRModel(vocab_size=100)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            model.save_checkpoint(checkpoint_path, epoch=10)
            
            # Load checkpoint
            loaded_model, checkpoint_info = AVSRModel.load_checkpoint(checkpoint_path)
            
            assert loaded_model.vocab_size == model.vocab_size
            assert checkpoint_info['epoch'] == 10
            
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


if __name__ == "__main__":
    pytest.main([__file__])
