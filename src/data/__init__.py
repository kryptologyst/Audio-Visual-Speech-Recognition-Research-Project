"""Audio-visual data processing pipeline."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..features.audio_features import extract_mel_spectrogram, load_audio
from ..features.visual_features import VisualFeatureExtractor, synchronize_audio_visual

logger = logging.getLogger(__name__)


class AudioVisualDataset(Dataset):
    """Dataset for audio-visual speech recognition."""
    
    def __init__(
        self,
        metadata_path: Union[str, Path],
        data_root: Union[str, Path],
        audio_config: Dict,
        visual_config: Dict,
        vocab: Optional[Dict[str, int]] = None,
        max_audio_length: Optional[float] = None,
        max_visual_length: Optional[int] = None,
        split: Optional[str] = None
    ):
        """Initialize dataset.
        
        Args:
            metadata_path: Path to metadata CSV file.
            data_root: Root directory for data files.
            audio_config: Audio processing configuration.
            visual_config: Visual processing configuration.
            vocab: Vocabulary mapping.
            max_audio_length: Maximum audio length in seconds.
            max_visual_length: Maximum visual length in frames.
            split: Data split (train/val/test).
        """
        self.data_root = Path(data_root)
        self.audio_config = audio_config
        self.visual_config = visual_config
        self.max_audio_length = max_audio_length
        self.max_visual_length = max_visual_length
        self.vocab = vocab or {}
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by split if specified
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        
        # Initialize feature extractors
        self.visual_extractor = VisualFeatureExtractor(
            image_size=visual_config['image_size'],
            crop_margin=visual_config['crop_margin']
        )
        
        logger.info(f"Loaded {len(self.metadata)} samples for split: {split or 'all'}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Item index.
            
        Returns:
            Dictionary containing audio, visual features and labels.
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = self.data_root / row['audio_path']
        audio, sr = load_audio(
            audio_path,
            sample_rate=self.audio_config['sample_rate'],
            normalize=True
        )
        
        # Extract audio features
        audio_features = extract_mel_spectrogram(
            audio,
            sample_rate=self.audio_config['sample_rate'],
            n_mels=self.audio_config['n_mels'],
            n_fft=self.audio_config['n_fft'],
            hop_length=self.audio_config['hop_length'],
            win_length=self.audio_config['win_length'],
            fmin=self.audio_config['fmin'],
            fmax=self.audio_config['fmax']
        )
        
        # Load visual features
        video_path = self.data_root / row['video_path']
        visual_features = self.visual_extractor.extract_from_video(
            video_path,
            fps=self.visual_config['fps'],
            max_frames=self.max_visual_length
        )
        
        # Synchronize audio and visual features
        audio_features, visual_features = synchronize_audio_visual(
            audio_features,
            visual_features,
            audio_sample_rate=self.audio_config['sample_rate'],
            visual_fps=self.visual_config['fps'],
            hop_length=self.audio_config['hop_length']
        )
        
        # Convert to tensors
        audio_features = audio_features.float()
        visual_features = torch.stack(visual_features).float()
        
        # Truncate if necessary
        if self.max_audio_length is not None:
            max_audio_frames = int(self.max_audio_length * self.audio_config['sample_rate'] / self.audio_config['hop_length'])
            if audio_features.size(-1) > max_audio_frames:
                audio_features = audio_features[:, :max_audio_frames]
                visual_features = visual_features[:max_audio_frames]
        
        # Process transcript
        transcript = row['transcript']
        if self.vocab:
            # Convert text to token IDs
            tokens = self._text_to_tokens(transcript)
            labels = torch.tensor(tokens, dtype=torch.long)
        else:
            # Use character-level encoding
            labels = torch.tensor([ord(c) for c in transcript], dtype=torch.long)
        
        return {
            'audio_features': audio_features,
            'visual_features': visual_features,
            'labels': labels,
            'audio_length': torch.tensor(audio_features.size(-1), dtype=torch.long),
            'visual_length': torch.tensor(visual_features.size(0), dtype=torch.long),
            'label_length': torch.tensor(len(labels), dtype=torch.long),
            'transcript': transcript,
            'id': row['id']
        }
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs.
        
        Args:
            text: Input text.
            
        Returns:
            List of token IDs.
        """
        # Simple word-level tokenization
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get('<unk>', 0))
        return tokens


class AudioVisualProcessor:
    """Audio-visual data processor for inference."""
    
    def __init__(
        self,
        audio_config: Dict,
        visual_config: Dict,
        vocab: Optional[Dict[str, int]] = None
    ):
        """Initialize processor.
        
        Args:
            audio_config: Audio processing configuration.
            visual_config: Visual processing configuration.
            vocab: Vocabulary mapping.
        """
        self.audio_config = audio_config
        self.visual_config = visual_config
        self.vocab = vocab or {}
        
        # Initialize feature extractors
        self.visual_extractor = VisualFeatureExtractor(
            image_size=visual_config['image_size'],
            crop_margin=visual_config['crop_margin']
        )
    
    def process_file(
        self,
        audio_path: Union[str, Path],
        video_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio and video files for inference.
        
        Args:
            audio_path: Path to audio file.
            video_path: Path to video file.
            
        Returns:
            Tuple of (audio_features, visual_features).
        """
        # Load and process audio
        audio, sr = load_audio(
            audio_path,
            sample_rate=self.audio_config['sample_rate'],
            normalize=True
        )
        
        audio_features = extract_mel_spectrogram(
            audio,
            sample_rate=self.audio_config['sample_rate'],
            n_mels=self.audio_config['n_mels'],
            n_fft=self.audio_config['n_fft'],
            hop_length=self.audio_config['hop_length'],
            win_length=self.audio_config['win_length'],
            fmin=self.audio_config['fmin'],
            fmax=self.audio_config['fmax']
        )
        
        # Load and process visual features
        visual_features = self.visual_extractor.extract_from_video(
            video_path,
            fps=self.visual_config['fps']
        )
        
        # Synchronize features
        audio_features, visual_features = synchronize_audio_visual(
            audio_features,
            visual_features,
            audio_sample_rate=self.audio_config['sample_rate'],
            visual_fps=self.visual_config['fps'],
            hop_length=self.audio_config['hop_length']
        )
        
        # Convert to tensors and add batch dimension
        audio_features = audio_features.float().unsqueeze(0)
        visual_features = torch.stack(visual_features).float().unsqueeze(0)
        
        return audio_features, visual_features
    
    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        video_paths: List[Union[str, Path]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch of audio-visual files.
        
        Args:
            audio_paths: List of audio file paths.
            video_paths: List of video file paths.
            
        Returns:
            Tuple of (audio_features, visual_features, audio_lengths, visual_lengths).
        """
        batch_audio = []
        batch_visual = []
        audio_lengths = []
        visual_lengths = []
        
        for audio_path, video_path in zip(audio_paths, video_paths):
            audio_features, visual_features = self.process_file(audio_path, video_path)
            
            batch_audio.append(audio_features.squeeze(0))
            batch_visual.append(visual_features.squeeze(0))
            audio_lengths.append(audio_features.size(-1))
            visual_lengths.append(visual_features.size(1))
        
        # Pad sequences to same length
        max_audio_len = max(audio_lengths)
        max_visual_len = max(visual_lengths)
        
        padded_audio = []
        padded_visual = []
        
        for audio_feat, visual_feat in zip(batch_audio, batch_visual):
            # Pad audio
            if audio_feat.size(-1) < max_audio_len:
                padding = max_audio_len - audio_feat.size(-1)
                audio_feat = torch.nn.functional.pad(audio_feat, (0, padding))
            
            # Pad visual
            if visual_feat.size(0) < max_visual_len:
                padding = max_visual_len - visual_feat.size(0)
                visual_feat = torch.nn.functional.pad(visual_feat, (0, 0, 0, 0, 0, padding))
            
            padded_audio.append(audio_feat)
            padded_visual.append(visual_feat)
        
        # Stack into batch tensors
        audio_batch = torch.stack(padded_audio)
        visual_batch = torch.stack(padded_visual)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        visual_lengths = torch.tensor(visual_lengths, dtype=torch.long)
        
        return audio_batch, visual_batch, audio_lengths, visual_lengths


def create_dataloader(
    dataset: AudioVisualDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """Create data loader for dataset.
    
    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        collate_fn: Custom collate function.
        
    Returns:
        DataLoader instance.
    """
    if collate_fn is None:
        collate_fn = avsr_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def avsr_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for AVSR dataset.
    
    Args:
        batch: List of dataset items.
        
    Returns:
        Batched data dictionary.
    """
    # Separate different types of data
    audio_features = [item['audio_features'] for item in batch]
    visual_features = [item['visual_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    audio_lengths = torch.stack([item['audio_length'] for item in batch])
    visual_lengths = torch.stack([item['visual_length'] for item in batch])
    label_lengths = torch.stack([item['label_length'] for item in batch])
    
    # Pad sequences to same length
    max_audio_len = max(audio_lengths).item()
    max_visual_len = max(visual_lengths).item()
    max_label_len = max(label_lengths).item()
    
    # Pad audio features
    padded_audio = []
    for audio_feat in audio_features:
        if audio_feat.size(-1) < max_audio_len:
            padding = max_audio_len - audio_feat.size(-1)
            audio_feat = torch.nn.functional.pad(audio_feat, (0, padding))
        padded_audio.append(audio_feat)
    
    # Pad visual features
    padded_visual = []
    for visual_feat in visual_features:
        if visual_feat.size(0) < max_visual_len:
            padding = max_visual_len - visual_feat.size(0)
            visual_feat = torch.nn.functional.pad(visual_feat, (0, 0, 0, 0, 0, padding))
        padded_visual.append(visual_feat)
    
    # Pad labels
    padded_labels = []
    for label in labels:
        if label.size(0) < max_label_len:
            padding = max_label_len - label.size(0)
            label = torch.nn.functional.pad(label, (0, padding), value=-1)  # Use -1 for padding
        padded_labels.append(label)
    
    return {
        'audio_features': torch.stack(padded_audio),
        'visual_features': torch.stack(padded_visual),
        'labels': torch.stack(padded_labels),
        'audio_lengths': audio_lengths,
        'visual_lengths': visual_lengths,
        'label_lengths': label_lengths,
        'transcripts': [item['transcript'] for item in batch],
        'ids': [item['id'] for item in batch]
    }
