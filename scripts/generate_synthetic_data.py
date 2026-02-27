#!/usr/bin/env python3
"""Synthetic data generator for AVSR testing and demonstration."""

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import random
import string

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, ensure_dir

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generator for synthetic audio-visual speech data."""
    
    def __init__(self, output_dir: str, config: Dict):
        """Initialize data generator.
        
        Args:
            output_dir: Output directory for generated data.
            config: Configuration dictionary.
        """
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create output directories
        ensure_dir(self.output_dir / "wav")
        ensure_dir(self.output_dir / "video")
        
        # Vocabulary for synthetic speech
        self.vocab = [
            "hello", "world", "good", "morning", "afternoon", "evening",
            "how", "are", "you", "today", "fine", "thank", "please",
            "yes", "no", "maybe", "sure", "okay", "great", "wonderful",
            "beautiful", "amazing", "fantastic", "excellent", "perfect"
        ]
        
        # Sample sentences
        self.sentences = [
            "hello world",
            "good morning",
            "how are you today",
            "thank you very much",
            "have a wonderful day",
            "this is amazing",
            "perfect weather today",
            "beautiful sunset tonight",
            "excellent work everyone",
            "fantastic performance"
        ]
        
        logger.info(f"Initialized synthetic data generator")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Sample sentences: {len(self.sentences)}")
    
    def generate_audio(self, text: str, duration: float = 3.0) -> np.ndarray:
        """Generate synthetic audio for given text.
        
        Args:
            text: Text to convert to audio.
            duration: Duration of audio in seconds.
            
        Returns:
            Generated audio signal.
        """
        sample_rate = self.config['audio']['sample_rate']
        num_samples = int(duration * sample_rate)
        
        # Generate synthetic speech-like signal
        # This is a simplified approach - in practice, you'd use TTS
        words = text.split()
        samples_per_word = num_samples // len(words)
        
        audio = np.zeros(num_samples)
        
        for i, word in enumerate(words):
            start_idx = i * samples_per_word
            end_idx = min((i + 1) * samples_per_word, num_samples)
            
            # Generate frequency based on word length
            base_freq = 200 + len(word) * 50  # 200-500 Hz range
            
            # Generate harmonic signal
            t = np.linspace(0, 1, end_idx - start_idx)
            signal = np.sin(2 * np.pi * base_freq * t)
            
            # Add harmonics
            signal += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            signal += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Add envelope (attack, sustain, decay)
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
            signal *= envelope
            
            # Add some noise
            noise = np.random.normal(0, 0.1, len(signal))
            signal += noise
            
            audio[start_idx:end_idx] = signal
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def generate_video(self, text: str, duration: float = 3.0) -> np.ndarray:
        """Generate synthetic video for given text.
        
        Args:
            text: Text to visualize.
            duration: Duration of video in seconds.
            
        Returns:
            Generated video frames.
        """
        fps = self.config['visual']['fps']
        image_size = self.config['visual']['image_size']
        num_frames = int(duration * fps)
        
        frames = []
        
        for frame_idx in range(num_frames):
            # Create synthetic lip movement frame
            frame = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            
            # Add face-like background
            cv2.rectangle(frame, (10, 10), (image_size[1]-10, image_size[0]-10), (220, 200, 180), -1)
            
            # Add mouth region
            mouth_width = 30 + int(10 * np.sin(frame_idx * 0.5))  # Animated mouth
            mouth_height = 15 + int(5 * np.cos(frame_idx * 0.3))
            
            center_x = image_size[1] // 2
            center_y = image_size[0] // 2 + 10
            
            mouth_x1 = center_x - mouth_width // 2
            mouth_y1 = center_y - mouth_height // 2
            mouth_x2 = center_x + mouth_width // 2
            mouth_y2 = center_y + mouth_height // 2
            
            # Draw mouth
            cv2.ellipse(frame, (center_x, center_y), (mouth_width//2, mouth_height//2), 0, 0, 360, (50, 50, 50), -1)
            
            # Add some text overlay
            cv2.putText(frame, text[:10], (10, image_size[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def generate_sample(self, sample_id: str, text: str, duration: float = 3.0) -> Dict:
        """Generate a single audio-visual sample.
        
        Args:
            sample_id: Unique identifier for the sample.
            text: Text content for the sample.
            duration: Duration of the sample.
            
        Returns:
            Dictionary containing sample metadata.
        """
        # Generate audio
        audio = self.generate_audio(text, duration)
        
        # Generate video
        video_frames = self.generate_video(text, duration)
        
        # Save audio file
        audio_path = self.output_dir / "wav" / f"{sample_id}.wav"
        torchaudio.save(
            str(audio_path),
            torch.from_numpy(audio).unsqueeze(0),
            self.config['audio']['sample_rate']
        )
        
        # Save video file
        video_path = self.output_dir / "video" / f"{sample_id}.mp4"
        self._save_video(video_frames, str(video_path), self.config['visual']['fps'])
        
        return {
            'id': sample_id,
            'audio_path': f"wav/{sample_id}.wav",
            'video_path': f"video/{sample_id}.mp4",
            'transcript': text,
            'speaker_id': f"speaker_{hash(text) % 5}",  # 5 different speakers
            'language': 'en',
            'duration': duration
        }
    
    def _save_video(self, frames: np.ndarray, output_path: str, fps: int) -> None:
        """Save video frames to file.
        
        Args:
            frames: Video frames array.
            output_path: Output video file path.
            fps: Frames per second.
        """
        height, width, channels = frames.shape[1:]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def generate_dataset(self, num_samples: int, splits: Dict[str, float] = None) -> None:
        """Generate complete dataset.
        
        Args:
            num_samples: Total number of samples to generate.
            splits: Dictionary with split names and ratios.
        """
        if splits is None:
            splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        # Calculate samples per split
        samples_per_split = {}
        remaining_samples = num_samples
        
        for split_name, ratio in splits.items():
            if split_name == list(splits.keys())[-1]:  # Last split gets remaining samples
                samples_per_split[split_name] = remaining_samples
            else:
                samples_per_split[split_name] = int(num_samples * ratio)
                remaining_samples -= samples_per_split[split_name]
        
        logger.info(f"Generating {num_samples} samples:")
        for split_name, count in samples_per_split.items():
            logger.info(f"  {split_name}: {count} samples")
        
        # Generate samples
        all_samples = []
        sample_id = 0
        
        for split_name, count in samples_per_split.items():
            logger.info(f"Generating {split_name} split...")
            
            for i in range(count):
                # Select random sentence
                text = random.choice(self.sentences)
                
                # Generate sample
                sample = self.generate_sample(f"{split_name}_{sample_id:04d}", text)
                sample['split'] = split_name
                
                all_samples.append(sample)
                sample_id += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Generated {i + 1}/{count} samples")
        
        # Save metadata
        metadata_df = pd.DataFrame(all_samples)
        metadata_path = self.output_dir / "meta.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        logger.info(f"Dataset generation completed!")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Audio files: {self.output_dir / 'wav'}")
        logger.info(f"Video files: {self.output_dir / 'video'}")
        
        # Print dataset statistics
        self._print_dataset_stats(metadata_df)
    
    def _print_dataset_stats(self, metadata_df: pd.DataFrame) -> None:
        """Print dataset statistics.
        
        Args:
            metadata_df: Dataset metadata DataFrame.
        """
        logger.info("\nDataset Statistics:")
        logger.info(f"Total samples: {len(metadata_df)}")
        
        # Split statistics
        split_counts = metadata_df['split'].value_counts()
        for split, count in split_counts.items():
            logger.info(f"  {split}: {count} samples")
        
        # Speaker statistics
        speaker_counts = metadata_df['speaker_id'].value_counts()
        logger.info(f"Speakers: {len(speaker_counts)}")
        for speaker, count in speaker_counts.items():
            logger.info(f"  {speaker}: {count} samples")
        
        # Duration statistics
        logger.info(f"Average duration: {metadata_df['duration'].mean():.2f}s")
        logger.info(f"Duration range: {metadata_df['duration'].min():.2f}s - {metadata_df['duration'].max():.2f}s")
        
        # Vocabulary statistics
        all_words = []
        for transcript in metadata_df['transcript']:
            all_words.extend(transcript.split())
        
        unique_words = set(all_words)
        logger.info(f"Unique words: {len(unique_words)}")
        logger.info(f"Total words: {len(all_words)}")


def main():
    """Main function for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic AVSR dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--config", type=str, default="configs/avsr_base.yaml", help="Config file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    try:
        from utils import load_config
        config = load_config(args.config)
    except ImportError:
        # Fallback configuration
        config = {
            'audio': {
                'sample_rate': 16000,
                'n_mels': 80,
                'n_fft': 1024,
                'hop_length': 256,
                'win_length': 1024,
                'fmin': 0,
                'fmax': 8000
            },
            'visual': {
                'fps': 25,
                'image_size': [64, 64],
                'crop_margin': 0.1
            }
        }
    
    # Create data generator
    generator = SyntheticDataGenerator(args.output_dir, config)
    
    # Generate dataset
    generator.generate_dataset(args.num_samples)
    
    logger.info("Synthetic data generation completed successfully!")


if __name__ == "__main__":
    main()
