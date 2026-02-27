"""Visual feature extraction utilities for lip reading."""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, Tuple, List, Optional
from pathlib import Path
import torchvision.transforms as transforms


class LipDetector:
    """Lip region detector using OpenCV."""
    
    def __init__(self):
        """Initialize lip detector."""
        # Load pre-trained cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
        )
    
    def detect_lips(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect lip region in a video frame.
        
        Args:
            frame: Input video frame (BGR format).
            
        Returns:
            Cropped lip region or None if not detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Focus on lower half of face for mouth region
            roi_gray = gray[y + int(h/2): y + h, x: x + w]
            roi_color = frame[y + int(h/2): y + h, x: x + w]
            
            # Detect mouth in the face region
            mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            
            for (mx, my, mw, mh) in mouths:
                # Extract mouth region
                mouth_img = roi_color[my: my + mh, mx: mx + mw]
                return mouth_img
        
        return None


class VisualFeatureExtractor:
    """Visual feature extractor for lip reading."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        crop_margin: float = 0.1
    ):
        """Initialize visual feature extractor.
        
        Args:
            image_size: Target image size (height, width).
            crop_margin: Margin for cropping around detected lips.
        """
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.lip_detector = LipDetector()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Extract visual features from a single frame.
        
        Args:
            frame: Input video frame.
            
        Returns:
            Visual features tensor or None if lips not detected.
        """
        # Detect lip region
        lip_region = self.lip_detector.detect_lips(frame)
        
        if lip_region is None:
            return None
        
        # Apply transforms
        try:
            features = self.transform(lip_region)
            return features
        except Exception:
            return None
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        fps: int = 25,
        max_frames: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Extract visual features from video file.
        
        Args:
            video_path: Path to video file.
            fps: Target frames per second.
            max_frames: Maximum number of frames to extract.
            
        Returns:
            List of visual feature tensors.
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        frame_skip = max(1, int(original_fps / fps))
        
        features = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Extract features from frame
            frame_features = self.extract_from_frame(frame)
            if frame_features is not None:
                features.append(frame_features)
            
            frame_count += 1
            
            # Check max frames limit
            if max_frames and len(features) >= max_frames:
                break
        
        cap.release()
        
        if not features:
            raise ValueError(f"No lip regions detected in video: {video_path}")
        
        return features


class VisualAugmentation:
    """Visual data augmentation for lip reading."""
    
    def __init__(self):
        """Initialize visual augmentation."""
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to image.
        
        Args:
            image: Input image tensor.
            
        Returns:
            Augmented image tensor.
        """
        # Convert to PIL for augmentation
        pil_image = transforms.ToPILImage()(image)
        
        # Apply augmentation
        augmented = self.transform(pil_image)
        
        # Convert back to tensor
        return transforms.ToTensor()(augmented)


class VisualEncoder(nn.Module):
    """Visual encoder for lip reading using CNN."""
    
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
        
        # Final projection layer
        self.projection = nn.Linear(self.output_size, hidden_dim)
    
    def _get_conv_output_size(self) -> int:
        """Calculate output size after convolution layers."""
        # Create dummy input to calculate output size
        dummy_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return output.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through visual encoder.
        
        Args:
            x: Input visual features of shape (batch, channels, height, width).
            
        Returns:
            Encoded visual features of shape (batch, hidden_dim).
        """
        # Apply convolution layers
        conv_out = self.conv_layers(x)
        
        # Flatten and project
        flattened = conv_out.view(conv_out.size(0), -1)
        encoded = self.projection(flattened)
        
        return encoded


def synchronize_audio_visual(
    audio_features: torch.Tensor,
    visual_features: List[torch.Tensor],
    audio_sample_rate: int = 16000,
    visual_fps: int = 25,
    hop_length: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synchronize audio and visual features.
    
    Args:
        audio_features: Audio features tensor.
        visual_features: List of visual feature tensors.
        audio_sample_rate: Audio sample rate.
        visual_fps: Visual frame rate.
        hop_length: Audio hop length.
        
    Returns:
        Tuple of synchronized (audio_features, visual_features).
    """
    # Calculate time alignment
    audio_frame_rate = audio_sample_rate / hop_length
    visual_frame_rate = visual_fps
    
    # Calculate number of visual frames needed
    audio_frames = audio_features.size(-1)
    visual_frames_needed = int(audio_frames * visual_frame_rate / audio_frame_rate)
    
    # Pad or truncate visual features
    if len(visual_features) < visual_frames_needed:
        # Pad with last frame
        last_frame = visual_features[-1] if visual_features else torch.zeros(3, 64, 64)
        padding_frames = visual_frames_needed - len(visual_features)
        visual_features.extend([last_frame] * padding_frames)
    else:
        # Truncate to needed frames
        visual_features = visual_features[:visual_frames_needed]
    
    # Stack visual features
    visual_tensor = torch.stack(visual_features)
    
    return audio_features, visual_tensor
