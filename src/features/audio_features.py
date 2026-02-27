"""Audio feature extraction utilities."""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path


def extract_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Extract mel-spectrogram features from audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        n_mels: Number of mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        fmin: Minimum frequency for mel filters.
        fmax: Maximum frequency for mel filters.
        normalize: Whether to normalize the spectrogram.
        
    Returns:
        Mel-spectrogram tensor of shape (n_mels, time_frames).
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add batch dimension
    
    # Compute mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=fmin,
        f_max=fmax,
        normalized=normalize
    )
    
    mel_spec = mel_transform(audio)
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-8)
    
    return mel_spec.squeeze(0)  # Remove batch dimension


def extract_mfcc(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256
) -> torch.Tensor:
    """Extract MFCC features from audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Sample rate of audio.
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        
    Returns:
        MFCC tensor of shape (n_mfcc, time_frames).
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add batch dimension
    
    # Compute MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length
        }
    )
    
    mfcc = mfcc_transform(audio)
    
    return mfcc.squeeze(0)  # Remove batch dimension


def apply_spec_augment(
    spectrogram: torch.Tensor,
    freq_mask_max: int = 27,
    time_mask_max: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    p: float = 0.5
) -> torch.Tensor:
    """Apply SpecAugment data augmentation to spectrogram.
    
    Args:
        spectrogram: Input spectrogram.
        freq_mask_max: Maximum frequency mask size.
        time_mask_max: Maximum time mask size.
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
        p: Probability of applying augmentation.
        
    Returns:
        Augmented spectrogram.
    """
    if torch.rand(1) > p:
        return spectrogram
    
    augmented = spectrogram.clone()
    
    # Apply frequency masks
    for _ in range(num_freq_masks):
        freq_mask_size = torch.randint(0, freq_mask_max + 1, (1,)).item()
        if freq_mask_size > 0:
            freq_start = torch.randint(0, max(1, augmented.size(0) - freq_mask_size), (1,)).item()
            augmented[freq_start:freq_start + freq_mask_size, :] = 0
    
    # Apply time masks
    for _ in range(num_time_masks):
        time_mask_size = torch.randint(0, time_mask_max + 1, (1,)).item()
        if time_mask_size > 0:
            time_start = torch.randint(0, max(1, augmented.size(1) - time_mask_size), (1,)).item()
            augmented[:, time_start:time_start + time_mask_size] = 0
    
    return augmented


def add_noise(
    audio: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float = 10.0
) -> torch.Tensor:
    """Add noise to audio signal with specified SNR.
    
    Args:
        audio: Clean audio signal.
        noise: Noise signal.
        snr_db: Signal-to-noise ratio in dB.
        
    Returns:
        Noisy audio signal.
    """
    # Calculate signal and noise power
    signal_power = torch.mean(audio ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate required noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    # Scale noise to achieve desired SNR
    noise_scaled = noise * torch.sqrt(required_noise_power / noise_power)
    
    # Add noise to signal
    noisy_audio = audio + noise_scaled
    
    return noisy_audio


def speed_perturb(
    audio: torch.Tensor,
    sample_rate: int,
    speed_factor: float = 1.0
) -> torch.Tensor:
    """Apply speed perturbation to audio.
    
    Args:
        audio: Input audio signal.
        sample_rate: Original sample rate.
        speed_factor: Speed factor (>1.0 speeds up, <1.0 slows down).
        
    Returns:
        Speed-perturbed audio signal.
    """
    if speed_factor == 1.0:
        return audio
    
    # Resample to achieve speed perturbation
    new_sample_rate = int(sample_rate * speed_factor)
    
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    )
    
    perturbed_audio = resampler(audio)
    
    # Pad or truncate to maintain original length
    target_length = len(audio)
    if len(perturbed_audio) > target_length:
        perturbed_audio = perturbed_audio[:target_length]
    elif len(perturbed_audio) < target_length:
        padding = target_length - len(perturbed_audio)
        perturbed_audio = torch.nn.functional.pad(perturbed_audio, (0, padding))
    
    return perturbed_audio


def normalize_audio(audio: torch.Tensor, method: str = "rms") -> torch.Tensor:
    """Normalize audio signal.
    
    Args:
        audio: Input audio signal.
        method: Normalization method ("rms", "peak", "minmax").
        
    Returns:
        Normalized audio signal.
    """
    if method == "rms":
        rms = torch.sqrt(torch.mean(audio ** 2))
        return audio / (rms + 1e-8)
    elif method == "peak":
        peak = torch.max(torch.abs(audio))
        return audio / (peak + 1e-8)
    elif method == "minmax":
        min_val = torch.min(audio)
        max_val = torch.max(audio)
        return (audio - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load audio file and return tensor.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        normalize: Whether to normalize the audio.
        
    Returns:
        Tuple of (audio_tensor, actual_sample_rate).
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio using torchaudio
    waveform, sr = torchaudio.load(str(file_path))
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    # Remove batch dimension
    waveform = waveform.squeeze(0)
    
    # Normalize if requested
    if normalize:
        waveform = normalize_audio(waveform, method="rms")
    
    return waveform, sr
