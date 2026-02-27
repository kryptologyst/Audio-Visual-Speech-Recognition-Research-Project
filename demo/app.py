"""Streamlit demo application for Audio-Visual Speech Recognition."""

import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import sys
from typing import Optional, Tuple
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.avsr_model import AVSRModel
from data import AudioVisualProcessor
from utils import get_device, load_config

# Page configuration
st.set_page_config(
    page_title="Audio-Visual Speech Recognition Demo",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h3>⚠️ PRIVACY AND ETHICS DISCLAIMER</h3>
    <p><strong>This is a research and educational demo only.</strong></p>
    <ul>
        <li>❌ <strong>DO NOT</strong> use for biometric identification or authentication</li>
        <li>❌ <strong>DO NOT</strong> use for voice cloning or deepfake generation</li>
        <li>❌ <strong>DO NOT</strong> use for surveillance or privacy-invasive applications</li>
        <li>✅ <strong>ONLY</strong> use for legitimate research and educational purposes</li>
    </ul>
    <p>By using this demo, you agree to use it responsibly and ethically.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎤 Audio-Visual Speech Recognition Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Model loading
@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    """Load AVSR model."""
    try:
        config = load_config(config_path)
        model, _ = AVSRModel.load_checkpoint(checkpoint_path)
        model.to(get_device())
        model.eval()
        return model, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Check if model files exist
checkpoint_path = "checkpoints/best_model.pt"
config_path = "configs/avsr_base.yaml"

if not os.path.exists(checkpoint_path):
    st.error(f"Model checkpoint not found at {checkpoint_path}")
    st.info("Please train a model first using the training script.")
    st.stop()

if not os.path.exists(config_path):
    st.error(f"Config file not found at {config_path}")
    st.stop()

# Load model
with st.spinner("Loading model..."):
    model, config = load_model(checkpoint_path, config_path)

if model is None:
    st.error("Failed to load model. Please check the checkpoint and config files.")
    st.stop()

# Initialize processor
processor = AudioVisualProcessor(
    audio_config=config['data']['audio'],
    visual_config=config['data']['visual']
)

st.success("Model loaded successfully!")

# Sidebar settings
st.sidebar.subheader("Model Settings")
beam_size = st.sidebar.slider("Beam Size", 1, 10, 5)
length_penalty = st.sidebar.slider("Length Penalty", 0.5, 2.0, 1.0)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
show_visual_contribution = st.sidebar.checkbox("Show Visual Contribution", True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Files")
    
    # File upload
    uploaded_audio = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'm4a'],
        help="Upload an audio file for speech recognition"
    )
    
    uploaded_video = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file with synchronized audio"
    )
    
    # Record audio option
    st.subheader("🎙️ Record Audio")
    if st.button("Start Recording"):
        st.info("Audio recording functionality would be implemented here using browser APIs")
    
    # Record video option
    st.subheader("📹 Record Video")
    if st.button("Start Video Recording"):
        st.info("Video recording functionality would be implemented here using browser APIs")

with col2:
    st.subheader("🎯 Inference Results")
    
    if uploaded_audio and uploaded_video:
        # Save uploaded files temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            video_path = os.path.join(temp_dir, "video.mp4")
            
            # Save audio file
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            # Save video file
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Process files
            if st.button("🚀 Run Inference", type="primary"):
                with st.spinner("Processing audio-visual data..."):
                    try:
                        start_time = time.time()
                        
                        # Process audio-visual data
                        audio_features, visual_features = processor.process_file(audio_path, video_path)
                        
                        # Run inference
                        with torch.no_grad():
                            predictions = model.transcribe(
                                audio_features=audio_features,
                                visual_features=visual_features,
                                beam_size=beam_size,
                                length_penalty=length_penalty
                            )
                        
                        inference_time = time.time() - start_time
                        
                        # Display results
                        st.success("Inference completed!")
                        
                        # Transcription result
                        st.subheader("📝 Transcription")
                        transcript = predictions[0] if predictions else "No transcription available"
                        st.markdown(f"**Result:** {transcript}")
                        
                        # Performance metrics
                        st.subheader("📊 Performance Metrics")
                        
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.metric(
                                "Inference Time",
                                f"{inference_time:.2f}s"
                            )
                        
                        with col_metric2:
                            audio_duration = len(audio_features.squeeze()) / config['data']['audio']['sample_rate']
                            rtf = inference_time / audio_duration
                            st.metric(
                                "Real-Time Factor",
                                f"{rtf:.2f}x"
                            )
                        
                        with col_metric3:
                            st.metric(
                                "Audio Duration",
                                f"{audio_duration:.2f}s"
                            )
                        
                        # Visual contribution analysis
                        if show_visual_contribution:
                            st.subheader("👁️ Visual Contribution Analysis")
                            
                            with st.spinner("Analyzing visual contribution..."):
                                visual_contrib = model.get_visual_contribution(
                                    audio_features=audio_features,
                                    visual_features=visual_features
                                )
                            
                            col_viz1, col_viz2 = st.columns(2)
                            
                            with col_viz1:
                                st.metric(
                                    "Visual Contribution Ratio",
                                    f"{visual_contrib['visual_contribution_ratio']:.3f}"
                                )
                            
                            with col_viz2:
                                st.metric(
                                    "Logit Difference",
                                    f"{visual_contrib['logit_difference']:.3f}"
                                )
                        
                        # Audio and video previews
                        st.subheader("🎵 Media Previews")
                        
                        col_media1, col_media2 = st.columns(2)
                        
                        with col_media1:
                            st.audio(uploaded_audio)
                        
                        with col_media2:
                            st.video(uploaded_video)
                        
                        # Feature visualizations
                        st.subheader("📈 Feature Visualizations")
                        
                        # Audio spectrogram
                        st.subheader("Audio Spectrogram")
                        audio_spec = audio_features.squeeze().cpu().numpy()
                        
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        im = ax.imshow(audio_spec, aspect='auto', origin='lower')
                        ax.set_xlabel('Time Frames')
                        ax.set_ylabel('Mel Frequency Bins')
                        ax.set_title('Mel-Spectrogram')
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                        
                        # Visual features (first frame)
                        st.subheader("Visual Features (First Frame)")
                        visual_frame = visual_features.squeeze()[0].cpu().numpy()
                        visual_frame = np.transpose(visual_frame, (1, 2, 0))
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(visual_frame)
                        ax.set_title('Lip Region (First Frame)')
                        ax.axis('off')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
                        st.info("Please check that your audio and video files are valid and synchronized.")
    
    elif uploaded_audio or uploaded_video:
        st.warning("Please upload both audio and video files for AVSR inference.")
    
    else:
        st.info("Upload audio and video files to start inference.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Audio-Visual Speech Recognition Research Demo</p>
    <p>⚠️ For research and educational purposes only. Use responsibly and ethically.</p>
</div>
""", unsafe_allow_html=True)

# Additional information
with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    ### Audio-Visual Speech Recognition (AVSR)
    
    This demo showcases an AVSR system that combines audio and visual information for improved speech recognition accuracy.
    
    **Key Features:**
    - 🎤 Audio processing using mel-spectrograms
    - 👁️ Visual processing of lip movements
    - 🔄 Multi-modal fusion for enhanced accuracy
    - 📊 Real-time performance metrics
    - 🔍 Visual contribution analysis
    
    **Technical Details:**
    - Model: Conformer-based architecture
    - Audio: 16kHz sampling, 80 mel bins
    - Visual: 64x64 lip region images
    - Fusion: Late fusion with cross-modal attention
    
    **Use Cases:**
    - Noisy environment speech recognition
    - Multimodal speech understanding
    - Research in audio-visual processing
    
    **Limitations:**
    - Requires synchronized audio-visual data
    - Performance depends on lip visibility
    - Trained on specific datasets
    """)

with st.expander("🔧 Technical Specifications"):
    st.markdown("""
    ### Model Architecture
    
    **Audio Encoder:**
    - Conformer blocks with multi-head self-attention
    - Convolutional modules for local dependencies
    - Feed-forward networks with residual connections
    
    **Visual Encoder:**
    - CNN layers for spatial feature extraction
    - Transformer layers for temporal modeling
    - Lip region detection and cropping
    
    **Fusion Module:**
    - Cross-modal attention mechanisms
    - Late fusion strategy
    - Residual connections and layer normalization
    
    **Training Configuration:**
    - Optimizer: AdamW with cosine annealing
    - Loss: CTC loss for sequence modeling
    - Augmentation: SpecAugment, noise injection
    - Regularization: Dropout, weight decay
    """)
