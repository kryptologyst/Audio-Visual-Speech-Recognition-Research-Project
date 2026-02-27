"""Evaluation metrics for Audio-Visual Speech Recognition."""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AVSRMetrics:
    """Metrics calculator for AVSR evaluation."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """Initialize metrics calculator.
        
        Args:
            vocab: Vocabulary mapping for token-to-text conversion.
        """
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Initialize metric accumulators
        self.reset()
    
    def reset(self) -> None:
        """Reset all metric accumulators."""
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.total_samples = 0
        self.total_visual_contribution = 0.0
        self.confidence_scores = []
        self.predictions = []
        self.references = []
    
    def update(
        self,
        predictions: List[str],
        references: List[str],
        visual_contributions: Optional[List[float]] = None,
        confidence_scores: Optional[List[float]] = None
    ) -> None:
        """Update metrics with new predictions and references.
        
        Args:
            predictions: List of predicted transcripts.
            references: List of reference transcripts.
            visual_contributions: List of visual contribution scores.
            confidence_scores: List of confidence scores.
        """
        batch_size = len(predictions)
        
        for i in range(batch_size):
            pred = predictions[i].lower().strip()
            ref = references[i].lower().strip()
            
            # Calculate WER and CER
            wer = self._calculate_wer(pred, ref)
            cer = self._calculate_cer(pred, ref)
            
            self.total_wer += wer
            self.total_cer += cer
            self.total_samples += 1
            
            # Store predictions and references
            self.predictions.append(pred)
            self.references.append(ref)
            
            # Visual contribution
            if visual_contributions is not None:
                self.total_visual_contribution += visual_contributions[i]
            
            # Confidence scores
            if confidence_scores is not None:
                self.confidence_scores.append(confidence_scores[i])
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary containing computed metrics.
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {
            'wer': self.total_wer / self.total_samples,
            'cer': self.total_cer / self.total_samples,
            'visual_contribution': self.total_visual_contribution / self.total_samples if self.total_visual_contribution > 0 else 0.0,
            'num_samples': self.total_samples
        }
        
        # Confidence metrics
        if self.confidence_scores:
            metrics.update({
                'avg_confidence': np.mean(self.confidence_scores),
                'confidence_std': np.std(self.confidence_scores),
                'confidence_min': np.min(self.confidence_scores),
                'confidence_max': np.max(self.confidence_scores)
            })
        
        return metrics
    
    def _calculate_wer(self, prediction: str, reference: str) -> float:
        """Calculate Word Error Rate (WER).
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            WER value.
        """
        pred_words = prediction.split()
        ref_words = reference.split()
        
        if len(ref_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        # Dynamic programming for edit distance
        dp = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]
        
        # Initialize base cases
        for i in range(len(pred_words) + 1):
            dp[i][0] = i
        for j in range(len(ref_words) + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[len(pred_words)][len(ref_words)] / len(ref_words)
    
    def _calculate_cer(self, prediction: str, reference: str) -> float:
        """Calculate Character Error Rate (CER).
        
        Args:
            prediction: Predicted text.
            reference: Reference text.
            
        Returns:
            CER value.
        """
        if len(reference) == 0:
            return 1.0 if len(prediction) > 0 else 0.0
        
        # Dynamic programming for edit distance
        dp = [[0] * (len(reference) + 1) for _ in range(len(prediction) + 1)]
        
        # Initialize base cases
        for i in range(len(prediction) + 1):
            dp[i][0] = i
        for j in range(len(reference) + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                if prediction[i-1] == reference[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[len(prediction)][len(reference)] / len(reference)


class VisualContributionAnalyzer:
    """Analyzer for visual modality contribution."""
    
    def __init__(self):
        """Initialize visual contribution analyzer."""
        self.reset()
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self.audio_only_scores = []
        self.visual_only_scores = []
        self.multimodal_scores = []
        self.samples = []
    
    def update(
        self,
        audio_only_logits: torch.Tensor,
        visual_only_logits: torch.Tensor,
        multimodal_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """Update analyzer with new predictions.
        
        Args:
            audio_only_logits: Logits from audio-only model.
            visual_only_logits: Logits from visual-only model.
            multimodal_logits: Logits from multimodal model.
            labels: Ground truth labels.
        """
        # Calculate CTC losses for each modality
        ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
        
        # Assume all sequences have same length for simplicity
        input_lengths = torch.full((audio_only_logits.size(0),), audio_only_logits.size(1), dtype=torch.long)
        target_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long)
        
        # Transpose logits for CTC loss
        audio_logits_t = audio_only_logits.transpose(0, 1)
        visual_logits_t = visual_only_logits.transpose(0, 1)
        multimodal_logits_t = multimodal_logits.transpose(0, 1)
        
        # Calculate losses
        audio_losses = ctc_loss(audio_logits_t, labels, input_lengths, target_lengths)
        visual_losses = ctc_loss(visual_logits_t, labels, input_lengths, target_lengths)
        multimodal_losses = ctc_loss(multimodal_logits_t, labels, input_lengths, target_lengths)
        
        # Store scores (lower loss = better performance)
        self.audio_only_scores.extend(-audio_losses.detach().cpu().numpy())
        self.visual_only_scores.extend(-visual_losses.detach().cpu().numpy())
        self.multimodal_scores.extend(-multimodal_losses.detach().cpu().numpy())
    
    def compute_contribution_metrics(self) -> Dict[str, float]:
        """Compute visual contribution metrics.
        
        Returns:
            Dictionary containing contribution metrics.
        """
        if not self.multimodal_scores:
            return {}
        
        audio_scores = np.array(self.audio_only_scores)
        visual_scores = np.array(self.visual_only_scores)
        multimodal_scores = np.array(self.multimodal_scores)
        
        # Calculate improvements
        audio_improvement = multimodal_scores - audio_scores
        visual_improvement = multimodal_scores - visual_scores
        
        # Calculate contribution ratios
        total_improvement = audio_improvement + visual_improvement
        visual_contribution_ratio = np.mean(visual_improvement / (total_improvement + 1e-8))
        
        return {
            'audio_only_performance': np.mean(audio_scores),
            'visual_only_performance': np.mean(visual_scores),
            'multimodal_performance': np.mean(multimodal_scores),
            'audio_improvement': np.mean(audio_improvement),
            'visual_improvement': np.mean(visual_improvement),
            'visual_contribution_ratio': visual_contribution_ratio,
            'multimodal_gain': np.mean(multimodal_scores - np.maximum(audio_scores, visual_scores))
        }


class ConfidenceCalibrator:
    """Confidence calibration for AVSR predictions."""
    
    def __init__(self):
        """Initialize confidence calibrator."""
        self.reset()
    
    def reset(self) -> None:
        """Reset calibrator state."""
        self.confidences = []
        self.accuracies = []
    
    def update(
        self,
        confidences: List[float],
        predictions: List[str],
        references: List[str]
    ) -> None:
        """Update calibrator with new predictions.
        
        Args:
            confidences: List of confidence scores.
            predictions: List of predicted transcripts.
            references: List of reference transcripts.
        """
        for conf, pred, ref in zip(confidences, predictions, references):
            # Calculate accuracy (simplified - in practice, use WER)
            accuracy = 1.0 if pred.lower().strip() == ref.lower().strip() else 0.0
            
            self.confidences.append(conf)
            self.accuracies.append(accuracy)
    
    def compute_calibration_metrics(self, num_bins: int = 10) -> Dict[str, float]:
        """Compute confidence calibration metrics.
        
        Args:
            num_bins: Number of bins for calibration analysis.
            
        Returns:
            Dictionary containing calibration metrics.
        """
        if not self.confidences:
            return {}
        
        confidences = np.array(self.confidences)
        accuracies = np.array(self.accuracies)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate calibration error
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': ece,
            'avg_confidence': np.mean(confidences),
            'avg_accuracy': np.mean(accuracies),
            'confidence_std': np.std(confidences),
            'accuracy_std': np.std(accuracies)
        }


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Create evaluation leaderboard.
    
    Args:
        results: Dictionary of results from different models/configurations.
        save_path: Optional path to save leaderboard.
        
    Returns:
        DataFrame containing leaderboard.
    """
    leaderboard_data = []
    
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        leaderboard_data.append(row)
    
    leaderboard = pd.DataFrame(leaderboard_data)
    
    # Sort by WER (lower is better)
    if 'wer' in leaderboard.columns:
        leaderboard = leaderboard.sort_values('wer')
    
    # Save if path provided
    if save_path:
        leaderboard.to_csv(save_path, index=False)
        logger.info(f"Leaderboard saved to {save_path}")
    
    return leaderboard
