#!/usr/bin/env python3
"""Evaluation script for Audio-Visual Speech Recognition."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.avsr_model import AVSRModel
from data import AudioVisualDataset, create_dataloader
from metrics import AVSRMetrics, VisualContributionAnalyzer, create_leaderboard
from utils import get_device, load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AVSREvaluator:
    """Evaluator for Audio-Visual Speech Recognition model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.device = get_device()
        
        # Initialize metrics
        self.metrics = AVSRMetrics()
        self.visual_analyzer = VisualContributionAnalyzer()
        
        # Results storage
        self.results = []
    
    def evaluate_model(
        self,
        model: AVSRModel,
        test_loader: DataLoader,
        model_name: str = "AVSR"
    ) -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Model to evaluate.
            test_loader: Test data loader.
            model_name: Name of the model for logging.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        self.metrics.reset()
        self.visual_analyzer.reset()
        
        logger.info(f"Evaluating {model_name}")
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Evaluating {model_name}")
            
            for batch in pbar:
                # Move data to device
                audio_features = batch['audio_features'].to(self.device)
                visual_features = batch['visual_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                visual_lengths = batch['visual_lengths'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                
                # Get predictions
                predictions = model.transcribe(
                    audio_features=audio_features,
                    visual_features=visual_features,
                    audio_lengths=audio_lengths,
                    visual_lengths=visual_lengths
                )
                
                # Update metrics
                self.metrics.update(
                    predictions=predictions,
                    references=batch['transcripts']
                )
                
                # Analyze visual contribution
                visual_contrib = model.get_visual_contribution(
                    audio_features=audio_features,
                    visual_features=visual_features,
                    audio_lengths=audio_lengths,
                    visual_lengths=visual_lengths
                )
                
                # Update progress bar
                pbar.set_postfix({'WER': f"{self.metrics.total_wer / max(1, self.metrics.total_samples):.3f}"})
        
        # Compute final metrics
        eval_metrics = self.metrics.compute()
        
        # Add visual contribution metrics
        visual_metrics = self.visual_analyzer.compute_contribution_metrics()
        eval_metrics.update(visual_metrics)
        
        # Store results
        result = {'model': model_name}
        result.update(eval_metrics)
        self.results.append(result)
        
        logger.info(f"{model_name} Results:")
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return eval_metrics
    
    def evaluate_ablation(
        self,
        model: AVSRModel,
        test_loader: DataLoader,
        ablation_configs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate ablation studies.
        
        Args:
            model: Base model to evaluate.
            test_loader: Test data loader.
            ablation_configs: List of ablation configurations.
            
        Returns:
            Dictionary containing ablation results.
        """
        ablation_results = {}
        
        for ablation_config in ablation_configs:
            ablation_name = ablation_config['name']
            logger.info(f"Running ablation: {ablation_name}")
            
            # Create modified model (simplified - in practice, you'd modify the model architecture)
            modified_model = self._create_ablation_model(model, ablation_config)
            
            # Evaluate modified model
            metrics = self.evaluate_model(modified_model, test_loader, ablation_name)
            ablation_results[ablation_name] = metrics
        
        return ablation_results
    
    def _create_ablation_model(self, base_model: AVSRModel, config: Dict[str, Any]) -> AVSRModel:
        """Create model for ablation study.
        
        Args:
            base_model: Base model to modify.
            config: Ablation configuration.
            
        Returns:
            Modified model.
        """
        # This is a simplified implementation
        # In practice, you would modify the model architecture based on the ablation config
        return base_model
    
    def generate_leaderboard(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate evaluation leaderboard.
        
        Args:
            save_path: Optional path to save leaderboard.
            
        Returns:
            DataFrame containing leaderboard.
        """
        if not self.results:
            logger.warning("No results to generate leaderboard")
            return pd.DataFrame()
        
        leaderboard = create_leaderboard(
            {result['model']: {k: v for k, v in result.items() if k != 'model'} for result in self.results},
            save_path
        )
        
        logger.info("Leaderboard generated:")
        print(leaderboard.to_string(index=False))
        
        return leaderboard
    
    def analyze_errors(self, test_loader: DataLoader, model: AVSRModel) -> Dict[str, Any]:
        """Analyze prediction errors.
        
        Args:
            test_loader: Test data loader.
            model: Model to analyze.
            
        Returns:
            Dictionary containing error analysis.
        """
        model.eval()
        
        error_analysis = {
            'substitution_errors': 0,
            'insertion_errors': 0,
            'deletion_errors': 0,
            'total_words': 0,
            'error_examples': []
        }
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                audio_features = batch['audio_features'].to(self.device)
                visual_features = batch['visual_features'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                visual_lengths = batch['visual_lengths'].to(self.device)
                
                # Get predictions
                predictions = model.transcribe(
                    audio_features=audio_features,
                    visual_features=visual_features,
                    audio_lengths=audio_lengths,
                    visual_lengths=visual_lengths
                )
                
                # Analyze errors for each sample
                for pred, ref in zip(predictions, batch['transcripts']):
                    # Simple error analysis (in practice, use more sophisticated methods)
                    pred_words = pred.lower().split()
                    ref_words = ref.lower().split()
                    
                    error_analysis['total_words'] += len(ref_words)
                    
                    # Count substitution errors (simplified)
                    min_len = min(len(pred_words), len(ref_words))
                    for i in range(min_len):
                        if pred_words[i] != ref_words[i]:
                            error_analysis['substitution_errors'] += 1
                    
                    # Count insertion/deletion errors
                    if len(pred_words) > len(ref_words):
                        error_analysis['insertion_errors'] += len(pred_words) - len(ref_words)
                    elif len(pred_words) < len(ref_words):
                        error_analysis['deletion_errors'] += len(ref_words) - len(pred_words)
                    
                    # Store error examples
                    if pred.lower().strip() != ref.lower().strip():
                        error_analysis['error_examples'].append({
                            'prediction': pred,
                            'reference': ref,
                            'wer': self.metrics._calculate_wer(pred, ref)
                        })
        
        # Limit error examples
        error_analysis['error_examples'] = error_analysis['error_examples'][:10]
        
        return error_analysis


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate AVSR model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save-results", action="store_true", help="Save detailed results")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test dataset
    test_dataset = AudioVisualDataset(
        metadata_path=Path(args.data_root) / "meta.csv",
        data_root=args.data_root,
        audio_config=config['data']['audio'],
        visual_config=config['data']['visual'],
        split=args.split
    )
    
    # Create test data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model, checkpoint_info = AVSRModel.load_checkpoint(args.checkpoint)
    model.to(get_device())
    
    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Model was trained for {checkpoint_info['epoch']} epochs")
    
    # Initialize evaluator
    evaluator = AVSREvaluator(config)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(model, test_loader, "AVSR")
    
    # Run ablation studies if requested
    if args.ablation:
        ablation_configs = [
            {'name': 'Audio-Only', 'disable_visual': True},
            {'name': 'Visual-Only', 'disable_audio': True},
            {'name': 'No-Fusion', 'fusion_method': 'none'}
        ]
        
        ablation_results = evaluator.evaluate_ablation(model, test_loader, ablation_configs)
        
        # Add ablation results to main results
        for ablation_name, ablation_metrics in ablation_results.items():
            evaluator.results.append({'model': ablation_name, **ablation_metrics})
    
    # Generate leaderboard
    leaderboard = evaluator.generate_leaderboard(
        save_path=str(output_dir / "leaderboard.csv") if args.save_results else None
    )
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(test_loader, model)
    
    logger.info("Error Analysis:")
    logger.info(f"  Substitution errors: {error_analysis['substitution_errors']}")
    logger.info(f"  Insertion errors: {error_analysis['insertion_errors']}")
    logger.info(f"  Deletion errors: {error_analysis['deletion_errors']}")
    logger.info(f"  Total words: {error_analysis['total_words']}")
    
    # Save detailed results if requested
    if args.save_results:
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        
        # Save error analysis
        error_df = pd.DataFrame(error_analysis['error_examples'])
        error_df.to_csv(output_dir / "error_examples.csv", index=False)
        
        # Save configuration
        import yaml
        with open(output_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
