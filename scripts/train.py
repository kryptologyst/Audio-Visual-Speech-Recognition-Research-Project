#!/usr/bin/env python3
"""Training script for Audio-Visual Speech Recognition."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.avsr_model import AVSRModel
from data import AudioVisualDataset, create_dataloader
from metrics import AVSRMetrics, VisualContributionAnalyzer
from utils import set_seed, get_device, load_config, save_config, EarlyStopping, log_model_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AVSRTrainer:
    """Trainer for Audio-Visual Speech Recognition model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize model
        self.model = AVSRModel.from_config(config['model'])
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = AVSRMetrics()
        self.visual_analyzer = VisualContributionAnalyzer()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10),
            min_delta=config['training'].get('min_delta', 0.001)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_wer = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Log model info
        log_model_info(self.model, logger)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['name'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=optimizer_config.get('betas', [0.9, 0.98]),
                eps=optimizer_config.get('eps', 1e-9)
            )
        elif optimizer_config['name'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['name'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['name'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move data to device
            audio_features = batch['audio_features'].to(self.device)
            visual_features = batch['visual_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            visual_lengths = batch['visual_lengths'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                audio_features=audio_features,
                visual_features=visual_features,
                audio_lengths=audio_lengths,
                visual_lengths=visual_lengths,
                labels=labels,
                label_lengths=label_lengths
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Validation metrics.
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move data to device
                audio_features = batch['audio_features'].to(self.device)
                visual_features = batch['visual_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                visual_lengths = batch['visual_lengths'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    audio_features=audio_features,
                    visual_features=visual_features,
                    audio_lengths=audio_lengths,
                    visual_lengths=visual_lengths,
                    labels=labels,
                    label_lengths=label_lengths
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                predictions = self.model.transcribe(
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
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Compute final metrics
        val_metrics = self.metrics.compute()
        val_metrics['loss'] = total_loss / num_batches
        
        return val_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints"
    ) -> None:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to save checkpoints.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        max_epochs = self.config['training']['max_epochs']
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_metrics['loss']:.4f}, WER = {val_metrics['wer']:.4f}")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_metrics['loss'],
                    'val/wer': val_metrics['wer'],
                    'val/cer': val_metrics['cer']
                })
            
            # Save checkpoint
            if val_metrics['wer'] < self.best_wer:
                self.best_wer = val_metrics['wer']
                checkpoint_path = checkpoint_dir / "best_model.pt"
                self.model.save_checkpoint(str(checkpoint_path), self.optimizer, epoch)
                logger.info(f"New best model saved with WER: {self.best_wer:.4f}")
            
            # Save latest checkpoint
            latest_path = checkpoint_dir / "latest_model.pt"
            self.model.save_checkpoint(str(latest_path), self.optimizer, epoch)
            
            # Early stopping
            if self.early_stopping(val_metrics['wer'], self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best WER: {self.best_wer:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AVSR model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-project", type=str, default="avsr-research", help="Wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=config,
            tags=["avsr", "training"]
        )
    
    # Create datasets
    train_dataset = AudioVisualDataset(
        metadata_path=Path(args.data_root) / "meta.csv",
        data_root=args.data_root,
        audio_config=config['data']['audio'],
        visual_config=config['data']['visual'],
        split="train"
    )
    
    val_dataset = AudioVisualDataset(
        metadata_path=Path(args.data_root) / "meta.csv",
        data_root=args.data_root,
        audio_config=config['data']['audio'],
        visual_config=config['data']['visual'],
        split="val"
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = AVSRTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        model, checkpoint_info = AVSRModel.load_checkpoint(args.resume)
        trainer.model = model
        trainer.optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        trainer.current_epoch = checkpoint_info['epoch']
        logger.info(f"Resumed training from epoch {trainer.current_epoch}")
    
    # Train model
    trainer.train(train_loader, val_loader, args.checkpoint_dir)
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
