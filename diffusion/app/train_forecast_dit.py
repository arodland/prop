#!/usr/bin/env python3
"""
Training script for 24-hour forecast DiT model.

Trains ForecastObservationConditionedDiT with:
- Aged observation conditioning (9D observations)
- Previous map conditioning via input concatenation
- Hierarchical CFG dropout
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
from pathlib import Path

from models_forecast import ForecastObservationConditionedDiT
from data.forecast_datamodule import ForecastDataModule


def main():
    parser = argparse.ArgumentParser(description='Train forecast DiT model')

    # Data
    parser.add_argument('--hdf5-path', type=str, default='data/forecast_sequences.h5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--num-observations', type=int, default=50,
                        help='Number of observations to sample per example')
    parser.add_argument('--train-batch', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--val-batch', type=int, default=64,
                        help='Validation batch size')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of dataloader workers')

    # Model
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='Hidden size')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--pred-type', type=str, default='v_prediction',
                        choices=['epsilon', 'v_prediction'],
                        help='Prediction type')
    parser.add_argument('--cfg-dropout-obs', type=float, default=0.10,
                        help='CFG dropout rate for observations only')
    parser.add_argument('--cfg-dropout-prev', type=float, default=0.10,
                        help='CFG dropout rate for previous map only')
    parser.add_argument('--cfg-dropout-both', type=float, default=0.10,
                        help='CFG dropout rate for both conditioning')

    # Training
    parser.add_argument('--max-epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--precision', type=str, default='32-true',
                        help='Training precision (32-true for weak signals)')
    parser.add_argument('--accumulate-grad-batches', type=int, default=2,
                        help='Gradient accumulation batches')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa-lrs', type=float, default=5e-5,
                        help='SWA learning rate')

    # Logging
    parser.add_argument('--log-dir', type=str, default='lightning_logs',
                        help='TensorBoard log directory')
    parser.add_argument('--experiment-name', type=str, default='forecast_dit',
                        help='Experiment name for logging')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/forecast',
                        help='Checkpoint directory')
    parser.add_argument('--save-top-k', type=int, default=3,
                        help='Save top k checkpoints')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Set matmul precision for performance
    torch.set_float32_matmul_precision("high")

    # Create model
    print("Creating model...")
    model = ForecastObservationConditionedDiT(
        input_size=(24, 48),
        patch_size=2,
        in_channels=8,  # 4 (noisy) + 4 (prev_map)
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        class_embed_size=256,
        max_observations=100,
        pred_type=args.pred_type,
        cfg_dropout_obs=args.cfg_dropout_obs,
        cfg_dropout_prev=args.cfg_dropout_prev,
        cfg_dropout_both=args.cfg_dropout_both,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data module
    print("Creating data module...")
    data = ForecastDataModule(
        hdf5_path=args.hdf5_path,
        train_batch=args.train_batch,
        val_batch=args.val_batch,
        num_workers=args.num_workers,
        num_observations=args.num_observations,
        train_fraction=0.9,
        seed=42,
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='forecast-dit-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Stochastic Weight Averaging
    if args.swa:
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=0.75,  # Start SWA at 75% of training
        )
        callbacks.append(swa_callback)
        print(f"Using SWA with learning rate {args.swa_lrs}")

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
    )

    # Print training info
    print("\n" + "="*80)
    print("Training Configuration:")
    print("="*80)
    print(f"Model: ForecastObservationConditionedDiT")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Depth: {args.depth}")
    print(f"  - Num heads: {args.num_heads}")
    print(f"  - Prediction type: {args.pred_type}")
    print(f"  - CFG dropout (obs/prev/both): {args.cfg_dropout_obs}/{args.cfg_dropout_prev}/{args.cfg_dropout_both}")
    print(f"\nData:")
    print(f"  - HDF5 path: {args.hdf5_path}")
    print(f"  - Train batch: {args.train_batch}")
    print(f"  - Val batch: {args.val_batch}")
    print(f"  - Num observations: {args.num_observations}")
    print(f"  - Num workers: {args.num_workers}")
    print(f"\nTraining:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"  - Effective batch size: {args.train_batch * args.accumulate_grad_batches}")
    print(f"  - Gradient clip: {args.gradient_clip_val}")
    print(f"  - SWA: {args.swa}")
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    print(f"Logs will be saved to: {args.log_dir}/{args.experiment_name}")
    print("="*80 + "\n")

    # Train
    print("Starting training...")
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=args.resume,
    )

    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
