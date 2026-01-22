#!/usr/bin/env python3
"""
Training script for observation-conditioned DiT model.

This model learns to interpolate from sparse observations using cross-attention.
Key features:
- Cross-attention to sparse observation embeddings
- Handles missing channels per observation
- SSN corruption during training (30% dropout/noise)
- Combined diffusion + observation fitting loss
- Float32 precision (critical for weak parameter signals)
"""
import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from models import ObservationConditionedDiT, IRIData
import diffusers


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Resuming from checkpoint: {sys.argv[1]}")
        model = ObservationConditionedDiT.load_from_checkpoint(sys.argv[1])
        # Don't load VAE from checkpoint - use finetuned version
        model.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
        for param in model.vae.parameters():
            param.requires_grad = False
        model.vae.eval()
    else:
        print("Creating new observation-conditioned DiT model")
        model = ObservationConditionedDiT(
            input_size=(24, 48),
            patch_size=2,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            class_embed_size=256,
            max_observations=60,  # Reduced from 100 for faster training
            pred_type='v_prediction'
        )

    # Checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename='obs-dit-v-{epoch}-{val_loss:.2g}',
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # Data module with more workers for parallelism
    data = IRIData("combined", train_batch=200, add_noise=0.0001)

    # Override dataloader to use more workers
    original_train_dataloader = data.train_dataloader
    def train_dataloader_with_workers():
        loader = original_train_dataloader()
        # Increase workers for CPU parallelism
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=32,  # Increased from 16
            pin_memory=True,
            persistent_workers=True,
        )
    data.train_dataloader = train_dataloader_with_workers

    # Trainer
    trainer = L.Trainer(
        max_epochs=250,
        log_every_n_steps=50,
        precision="32-true",  # CRITICAL: float32 for weak parameter signals
        callbacks=[
            checkpoint_callback,
            ModelCheckpoint(
                dirpath="checkpoints",
                filename='obs-dit-averaged',
                save_top_k=1
            ),
            StochasticWeightAveraging(swa_lrs=1e-5),
        ],
        logger=TensorBoardLogger("lightning_logs", name="obs-dit"),
        gradient_clip_val=1.0,  # Clip gradients for stability
    )

    print("\n" + "=" * 60)
    print("Observation-Conditioned DiT Training")
    print("=" * 60)
    print(f"Batch size: {data.train_batch}")
    print(f"Max epochs: {trainer.max_epochs}")
    print(f"Precision: {trainer.precision}")
    print(f"Max observations: {model.hparams.max_observations}")
    print("=" * 60 + "\n")

    trainer.fit(model, data)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.6f}")
    print("=" * 60)
