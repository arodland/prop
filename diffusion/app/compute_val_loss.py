#!/usr/bin/env python3
"""
Compute validation loss for a checkpoint without training.
Supports both UNet and DiT models, with configurable precision.
"""
import sys
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
from models import ConditionedDiffusionModel, DiTDiffusionModel, IRIData
from util import scale_to_diffusion, scale_from_diffusion


def compute_validation_loss(
    checkpoint_path: str,
    model_type: str = "unet",
    precision: str = "float32",
    batch_size: int = 32,
    num_batches: int = None,
):
    """
    Compute validation loss for a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: "unet" or "dit"
        precision: "float32" or "bfloat16"
        batch_size: Batch size for validation
        num_batches: Number of batches to evaluate (None = all)

    Returns:
        Mean validation loss
    """
    print("=" * 60)
    print(f"Validation Loss Computation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {model_type}")
    print(f"Precision:  {precision}")
    print("=" * 60)

    # Load model
    if model_type == "unet":
        model = ConditionedDiffusionModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cuda"
        )
    elif model_type == "dit":
        model = DiTDiffusionModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cuda"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    model = model.cuda()

    # Set precision
    if precision == "bfloat16":
        print("\nConverting model to bfloat16...")
        # Convert diffusion model (UNet/DiT) to bfloat16
        model.model = model.model.to(torch.bfloat16)
        # Convert parameter encoder to bfloat16
        model.param_encoder = model.param_encoder.to(torch.bfloat16)
        # Keep VAE in float32 for stability (it uses float inputs)
        model.vae = model.vae.to(torch.float32)
        print("  Diffusion model: bfloat16")
        print("  VAE: float32 (for stability)")
    elif precision == "float32":
        print("\nModel in float32 (default)")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Load validation data
    print("\nLoading validation data...")
    data = IRIData("combined", train_batch=batch_size, add_noise=0.001)
    data.setup()
    val_loader = data.val_dataloader()

    print(f"Validation batches: {len(val_loader)}")
    if num_batches:
        print(f"Will evaluate: {num_batches} batches")

    # Compute validation loss
    print("\nComputing validation loss...")
    val_losses = []
    val_losses_pixel_ch0 = []
    val_losses_pixel_ch1 = []
    val_losses_pixel_ch2 = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            if num_batches and batch_idx >= num_batches:
                break

            # Move batch to GPU
            images = batch["images"].cuda()
            raw_target = batch["raw_target"].cuda()

            # Manually implement validation logic to handle precision
            # Step 1: Encode images with VAE (always float32 for stability)
            latents_raw = model.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(model.unscale_latents(latents_raw), (0, 2, 0, 1))

            # Step 2: Convert latents and targets for bfloat16 mode
            if precision == "bfloat16":
                latents = latents.to(torch.bfloat16)
                raw_target = raw_target.to(torch.bfloat16)

            # Step 3: Encode parameters
            # DiT uses rope_enc (no scrambler), UNet uses full param_encoder (with scrambler)
            if model_type == "dit":
                encoded_targets = model.param_encoder.rope_enc(raw_target)
            else:
                encoded_targets = model.param_encoder(raw_target)

            # Step 4: Sample noise and timesteps
            noise = torch.randn_like(latents)
            steps = torch.randint(
                model.scheduler.config.num_train_timesteps,
                (images.size(0),),
                device=model.device
            )

            # Step 5: Add noise to latents
            noisy_latents = model.scheduler.add_noise(latents, noise, steps)

            # Step 6: Model forward pass
            model_pred = model.model(noisy_latents, steps, class_labels=encoded_targets).sample

            # Step 7: Compute loss using model's loss function
            loss = model.model_loss(model_pred, latents, noise, steps)
            val_losses.append(loss.item())

            # Step 8: Compute per-channel pixel-space losses
            # First, extract predicted clean latent from model prediction
            if model.hparams.pred_type == 'epsilon':
                # model_pred is predicted noise, so predicted_latent = (noisy - noise) / sqrt(alpha)
                alpha_t = model.scheduler.alphas_cumprod[steps].view(-1, 1, 1, 1)
                predicted_latent = (noisy_latents - (1 - alpha_t).sqrt() * model_pred) / alpha_t.sqrt()
            elif model.hparams.pred_type == 'v_prediction':
                # v = sqrt(alpha_t) * noise - sqrt(1-alpha_t) * x
                # Solve for x: x = sqrt(1-alpha_t) * v + sqrt(alpha_t) * noisy
                alpha_t = model.scheduler.alphas_cumprod[steps].view(-1, 1, 1, 1)
                sigma_t = (1 - alpha_t)
                predicted_latent = alpha_t.sqrt() * noisy_latents - sigma_t.sqrt() * model_pred
            else:
                raise ValueError(f"Unknown prediction type: {model.hparams.pred_type}")

            # Decode to pixel space (remove padding and scale)
            target_latent_unpadded = latents[..., :23, :46]
            predicted_latent_unpadded = predicted_latent[..., :23, :46]

            target_latent_scaled = model.scale_latents(target_latent_unpadded)
            predicted_latent_scaled = model.scale_latents(predicted_latent_unpadded)

            # Decode with VAE (cast to float32 for VAE stability)
            target_pixels = model.vae.decode(target_latent_scaled.to(torch.float32)).sample
            predicted_pixels = model.vae.decode(predicted_latent_scaled.to(torch.float32)).sample

            # Scale from [-1, 1] to [0, 1]
            target_pixels = scale_from_diffusion(target_pixels).clamp(0, 1)
            predicted_pixels = scale_from_diffusion(predicted_pixels).clamp(0, 1)

            # Compute per-channel MSE
            for ch_idx in range(3):
                ch_loss = F.mse_loss(predicted_pixels[:, ch_idx], target_pixels[:, ch_idx])
                if ch_idx == 0:
                    val_losses_pixel_ch0.append(ch_loss.item())
                elif ch_idx == 1:
                    val_losses_pixel_ch1.append(ch_loss.item())
                elif ch_idx == 2:
                    val_losses_pixel_ch2.append(ch_loss.item())

    # Compute statistics
    mean_loss = sum(val_losses) / len(val_losses)
    min_loss = min(val_losses)
    max_loss = max(val_losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in val_losses) / len(val_losses)) ** 0.5

    # Compute per-channel pixel statistics
    mean_pixel_ch0 = sum(val_losses_pixel_ch0) / len(val_losses_pixel_ch0)
    mean_pixel_ch1 = sum(val_losses_pixel_ch1) / len(val_losses_pixel_ch1)
    mean_pixel_ch2 = sum(val_losses_pixel_ch2) / len(val_losses_pixel_ch2)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nBatches evaluated: {len(val_losses)}")
    print(f"\nValidation Loss - Latent Space ({precision}):")
    print(f"  Mean: {mean_loss:.6f}")
    print(f"  Std:  {std_loss:.6f}")
    print(f"  Min:  {min_loss:.6f}")
    print(f"  Max:  {max_loss:.6f}")
    print(f"\nValidation Loss - Pixel Space Per Channel ({precision}):")
    print(f"  Channel 0: {mean_pixel_ch0:.6f}")
    print(f"  Channel 1: {mean_pixel_ch1:.6f}")
    print(f"  Channel 2: {mean_pixel_ch2:.6f}")

    return mean_loss


def main():
    parser = argparse.ArgumentParser(
        description="Compute validation loss for a model checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "dit"],
        help="Model architecture (default: unet)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Precision for inference (default: float32)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches to evaluate (default: all)"
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Compute validation loss
    mean_loss = compute_validation_loss(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        precision=args.precision,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )

    print(f"\nFinal validation loss: {mean_loss:.6f}")


if __name__ == "__main__":
    main()
