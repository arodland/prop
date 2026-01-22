#!/usr/bin/env python3
"""
Test VAE reconstruction quality to see if it's the bottleneck.
"""
import torch
import torch.nn.functional as F
from models import IRIData
from util import scale_to_diffusion, scale_from_diffusion
import diffusers
import numpy as np
from tqdm import tqdm

def test_vae_reconstruction():
    """Measure VAE round-trip reconstruction loss on validation set."""

    print("=" * 60)
    print("VAE Reconstruction Quality Test")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
    vae = vae.cuda()
    vae.eval()

    print(f"VAE latent magnitude: {vae.latent_magnitude}")

    # Load validation data
    print("\nLoading validation data...")
    data = IRIData("combined", train_batch=32, add_noise=0.001)
    data.setup()
    val_loader = data.val_dataloader()

    print(f"Validation batches: {len(val_loader)}")

    # Test reconstruction
    print("\nTesting reconstruction...")

    all_losses = []
    all_losses_per_channel = [[], [], []]
    all_latent_magnitudes = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            images = batch["images"].cuda()  # (B, 3, 181, 361)

            # Encode
            images_scaled = scale_to_diffusion(images)  # Scale to [-1, 1]
            latents_raw = vae.encode(images_scaled).latents  # (B, 4, 23, 46)
            latents_unscaled = latents_raw / vae.latent_magnitude

            # Track latent statistics
            all_latent_magnitudes.append(latents_unscaled.abs().mean().item())

            # Decode
            latents_scaled = latents_unscaled * vae.latent_magnitude
            reconstructed = vae.decode(latents_scaled).sample  # (B, 3, 181, 361)
            reconstructed_scaled = scale_from_diffusion(reconstructed)  # Scale to [0, 1]

            # Compute loss
            loss = F.mse_loss(reconstructed_scaled, images)
            all_losses.append(loss.item())

            # Per-channel loss
            for c in range(3):
                channel_loss = F.mse_loss(reconstructed_scaled[:, c], images[:, c])
                all_losses_per_channel[c].append(channel_loss.item())

    # Statistics
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    min_loss = np.min(all_losses)
    max_loss = np.max(all_losses)

    mean_latent_mag = np.mean(all_latent_magnitudes)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nOverall Reconstruction MSE Loss:")
    print(f"  Mean: {mean_loss:.6f}")
    print(f"  Std:  {std_loss:.6f}")
    print(f"  Min:  {min_loss:.6f}")
    print(f"  Max:  {max_loss:.6f}")

    print(f"\nPer-Channel Reconstruction MSE Loss:")
    for c in range(3):
        mean_c = np.mean(all_losses_per_channel[c])
        print(f"  Channel {c}: {mean_c:.6f}")

    print(f"\nLatent Statistics:")
    print(f"  Mean absolute value: {mean_latent_mag:.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    print(f"\nVAE reconstruction loss: {mean_loss:.6f}")

    if mean_loss > 0.05:
        print("\n⚠️  WARNING: VAE reconstruction loss is HIGH (>0.05)")
        print("   This is likely a SIGNIFICANT bottleneck.")
        print("   The diffusion model cannot do better than the VAE quality.")
        print(f"\n   Estimated upper bound on diffusion loss: ~{mean_loss:.3f}")

    elif mean_loss > 0.02:
        print("\n⚠️  CAUTION: VAE reconstruction loss is MODERATE (0.02-0.05)")
        print("   This may be limiting diffusion model performance.")
        print(f"\n   Estimated upper bound on diffusion loss: ~{mean_loss:.3f}")

    elif mean_loss > 0.01:
        print("\n✓ VAE reconstruction quality is GOOD (0.01-0.02)")
        print("  VAE is probably not the main bottleneck.")

    else:
        print("\n✓ VAE reconstruction quality is EXCELLENT (<0.01)")
        print("  VAE is definitely not the bottleneck.")

    # Compare to reported diffusion loss
    print("\n" + "=" * 60)
    print("Comparison to Diffusion Model Loss")
    print("=" * 60)

    reported_loss = 0.100
    print(f"\nReported diffusion model loss: ~{reported_loss:.3f}")
    print(f"VAE reconstruction loss:        {mean_loss:.6f}")

    if mean_loss > reported_loss * 0.5:
        print(f"\n🚨 CRITICAL: VAE loss is {mean_loss/reported_loss*100:.0f}% of diffusion loss!")
        print("   VAE reconstruction quality is the PRIMARY BOTTLENECK.")
        print("\n   Recommendations:")
        print("   1. Fine-tune VAE further on this dataset")
        print("   2. Use a higher-capacity VAE")
        print("   3. Increase VAE latent dimensions")

    elif mean_loss > reported_loss * 0.2:
        print(f"\n⚠️  VAE loss is {mean_loss/reported_loss*100:.0f}% of diffusion loss")
        print("   VAE quality may be a significant factor.")
        print("\n   The diffusion model may be near its theoretical limit given VAE quality.")

    else:
        print(f"\n✓ VAE loss is only {mean_loss/reported_loss*100:.0f}% of diffusion loss")
        print("  VAE is not the primary bottleneck.")
        print("  Look at other factors: conditioning, architecture, scheduler, etc.")

    # Additional diagnostics
    print("\n" + "=" * 60)
    print("Additional Diagnostics")
    print("=" * 60)

    # Sample a batch for visual inspection
    print("\nSample statistics (first batch):")
    batch = next(iter(val_loader))
    images = batch["images"].cuda()

    with torch.no_grad():
        images_scaled = scale_to_diffusion(images)
        latents_raw = vae.encode(images_scaled).latents
        latents_unscaled = latents_raw / vae.latent_magnitude
        latents_scaled = latents_unscaled * vae.latent_magnitude
        reconstructed = vae.decode(latents_scaled).sample
        reconstructed_scaled = scale_from_diffusion(reconstructed)

    print(f"\n  Original image:")
    print(f"    Mean: {images.mean().item():.4f}")
    print(f"    Std:  {images.std().item():.4f}")
    print(f"    Min:  {images.min().item():.4f}")
    print(f"    Max:  {images.max().item():.4f}")

    print(f"\n  Reconstructed image:")
    print(f"    Mean: {reconstructed_scaled.mean().item():.4f}")
    print(f"    Std:  {reconstructed_scaled.std().item():.4f}")
    print(f"    Min:  {reconstructed_scaled.min().item():.4f}")
    print(f"    Max:  {reconstructed_scaled.max().item():.4f}")

    print(f"\n  Latent representation:")
    print(f"    Mean: {latents_unscaled.mean().item():+.4f}")
    print(f"    Std:  {latents_unscaled.std().item():.4f}")
    print(f"    Min:  {latents_unscaled.min().item():+.4f}")
    print(f"    Max:  {latents_unscaled.max().item():+.4f}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_vae_reconstruction()
