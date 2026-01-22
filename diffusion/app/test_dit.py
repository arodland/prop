#!/usr/bin/env python3
"""
Quick test script to verify DiT model forward pass works correctly.
"""
import torch
import sys
sys.path.insert(0, '/home/andrew/code/prop/diffusion/app')

from models import DiTDiffusionModel

def test_dit_forward():
    print("Testing DiT model forward pass...")

    # Create model
    model = DiTDiffusionModel(pred_type='epsilon')
    model.eval()

    # Create dummy batch
    batch_size = 4
    latents = torch.randn(batch_size, 4, 24, 48)
    timesteps = torch.randint(0, 1000, (batch_size,))
    class_labels = torch.randn(batch_size, 256)

    print(f"Input shape: {latents.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Class labels shape: {class_labels.shape}")

    # Forward pass
    with torch.no_grad():
        output = model.model(latents, timesteps, class_labels=class_labels)

    print(f"Output type: {type(output)}")
    print(f"Has .sample attribute: {hasattr(output, 'sample')}")
    print(f"Output shape: {output.sample.shape}")

    # Check output shape
    assert output.sample.shape == latents.shape, \
        f"Output shape {output.sample.shape} doesn't match input shape {latents.shape}"

    # Test without class labels
    print("\nTesting without class labels...")
    with torch.no_grad():
        output_uncond = model.model(latents, timesteps, class_labels=None)

    print(f"Unconditioned output shape: {output_uncond.sample.shape}")
    assert output_uncond.sample.shape == latents.shape

    print("\n✅ All tests passed!")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Just the DiT backbone
    dit_params = sum(p.numel() for p in model.model.model.parameters())
    print(f"DiT backbone parameters: {dit_params:,}")

if __name__ == "__main__":
    test_dit_forward()
