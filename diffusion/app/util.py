def summarize_tensor(x):
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"


def scale_to_diffusion(x):
    """Scale images from [0, 1] to [-1, 1] for diffusion models and VAE."""
    return x * 2.0 - 1.0


def scale_from_diffusion(x):
    """Scale images from [-1, 1] back to [0, 1] for visualization."""
    return (x + 1.0) / 2.0
