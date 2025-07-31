# CRUSH.md - Codebase Guidelines for AI Agents

## Project Setup
- Uses Pixi for dependency management (pixi.toml)
- Runs in containerized environment via `./run` script
- PyTorch + Lightning for deep learning, Diffusers for diffusion models

## Commands
```bash
# Build the container
podman -r build -t prop-diffusion .

# Run training scripts
./run python app/train_diffusion.py
./run python app/train_guidance.py
./run python app/train_vae.py

# Run generation/inference
./run python app/generate.py --checkpoint path/to/checkpoint.ckpt
./run python app/generate_guided.py
```

## Code Style
- **Imports**: stdlib → third-party → local (torch, lightning as L, diffusers)
- **Type hints**: Use for CLI args (Path, int), minimal elsewhere
- **Naming**: snake_case functions/vars, PascalCase classes
- **No docstrings** except CLI main functions (one-line descriptions)
- **Device handling**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **Context managers**: Use `torch.no_grad()` or `torch.inference_mode()` for inference
- **Error handling**: Minimal - rely on framework defaults
- **File organization**: Separate model definitions (models.py) from training scripts
