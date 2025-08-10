import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import DiffusionModel

def main(
    checkpoint: Path = Path("checkpoint.ckpt"),
    num_timesteps: int = 1000,
    num_samples: int = 1,
    seed: int = 0,
    channels: int = 3,
):
    """Generates images from a trained diffusion model."""

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with device:
        model = DiffusionModel()
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scheduler = diffusers.schedulers.DDIMScheduler()
    pipe = diffusers.DDIMPipeline(model.model, scheduler)
    pipe = pipe.to(device=device)

    with torch.inference_mode():
        (pil_images, ) = pipe(
            batch_size=num_samples,
            num_inference_steps=num_timesteps,
            output_type="pil",
            return_dict=False
        )
    images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                         for pil_image in pil_images])
    image_grid = tv.utils.make_grid(images, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "out/generated.png"
    tv.utils.save_image(image_grid, filename)
    print(f"Generated images saved to {filename}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
