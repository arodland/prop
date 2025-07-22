import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import DiffusionModel, GuidanceModel
from tqdm import tqdm
import torch.nn.functional as F
import contextlib

def main(
    diffuser_checkpoint: Path = Path("diffuser_checkpoint.ckpt"),
    guidance_checkpoint: Path = Path("guidance_checkpoint.ckpt"),
    num_timesteps: int = 20,
    num_samples: int = 9,
    seed: int = 0,
    day: float = 123.0,
    hour: float = 12.0,
    ssn: float = 100.0,
    guidance_scale: float = 4.0,
):
    """Generates images from a trained diffusion model."""

    torch.manual_seed(seed)
    dm = DiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(device="cuda")
    gm = GuidanceModel.load_from_checkpoint(guidance_checkpoint).to(device="cuda")
    dm.eval()
    gm.eval()

    scheduler = diffusers.schedulers.DDPMScheduler()
    scheduler.set_timesteps(num_timesteps, device=dm.device)

    target = torch.tensor([
        math.sin(day * 2 * math.pi / 365),
        math.cos(day * 2 * math.pi / 365),
        math.sin(hour * 2 * math.pi / 24),
        math.cos(hour * 2 * math.pi / 24),
        ssn / 100.0 - 1.0,
    ])
    target = target.to(dm.device).unsqueeze(0).repeat(num_samples, 1)

    x = torch.randn((num_samples, 3, 184, 368), device=dm.device)

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        print("")
        torch.compiler.cudagraph_mark_step_begin()
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = dm.model(model_input, t).sample
        x = x.detach().requires_grad_()
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample

        guidance_out = gm.model(x0)
        print(guidance_out)
        guidance_loss = F.mse_loss(guidance_out, target)
        print(guidance_loss)
        guidance_grad = -torch.autograd.grad(guidance_loss * guidance_scale, x)[0]
        x = x.detach() + guidance_grad
        x = scheduler.step(noise_pred, t, x).prev_sample

    image_grid = tv.utils.make_grid(x, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "out/generated.png"
    tv.utils.save_image(image_grid, filename)
    print(f"Generated images saved to {filename}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
