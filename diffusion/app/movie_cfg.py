import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import ConditionedDiffusionModel
from tqdm import tqdm
import torch.nn.functional as F
import contextlib
from util import scale_from_diffusion

def main(
    diffuser_checkpoint: Path = Path("diffuser_checkpoint.ckpt"),
    num_timesteps: int = 100,
    num_samples: int = 9,
    seed: int = 0,
    year: float = 2023.0,
    day: float = 123.0,
    ssn: float = 100.0,
    guidance_scale: float = 15.0,
):
    """Generates images from a trained diffusion model."""

    dm = ConditionedDiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(device="cuda")
    dm.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(
        rescale_betas_zero_snr=True, prediction_type=dm.inference_scheduler.config.prediction_type)
    scheduler.set_timesteps(num_timesteps, device=dm.device)

    for qhour in range(96):
        torch.manual_seed(seed)
        hour = float(qhour) / 4.0
        target = torch.tensor([
            (year - 2000. + day / 365.) / 50.,
            math.sin(day * 2 * math.pi / 365),
            math.cos(day * 2 * math.pi / 365),
            math.sin(hour * 2 * math.pi / 24),
            math.cos(hour * 2 * math.pi / 24),
            ssn / 100.0 - 1.0,
        ])
        target = target.to(dm.device)
        encoded_target = dm.param_encoder(target).unsqueeze(0).repeat(num_samples, 1)
        null_target = torch.zeros_like(encoded_target)

        x = torch.randn((num_samples, 4, 24, 48), device=dm.device)

        for i, t in tqdm(enumerate(scheduler.timesteps)):
            torch.compiler.cudagraph_mark_step_begin()
            model_input = scheduler.scale_model_input(x, t)
            with torch.no_grad():
                noise_pred_guided = dm.model(model_input, t, class_labels=encoded_target).sample
                noise_pred_unguided = dm.model(model_input, t, class_labels=null_target).sample
                noise_pred = noise_pred_unguided + guidance_scale * (noise_pred_guided - noise_pred_unguided)
            x = x.detach().requires_grad_()
            x = scheduler.step(noise_pred, t, x).prev_sample

        outs = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x)).sample)
        outs = outs[..., :181, :361].flip((2,))

        image_grid = tv.utils.make_grid(outs, nrow=math.ceil(math.sqrt(num_samples)))

        filename = f"out/generated-{qhour:02d}.png"
        tv.utils.save_image(image_grid, filename)

        ensemble = torch.quantile(outs, 0.5, dim=0)
        tv.utils.save_image(ensemble, f"out/ensemble-{qhour:02d}.png")

        stdev = torch.std(outs, dim=0)
        stdev = (stdev * 100.).clip(0, 1)
        tv.utils.save_image(stdev, f"out/stdev-{qhour:02d}.png")

if __name__ == "__main__":
    jsonargparse.CLI(main)
