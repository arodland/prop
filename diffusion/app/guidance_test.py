import sys
import math
import datetime
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from models import GuidanceModel, IRIData

if __name__ == "__main__":
    model = GuidanceModel.load_from_checkpoint(sys.argv[1])

    ds = IRIData("combined", test_batch=1)
    ds.setup()
    data = ds.test_dataloader()

    with torch.no_grad():
        for i, sample in enumerate(data):
            images = sample["images"].to(device=model.device)
            pred = model.model(images)
            target = sample["target"].to(device=model.device)
            loss = F.mse_loss(pred, target)

            pred_toy = torch.atan2(pred[:, 0], pred[:, 1]) / (2 * math.pi)
            target_toy = torch.atan2(target[:, 0], target[:, 1]) / (2 * math.pi)
            pred_tod = torch.atan2(pred[:, 2], pred[:, 3]) / (2 * math.pi)
            target_tod = torch.atan2(target[:, 2], target[:, 3]) / (2 * math.pi)

            dt_base = datetime.datetime(year=2025, month=1, day=1)
            pred_toy_str = (dt_base + datetime.timedelta(days=pred_toy.item() * 365)).strftime("%m-%d")
            target_toy_str = (dt_base + datetime.timedelta(days=target_toy.item() * 365)).strftime("%m-%d")
            pred_tod_str = (dt_base + datetime.timedelta(hours=pred_tod.item() * 24)).strftime("%H:%M")
            target_tod_str = (dt_base + datetime.timedelta(hours=target_tod.item() * 24)).strftime("%H:%M")

            pred_ssn = pred[:, 4] * 100 + 100
            target_ssn = target[:, 4] * 100 + 100

            print(f"Predicted: XXXX-{pred_toy_str} {pred_tod_str} SSN {pred_ssn.item():.2f}, "
                f"Target: XXXX-{target_toy_str} {target_tod_str} SSN {target_ssn.item():.2f}, "
                f"Loss: {loss.item():.4f}")
            if i >= 10:
                break
