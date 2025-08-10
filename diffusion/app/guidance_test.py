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

            pred_toy = torch.atan2(pred[:, 1], pred[:, 2]) / (2 * math.pi)
            target_toy = torch.atan2(target[:, 1], target[:, 2]) / (2 * math.pi)
            pred_tod = torch.atan2(pred[:, 3], pred[:, 4]) / (2 * math.pi)
            target_tod = torch.atan2(target[:, 3], target[:, 4]) / (2 * math.pi)
            pred_year = int(round((pred[:, 0] * 50. + 2000 - pred_toy).item()))
            target_year = int(round((target[:, 0] * 50. + 2000 - target_toy).item()))

            pred_base = datetime.datetime(year=pred_year, month=1, day=1)
            target_base = datetime.datetime(year=target_year, month=1, day=1)
            pred_date_str = (pred_base + datetime.timedelta(days=pred_toy.item() * 365)).strftime("%Y-%m-%d")
            target_date_str = (target_base + datetime.timedelta(days=target_toy.item() * 365)).strftime("%Y-%m-%d")
            pred_tod_str = (pred_base + datetime.timedelta(hours=pred_tod.item() * 24)).strftime("%H:%M")
            target_tod_str = (target_base + datetime.timedelta(hours=target_tod.item() * 24)).strftime("%H:%M")

            pred_ssn = pred[:, 4] * 100 + 100
            target_ssn = target[:, 4] * 100 + 100

            print(f"Predicted: {pred_date_str} {pred_tod_str} SSN {pred_ssn.item():.2f}, "
                  f"Target: {target_date_str} {target_tod_str} SSN {target_ssn.item():.2f}, "
                  f"Loss: {loss.item():.4f}")
            if i >= 10:
                break
