import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from models import DiffusionModel, IRIData
import diffusers


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = DiffusionModel.load_from_checkpoint(sys.argv[1])
        # Don't load VAE from checkpoint
        model.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
    else:
        model = DiffusionModel()

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename='vdiffusion-' +
                                          '{epoch}-{val_loss:.2g}', save_top_k=1, monitor="val_loss", mode="min")
    data = IRIData("combined", train_batch=64, add_noise=0.001)
    trainer = L.Trainer(max_epochs=100,
                        log_every_n_steps=10,
                        # accumulate_grad_batches=4,
                        precision="bf16-mixed",
                        callbacks=[
                            checkpoint_callback,
                            ModelCheckpoint(dirpath="checkpoints", filename='vdiffusion-averaged'),
                            StochasticWeightAveraging(swa_lrs=1e-5),
                        ],
                        logger=TensorBoardLogger("lightning_logs", name="vdiffusion"))
    trainer.fit(model, data)
