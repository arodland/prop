import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from models import DiTDiffusionModel, IRIData
import diffusers


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = DiTDiffusionModel.load_from_checkpoint(sys.argv[1])
        # Don't load VAE from checkpoint
        model.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
    else:
        model = DiTDiffusionModel(pred_type='v_prediction')

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename='dit-diffusion-v-' +
                                          '{epoch}-{val_loss:.2g}', save_top_k=1, monitor="val_loss", mode="min")
    data = IRIData("combined", train_batch=250, add_noise=0.0001)
    trainer = L.Trainer(max_epochs=250,
                        log_every_n_steps=50,
                        # accumulate_grad_batches=4,
                        precision="32-true",
                        callbacks=[
                            checkpoint_callback,
                            ModelCheckpoint(dirpath="checkpoints", filename='dit-diffusion-averaged'),
                            StochasticWeightAveraging(swa_lrs=1e-5),
                        ],
                        logger=TensorBoardLogger("lightning_logs", name="dit-diffusion"))
    trainer.fit(model, data)
