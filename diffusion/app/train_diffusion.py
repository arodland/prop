import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from models import DiffusionModel, IRIData

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = DiffusionModel.load_from_checkpoint(sys.argv[1])
    else:
        model = DiffusionModel()

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename='diffusion-' +
                                          '{epoch}-{val_loss:.2g}', save_top_k=1, monitor="val_loss", mode="min")
    data = IRIData("combined", train_batch=8)
    trainer = L.Trainer(max_epochs=30,
                        log_every_n_steps=10,
                        # accumulate_grad_batches=4,
                        precision="bf16-mixed", callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger("lightning_logs", name="diffusion"))
    trainer.fit(model, data)
