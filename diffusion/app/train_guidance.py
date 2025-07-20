import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from models import GuidanceModel, IRIData

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = GuidanceModel.load_from_checkpoint(sys.argv[1])
    else:
        model = GuidanceModel()

    data = IRIData("combined", train_batch=128)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename='guidance-' +
                                          '-{v_num}-{epoch}-{val_loss:.2g}', save_top_k=1, monitor="val_loss", mode="min")
    trainer = L.Trainer(max_epochs=250,
                        precision="bf16-mixed", callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger("lightning_logs", name="guidance"))
    trainer.fit(model, data)
