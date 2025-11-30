import lightning as L
from typing import override
from lightning.pytorch.callbacks import Callback


class LogLearningRate(Callback):
    def __init__(self):
        super().__init__()

    @override
    def on_validation_epoch_end(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
    ) -> None:
        pl_module.log('lr', trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True,
                      sync_dist=True)
