import torch
import lightning as L
from utils import Color as Co, printf
from typing_extensions import override
from lightning.pytorch.callbacks import Callback
from fvcore.nn import FlopCountAnalysis, flop_count_table


class CheckFLIOS(Callback):
    def __init__(self, data_back, model):
        self.data_back = data_back
        self.test_model = model

    @override
    def on_sanity_check_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Called when the validation sanity check ends."""
        flops = FlopCountAnalysis(self.test_model, torch.rand(1, *self.data_back).to(pl_module.device))
        print(f"FLOPS:\n {flop_count_table(flops)}")
