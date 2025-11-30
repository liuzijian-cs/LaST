import lightning as L
from utils import Color as Co, printf
from typing_extensions import override
from lightning.pytorch.callbacks import Callback


class PrintConfig(Callback):
    def __init__(self, args):
        self.args = args

    @override
    def on_sanity_check_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Called when the validation sanity check ends."""
        printf(s="PrintConfig Callback", m=f"{Co.G}Sanity Checking Successful! Ready to start training >>>>>> {Co.RE}")
        printf(s="PrintConfig",
               m=f"{Co.B}Data: {Co.C}{self.args.data}{Co.B} [{Co.Y}{self.args.data_back} {Co.B}>>> {Co.Y}{self.args.data_pred}{Co.B}], train({Co.Y}{1 - self.args.data_val_rate - self.args.data_test_rate:.2f}{Co.B})/valid({Co.Y}{self.args.data_val_rate:.2f}{Co.B})/test({Co.Y}{self.args.data_test_rate:.2f}{Co.B}), Mean=[{Co.Y}{self.args.data_mean}{Co.B}], Std=[{Co.Y}{self.args.data_std}{Co.B}]{Co.RE}.")

        printf(s="PrintConfig",
               m=f"{Co.B}Model: {Co.C}{self.args.model}{Co.B} [dim{Co.Y}{self.args.d_model}{Co.B}_layer{Co.Y}{self.args.n_layer}{Co.B}_ffn{Co.Y}{self.args.r_forward}{Co.B}_dropout{Co.Y}{self.args.dropout}{Co.B}_droppath{Co.Y}{self.args.drop_path}{Co.B}_bias{Co.Y}{self.args.attn_bias}{Co.B}_patch{Co.Y}{self.args.s_patch}{Co.B}_heads{Co.Y}{self.args.n_heads}{Co.B}_kernel{Co.Y}{self.args.s_kernel}{Co.B}]{Co.RE}")

        printf(s="PrintConfig",
               m=f"{Co.B}Train: Epoch[{Co.Y}{self.args.epoch}{Co.B}], Learning Rate[{Co.Y}{self.args.lr}{Co.B}], LR Scheduler[{Co.Y}{self.args.lr_scheduler}{Co.B}], Batch[{Co.Y}{self.args.batch_size}{Co.B}], AccGrad[{Co.Y}{self.args.accumulate_grad_batches}{Co.B}], {Co.RE}")

        printf(s="PrintConfig",
               m=f"{Co.B}Trainer: Seed [{Co.Y}{self.args.seed}{Co.B}], Device [{Co.Y}{pl_module.device}{Co.B}], AMP [{Co.Y}{self.args.fp16}{Co.B}],  Worker[{Co.Y}{self.args.num_workers}{Co.B}], Valid Every[{Co.Y}{self.args.n_val_every}{Co.B}], Save Top[{Co.Y}{self.args.n_model_save}{Co.B}] {Co.RE}")
        printf(s="PrintConfig Callback",
               m=f"{Co.G}Start Training On {self.args.version_path} >>>>>> {Co.Y}Good Luck :) {Co.RE}")
