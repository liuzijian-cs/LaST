import torch
import torch.nn as nn
import lightning as L

from .module import Encoder, Translator, Decoder


class DEMO(L.LightningModule):
    def __init__(self,
                 d_model: int = 128,
                 s_patch: int = 2,
                 n_coder: int = 6,
                 n_layer: int = 6,
                 s_kernel: int = 3,
                 s_data_x: [int] = None,
                 s_data_y: [int] = None,
                 lr: float = 1e-3,
                 ):
        super(DEMO, self).__init__()

        self.lr = lr

        self.encoder = Encoder(d_model=d_model, s_patch=s_patch, n_coder=n_coder, s_kernel=s_kernel, s_data_x=s_data_x)
        self.translator = Translator(d_model=d_model, n_layer=n_layer, s_kernel=s_kernel,
                                     len_back=s_data_x[0], len_pred=s_data_y[0])
        self.decoder = Decoder(d_model=d_model, s_patch=s_patch, n_coder=n_coder, s_kernel=s_kernel, s_data_y=s_data_y)
        self.example_input_array = torch.rand(1, *s_data_x)

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.translator(x)
        x = self.decoder(x)
        return x

    def shared_step(self, batch: torch.Tensor):
        seq_back, seq_true = batch
        seq_pred = self(seq_back)
        loss = self.criterion(seq_pred, seq_true)
        return loss, seq_pred, seq_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, seq_pred, seq_true = self.shared_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return {  # Callback: LogValidationMetric need 'seq_pred' and 'seq_true'
            'loss': loss,
            'seq_pred': seq_pred,
            'seq_true': seq_true,
        }

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }



if __name__ == '__main__':
    import time

    test_step = 600


    def test():
        test_module = DEMO(lr=1e-3, d_model=128, s_patch=2, n_coder=6, n_layer=6, s_kernel=3, s_data_x=[10, 1, 64, 64],
                           s_data_y=[10, 1, 64, 64]).cuda()
        start_time = time.time()
        test_output = test_input = None
        for _ in range(test_step):
            test_input = torch.randn(32, 10, 1, 64, 64).cuda()
            test_output = test_module(test_input)
        time_cost = time.time() - start_time
        print(f"Test Result:")
        print(f"Time cost: {time_cost:.6f}s (in {test_step} steps), Average time cost: {time_cost / test_step:.6f}s")
        print(f"Input data: {test_input.shape}, Output data: {test_output.shape}")


    test()
