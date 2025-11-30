import torch
import torch.nn as nn


class Translator(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 n_layer: int = 6,
                 s_kernel: int = 3,
                 len_back: int = 10,
                 len_pred: int = 10,
                 ):
        super(Translator, self).__init__()
        # First Convolution:
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=len_back * d_model, out_channels=d_model * 2, kernel_size=s_kernel,
                      padding='same'),
            nn.GroupNorm(num_groups=1, num_channels=d_model * 2), nn.SiLU(inplace=True)
        )

        # Convolution:
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=d_model * 2, out_channels=d_model * 2, kernel_size=s_kernel,
                          padding='same'),
                nn.GroupNorm(num_groups=1, num_channels=d_model * 2), nn.SiLU(inplace=True)
            ) for _ in range(n_layer)
        ])

        # Final Convolution:
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model * 2, out_channels=len_pred * d_model, kernel_size=s_kernel,
                      padding='same'),
            nn.GroupNorm(num_groups=1, num_channels=len_pred * d_model), nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,D,T,H,W]
        B, D, T, _, _ = x.shape
        x = self.first_conv(x.reshape(B, -1, *x.shape[3:]))  # [B,D,T,H,W] > [B,DT,H,W]  > first conv
        for layer in self.conv:
            x = layer(x)
        x = self.final_conv(x)
        return x.reshape(B, D, -1, *x.shape[2:])  # [B,D,T,H,W]


if __name__ == '__main__':
    import time

    test_step = 600


    def test():
        test_module = Translator(d_model=128,  n_layer=6, s_kernel=3, len_back=10, len_pred=12).cuda()
        start_time = time.time()
        test_output = test_input = None
        for _ in range(test_step):
            test_input = torch.randn(32, 128, 10, 16, 16).cuda()
            test_output = test_module(test_input)
        time_cost = time.time() - start_time
        print(f"Test Result:")
        print(f"Time cost: {time_cost:.6f}s (in {test_step} steps), Average time cost: {time_cost / test_step:.6f}s")
        print(f"Input data: {test_input.shape}, Output data: {test_output.shape}")


    test()