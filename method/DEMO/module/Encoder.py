import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 s_patch: int = 2,
                 n_coder: int = 6,
                 s_kernel: int = 3,
                 s_data_x: [int] = None,
                 ):
        super(Encoder, self).__init__()
        # Initialization:
        self.s_data_x = s_data_x if s_data_x is not None else [10, 1, 64, 64]

        # Transform Patch & PixelShuffle:
        self.patched = nn.PixelUnshuffle(downscale_factor=s_patch)

        # Initial Convolution:
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=s_data_x[1] * s_patch * s_patch, out_channels=d_model, kernel_size=s_kernel,
                      padding='same'),
            nn.GroupNorm(num_groups=1, num_channels=d_model), nn.SiLU(inplace=True),
        )
        # Down Sample:
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=s_kernel, stride=2,
                      padding=(s_kernel - 1) // 2),
            nn.GroupNorm(num_groups=1, num_channels=d_model), nn.SiLU(inplace=True),
        )

        # Convolution:
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=s_kernel, padding='same'),
                nn.GroupNorm(num_groups=1, num_channels=d_model), nn.SiLU(inplace=True),
            ) for _ in range(n_coder)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,c,h,w]
        B, T, _, _, _ = x.shape
        x = self.patched(x)  # [B,T,c,h,w] > [B,T,c*p*p,h/p,w/p] = [B,T,C,H,W]
        x = self.first_conv(x.reshape(-1, *x.shape[2:]))  # [B,T,C,H,W] > [BT,C,H,W] > self.first_conv
        x = self.down_sample(x)
        x = x.reshape(B, T, *x.shape[1:]).transpose(1, 2)  # [BT,D,H,W] > [B,D,T,H,W]
        for layer in self.conv:
            x = layer(x)
        return x  # [B,D,T,H,W]


if __name__ == '__main__':
    import time

    test_step = 600


    def test():
        test_module = Encoder(d_model=128, s_patch=2, n_coder=6, s_kernel=3, s_data_x=[10, 1, 64, 64]).cuda()
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
