import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 s_patch: int = 2,
                 n_coder: int = 6,
                 s_kernel: int = 3,
                 s_data_y: [int] = None
                 ):
        super(Decoder, self).__init__()
        # Initialization:
        self.s_data_y = s_data_y if s_data_y is not None else [10, 1, 64, 64]

        # Convolution:
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=s_kernel, padding='same'),
                nn.GroupNorm(num_groups=1, num_channels=d_model), nn.SiLU(inplace=True),
            ) for _ in range(n_coder)
        ])

        # Up Sample:
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, kernel_size=s_kernel, stride=2,
                               padding=((s_kernel - 1) // 2), output_padding=1),
            nn.GroupNorm(num_groups=1, num_channels=d_model), nn.SiLU(inplace=True),
        )

        # Final Convolution:
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=s_data_y[1] * s_patch * s_patch, kernel_size=s_kernel,
                      padding='same'),
            nn.GroupNorm(num_groups=1, num_channels=s_data_y[1] * s_patch * s_patch), nn.SiLU(inplace=True),
        )

        # Transform Patch & PixelShuffle:
        self.patch = nn.PixelShuffle(upscale_factor=s_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,D,T,H,W]
        B, _, T, _, _ = x.shape
        for layer in self.conv:
            x = layer(x)
        x = x.transpose(1, 2)
        x = x.reshape(-1, *x.shape[2:])
        x = self.up_sample(x)
        x = self.final_conv(x)
        x = x.reshape(B, T, *x.shape[1:])
        x = self.patch(x)
        return x


if __name__ == '__main__':
    import time

    test_step = 600


    def test():
        test_module = Decoder(d_model=128, s_patch=2, n_coder=6, s_kernel=3, s_data_y=[10, 1, 64, 64]).cuda()
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
