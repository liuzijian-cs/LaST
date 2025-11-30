import torch
import torchmetrics
class MeanSquaredErrorVP(torchmetrics.Metric):
    """
    Compute mean squared error (MSE) for Video Prediction & Spatio-Temporal Prediction:
        Calculate MSE over axis=(0, 1) for tensors shaped (B, T, C, H, W).
        Calculate MSE over axis=(0) for tensors shaped (B * T, C, H, W).
        This computation averages the errors across batch samples (B) and temporal (T) dimensions.
        The result provides a separate average error for each image.
    计算视频预测和时空预测的均方误差 (MSE)：
        对于形状为 (B, T, C, H, W) 的张量，在轴 (0, 1) 上计算 MSE；对于形状为 (B * T, C, H, W) 的张量，在轴 (0) 上计算 MSE。
        这样计算出的 MSE 将对批样本 (B) 和时间维度 (T) 求平均误差。
        结果会提供每张图像的单独平均误差。
    input: tensor of shape (B, T, C, H, W) or tensor of shape (B * T, C, H, W)
    numpy: same as numpy method: np.mean((pred-true)**2,axis=(0,1)).sum() << (B, T, C, H, W)
    """

    def __init__(self):
        super(MeanSquaredErrorVP, self).__init__()
        self.add_state("sum_squared_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, true):
        self.sum_squared_mse += torch.square(pred - true).sum()
        self.total += pred.size(0) * pred.size(1) if pred.dim() == 5 else pred.size(0)

    def compute(self):
        return self.sum_squared_mse / self.total