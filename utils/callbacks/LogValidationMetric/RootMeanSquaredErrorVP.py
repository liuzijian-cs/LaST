import torch
import torchmetrics


class RootMeanSquaredErrorVP(torchmetrics.Metric):

    def __init__(self):
        super(RootMeanSquaredErrorVP, self).__init__()
        self.add_state("sum_squared_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, true):
        self.sum_squared_mse += torch.square(pred - true).sum()
        self.total += pred.size(0) * pred.size(1) if pred.dim() == 5 else pred.size(0)

    def compute(self):
        return torch.sqrt(self.sum_squared_mse / self.total)
