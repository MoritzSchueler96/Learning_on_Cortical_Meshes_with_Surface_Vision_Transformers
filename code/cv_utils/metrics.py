import torchmetrics
import torch
from sklearn.metrics import balanced_accuracy_score

class RMSE(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape

        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors.float() / self.n_observations)


class BalancedAccuracy(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("y_hat", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("y", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape[0] == target.shape[0]

        y_hat = torch.argmax(preds, dim=1)
        self.y_hat = y_hat.detach().cpu()
        self.y = target.detach().cpu()

    def compute(self):
        return balanced_accuracy_score(self.y, self.y_hat)
