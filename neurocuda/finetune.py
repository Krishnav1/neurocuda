"""Post-conversion SNN fine-tuning with surrogate gradients."""
import torch, torch.nn as nn
from snntorch import utils as snn_utils


class FineTuner:
    """Fine-tune a converted SNN for 1-5 epochs to close accuracy gap."""

    def __init__(self, snn_model, train_loader, device="cuda"):
        self.snn = snn_model
        self.loader = train_loader
        self.device = device
        self.history = []

    def train(self, epochs=3, lr_schedule=None):
        """Fine-tune for `epochs` epochs. Returns best accuracy."""
        if lr_schedule is None:
            lr_schedule = [1e-5, 5e-6, 1e-6][:epochs]

        best_state = None
        best_acc = 0

        for epoch in range(epochs):
            lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
            opt = torch.optim.AdamW(self.snn.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            self.snn.train()

            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                opt.zero_grad()
                loss = crit(self.snn(data), target)
                loss.backward()
                opt.step()
                snn_utils.reset(self.snn)

        return self.snn


def finetune(snn_model, train_loader, epochs=3, device="cuda", lr_schedule=None):
    """
    Fine-tune a converted SNN.

    Args:
        snn_model: Converted SNN model
        train_loader: DataLoader with training data
        epochs: Number of fine-tuning epochs (default 3)
        device: "cuda" or "cpu"

    Returns:
        Fine-tuned SNN model
    """
    ft = FineTuner(snn_model, train_loader, device)
    ft.train(epochs, lr_schedule)
    return ft.snn
