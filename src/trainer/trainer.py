import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model


class Trainer:
    def __init__(self, model: Model, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_loop(self, dataloader: DataLoader, device: torch.device) -> None:
        self.model.to(device)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def test_loop(self, dataloader: DataLoader, device: torch.device):
        self.model.to(device)
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

                pred_class = torch.argmax(pred, dim=1)
                true_class = torch.argmax(y, dim=1)
                correct += (pred_class == true_class).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    def train(self, batch_size: int, train_dl: DataLoader, test_dl: DataLoader, epochs: int, device: torch.device) -> None:
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_dl, batch_size, device)
            self.test_loop(test_dl, device)
        # print("Done!")

    def save(self, path: str) -> None:
        self.model.save(path)
